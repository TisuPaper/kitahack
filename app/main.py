# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model_loader import (
    champion, champion_preprocess,
    challenger_model, challenger_processor,
    fallback_model, fallback_processor,
    device,
)

from PIL import Image
import io
import math
import torch
import cv2
import numpy as np


app = FastAPI(title="Deepfake Detector API")

# ---- CORS (allow Flutter web app to call API) ----
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve static files (live detection page) ----
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/live")
async def live_detection_page():
    return FileResponse(os.path.join(_static_dir, "live.html"))


# ---- Face detector (Haar cascade) ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_largest_face(pil_img: Image.Image) -> tuple[Image.Image, bool, dict]:
    """Returns (cropped_image, face_found, face_meta)."""
    rgb = np.array(pil_img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return pil_img, False, {}

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
    )

    if len(faces) == 0:
        return pil_img, False, {}

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    img_area = rgb.shape[0] * rgb.shape[1]

    pad = int(0.25 * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)

    face_meta = {
        "face_width": int(w),
        "face_height": int(h),
        "face_area_ratio": round((w * h) / max(img_area, 1), 4),
    }
    return Image.fromarray(rgb[y1:y2, x1:x2]), True, face_meta


# =========================================================================
# Ensemble parameters — TUNED via grid search on 80 FF++ C23 videos
# (40 real + 10 each of Deepfakes/Face2Face/FaceSwap/NeuralTextures)
#
# Strategy: Logit stacking — each model's logit(p_fake) is weighted
# and a learned bias corrects for miscalibration across models.
#
# Best config (68.8% balanced video accuracy, 65% real / 72.5% fake):
#   sigmoid(0.25*logit(champ) + 1.0*logit(chall) + 0.0*logit(fall) + 2.5)
#
# Individual model profiles:
#   Champion (XceptionNet): Excellent at reals (97%), good at Deepfakes,
#                           weak on Face2Face/FaceSwap/NeuralTextures
#   Challenger (ViT):       Most balanced across all forgery types
#   Fallback (EfficientNet): Currently not contributing (weight=0)
# =========================================================================
STACKING_WEIGHT_CHAMPION = 0.25
STACKING_WEIGHT_CHALLENGER = 1.0
STACKING_WEIGHT_FALLBACK = 0.0
STACKING_BIAS = 2.5

MIN_FACE_SIZE = 80  # pixels


# ---- Logit-domain stacking ----
def _logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1 / (1 + math.exp(-x))


def _stacking_blend(champ_fake: float, chall_fake: float, fall_fake: float) -> float:
    """Compute final p_fake via learned logit stacking."""
    z = (STACKING_WEIGHT_CHAMPION * _logit(champ_fake)
         + STACKING_WEIGHT_CHALLENGER * _logit(chall_fake)
         + STACKING_WEIGHT_FALLBACK * _logit(fall_fake)
         + STACKING_BIAS)
    return _sigmoid(z)



def _run_champion(img: Image.Image) -> tuple[float, float]:
    """Returns (real_prob, fake_prob) from FaceForge."""
    x = champion_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(champion(x), dim=1)
    return float(probs[0][0]), float(probs[0][1])


def _run_challenger(img: Image.Image) -> tuple[float, float]:
    """Returns (real_prob, fake_prob) from prithivMLmods ViT."""
    inputs = challenger_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(challenger_model(**inputs).logits, dim=1)

    id2label = challenger_model.config.id2label
    label_map = {"Realism": "real", "Deepfake": "fake"}
    rp, fp = 0.0, 0.0
    for idx, ln in id2label.items():
        c = label_map.get(ln, ln.lower())
        pv = float(probs[0][int(idx)])
        if c == "real": rp = pv
        elif c == "fake": fp = pv
    return rp, fp


def _run_fallback(img: Image.Image) -> tuple[float, float]:
    """Returns (real_prob, fake_prob) from EfficientNet fallback."""
    inputs = fallback_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(fallback_model(**inputs).logits, dim=1)

    id2label = fallback_model.config.id2label
    label_map = {"Realism": "real", "Deepfake": "fake", "Real": "real", "Fake": "fake"}
    rp, fp = 0.0, 0.0
    for idx, ln in id2label.items():
        c = label_map.get(ln, ln.lower())
        pv = float(probs[0][int(idx)])
        if c == "real": rp = pv
        elif c == "fake": fp = pv
    return rp, fp


@app.get("/")
def root():
    return {"message": "Deepfake Detector API is running. Go to /docs to test."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    warnings = []
    orig_w, orig_h = img.size

    # ====== QUALITY GATE ======
    if orig_w < 128 or orig_h < 128:
        warnings.append(f"Low resolution ({orig_w}×{orig_h}). Results may be less reliable.")

    img, face_found, face_meta = crop_largest_face(img)

    quality_ok = True
    if not face_found:
        quality_ok = False
        warnings.append("No face detected — running on full image. Accuracy is significantly lower.")
    elif face_meta.get("face_width", 0) < MIN_FACE_SIZE or face_meta.get("face_height", 0) < MIN_FACE_SIZE:
        quality_ok = False
        warnings.append(
            f"Small face ({face_meta['face_width']}×{face_meta['face_height']}px). "
            "Results may be less reliable."
        )

    # ====== MULTI-MODEL ENSEMBLE (Logit Stacking) ======
    champ_real, champ_fake = _run_champion(img)
    chall_real, chall_fake = _run_challenger(img)

    fall_real, fall_fake = 0.0, 0.0
    if fallback_model is not None:
        fall_real, fall_fake = _run_fallback(img)

    final_fake = _stacking_blend(champ_fake, chall_fake, fall_fake)
    final_real = 1.0 - final_fake
    label = "Fake" if final_fake > 0.5 else "Real"
    confidence = round(max(final_real, final_fake), 4)

    if not quality_ok:
        warnings.append("Input quality too low for reliable detection.")
        label = "Uncertain"

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": {
            "real": round(final_real, 4),
            "fake": round(final_fake, 4),
        },
        "face_found": face_found,
        "warnings": warnings,
    }


# =========================================================================
# VIDEO DETECTION
# =========================================================================

import tempfile
import os

MAX_FRAMES = 10  # Extract up to this many evenly-spaced frames


def _extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list[Image.Image]:
    """Extract evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # Pick evenly-spaced frame indices (skip first/last 10% for intro/outro)
    start = int(total_frames * 0.10)
    end = int(total_frames * 0.90)
    if end <= start:
        start, end = 0, total_frames

    n = min(max_frames, end - start)
    if n <= 0:
        cap.release()
        return []

    indices = [start + int(i * (end - start) / n) for i in range(n)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()
    return frames


def _analyze_single_frame(img: Image.Image) -> dict:
    """
    Run the full Champion-Challenger + forensic pipeline on a single frame.
    Returns a dict with prediction, confidence, probabilities, etc.
    """
    img_cropped, face_found, face_meta = crop_largest_face(img)

    quality_ok = True
    if not face_found:
        quality_ok = False
    elif face_meta.get("face_width", 0) < MIN_FACE_SIZE or face_meta.get("face_height", 0) < MIN_FACE_SIZE:
        quality_ok = False

    analysis_img = img_cropped

    # MULTI-MODEL ENSEMBLE (Logit Stacking)
    champ_real, champ_fake = _run_champion(analysis_img)
    chall_real, chall_fake = _run_challenger(analysis_img)

    fall_fake = 0.0
    if fallback_model is not None:
        _fr, fall_fake = _run_fallback(analysis_img)

    final_fake = _stacking_blend(champ_fake, chall_fake, fall_fake)
    final_real = 1.0 - final_fake

    if not quality_ok:
        label = "Uncertain"
    else:
        label = "Fake" if final_fake > 0.5 else "Real"

    return {
        "prediction": label,
        "confidence": round(max(final_real, final_fake), 4),
        "p_fake": round(final_fake, 4),
        "face_found": face_found,
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    # Validate: reject image uploads, allow video or unknown content types
    if file.content_type and file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a video file, not an image.")

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Write to temp file for OpenCV
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = _extract_frames(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")

    warnings = []

    # ---- Analyze each frame ----
    frame_results = []
    for i, frame in enumerate(frames):
        result = _analyze_single_frame(frame)
        result["frame_index"] = i
        frame_results.append(result)

    # ---- Aggregate (average p_fake across frames, better than majority vote) ----
    predictions = [r["prediction"] for r in frame_results]
    p_fakes = [r["p_fake"] for r in frame_results]
    confidences = [r["confidence"] for r in frame_results]

    fake_count = predictions.count("Fake")
    real_count = predictions.count("Real")
    uncertain_count = predictions.count("Uncertain")
    total = len(predictions)

    avg_p_fake = sum(p_fakes) / len(p_fakes)
    avg_confidence = sum(confidences) / len(confidences)

    # Use average p_fake for final decision (outperforms majority vote)
    if uncertain_count > total * 0.5:
        overall_label = "Uncertain"
    elif avg_p_fake > 0.5:
        overall_label = "Fake"
    else:
        overall_label = "Real"

    # Temporal consistency: high variance in p_fake across frames is suspicious
    p_fake_std = float(np.std(p_fakes))
    if p_fake_std > 0.25:
        warnings.append(
            f"High variance across frames (std={p_fake_std:.3f}). "
            "Inconsistent predictions may indicate partial manipulation."
        )

    # Face detection rate
    faces_found = sum(1 for r in frame_results if r["face_found"])
    if faces_found < total * 0.5:
        warnings.append(
            f"Face detected in only {faces_found}/{total} frames. "
            "Results may be less reliable."
        )

    return {
        "prediction": overall_label,
        "confidence": round(avg_confidence, 4),
        "probabilities": {
            "real": round(1 - avg_p_fake, 4),
            "fake": round(avg_p_fake, 4),
        },
        "frames_analyzed": total,
        "warnings": warnings,
    }


# =========================================================================
# AUDIO DETECTION (Multi-Model Ensemble)
# =========================================================================

from app.audio_model_loader import audio_models
from app.audio_inference import predict_ensemble

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    Analyze an audio file for deepfake voice detection.
    Uses 3 models (CNN-LSTM, TCN, TCN-LSTM) as an ensemble.
    """
    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. "
                   f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}",
        )

    if not audio_models:
        raise HTTPException(
            status_code=503,
            detail="No audio models loaded. Check models/ directory.",
        )

    # Save to temp file
    import tempfile
    suffix = ext or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict_ensemble(tmp_path, audio_models)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "type": "audio",
        **result,
    }