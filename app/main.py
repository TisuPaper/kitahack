# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model_loader import (
    champion, champion_preprocess,
    challenger_model, challenger_processor,
    device,
)
from app.forensic_features import compute_all_features

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
# Thresholds — TUNED on 40 FF++ C23 samples (20 real + 20 fake)
#
# Grid search result:
#   t_real=0.20, t_fake=0.55 → 90% accuracy, 100% coverage, 0 FP
#
# FaceForge p_fake on reals: all ≤ 0.153  (no overlap with threshold)
# FaceForge p_fake on fakes: 13/20 ≥ 0.638 (4 complete misses at ~0.00)
# =========================================================================
CONFIDENT_REAL = 0.20   # p_fake ≤ this → Real (champion only)
CONFIDENT_FAKE = 0.55   # p_fake ≥ this → Fake (champion only)
# Scale-aware disagreement: only flag uncertain in the true midrange
UNCERTAIN_LOW = 0.20    # only consider uncertain if p_fake > this
UNCERTAIN_HIGH = 0.55   # only consider uncertain if p_fake < this
DISAGREE_THRESH = 0.20  # |champ - challenger| > this AND in midrange → Uncertain
# Quality gate
MIN_FACE_SIZE = 80      # pixels


# ---- Logit-domain blending ----
def _logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _blend_logits(p1: float, p2: float, w1: float = 0.85, w2: float = 0.15) -> float:
    """Blend two probabilities in logit space, then convert back."""
    blended = w1 * _logit(p1) + w2 * _logit(p2)
    return _sigmoid(blended)


# ---- Reference statistics for forensic z-scores ----
REAL_REFERENCE = {
    "high_freq_ratio": 0.0052, "mean_magnitude": 6.2670,
    "noise_std": 5.0216, "noise_uniformity": 0.5855,
    "ela_mean": 1.4007, "ela_std": 0.9984,
    "ela_max_deviation": 0.5969, "laplacian_variance": 256.4983,
}
REAL_STD = {
    "high_freq_ratio": 0.0034, "mean_magnitude": 0.2965,
    "noise_std": 2.6479, "noise_uniformity": 0.0960,
    "ela_mean": 0.2844, "ela_std": 0.2191,
    "ela_max_deviation": 0.2706, "laplacian_variance": 283.5662,
}


def compute_deviations(features: dict) -> dict:
    deviations = {}
    for key in REAL_REFERENCE:
        if key in features:
            std = REAL_STD[key]
            deviations[key] = round((features[key] - REAL_REFERENCE[key]) / std, 3) if std > 0 else 0.0
    return deviations


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

    # ====== STEP 1: Champion (FaceForge XceptionNet) ======
    champ_real, champ_fake = _run_champion(img)

    # ====== STEP 2: Decision logic ======
    challenger_used = False
    chall_real, chall_fake = None, None
    decision_path = "champion_confident"

    if not quality_ok:
        # Quality gate failed → always uncertain unless champion is very confident
        if champ_fake >= 0.95:
            label = "Fake"
            final_real, final_fake = champ_real, champ_fake
            decision_path = "low_quality_but_confident"
        elif champ_fake <= 0.05:
            label = "Real"
            final_real, final_fake = champ_real, champ_fake
            decision_path = "low_quality_but_confident"
        else:
            label = "Uncertain"
            final_real, final_fake = champ_real, champ_fake
            decision_path = "low_quality_uncertain"
            warnings.append("Input quality too low for reliable detection.")

    elif champ_fake >= CONFIDENT_FAKE:
        label = "Fake"
        final_real, final_fake = champ_real, champ_fake

    elif champ_fake <= CONFIDENT_REAL:
        label = "Real"
        final_real, final_fake = champ_real, champ_fake

    else:
        # ====== Uncertain zone → call challenger ======
        challenger_used = True
        chall_real, chall_fake = _run_challenger(img)
        disagreement = abs(champ_fake - chall_fake)

        # Scale-aware: only go Uncertain if truly in midrange AND models disagree
        in_midrange = UNCERTAIN_LOW < champ_fake < UNCERTAIN_HIGH

        if in_midrange and disagreement > DISAGREE_THRESH:
            label = "Uncertain"
            # Logit blend for the reported probability
            final_fake = _blend_logits(champ_fake, chall_fake)
            final_real = 1 - final_fake
            decision_path = "models_disagree"
            warnings.append(
                f"Models disagree in uncertain zone "
                f"(FaceForge={champ_fake:.1%} fake, ViT={chall_fake:.1%} fake)."
            )
        else:
            # Models agree (or champion is leaning one way) → logit blend
            final_fake = _blend_logits(champ_fake, chall_fake)
            final_real = 1 - final_fake
            label = "Fake" if final_fake > 0.5 else "Real"
            decision_path = "challenger_consulted"

    confidence = round(max(final_real, final_fake), 4)

    # ====== FORENSIC FEATURES ======
    features = compute_all_features(img)
    deviations = compute_deviations(features)
    anomaly_flags = sum(1 for v in deviations.values() if abs(v) > 1.5)

    # ====== BUILD RESPONSE ======
    response = {
        "prediction": label,
        "confidence": confidence,
        "probabilities": {
            "real": round(final_real, 4),
            "fake": round(final_fake, 4),
        },
        "face_found": face_found,
        "decision": {
            "path": decision_path,
            "champion": {
                "model": "FaceForge-XceptionNet",
                "real": round(champ_real, 4),
                "fake": round(champ_fake, 4),
            },
            "thresholds": {
                "confident_real": CONFIDENT_REAL,
                "confident_fake": CONFIDENT_FAKE,
            },
            "calibration": {
                "samples": 40,
                "accuracy": "90.0%",
                "false_positive_rate": "0.0%",
                "coverage": "100%",
            },
        },
        "forensic_analysis": {
            "features": features,
            "deviations_from_real": deviations,
            "anomaly_flags": f"{anomaly_flags}/{len(deviations)}",
        },
        "warnings": warnings,
    }

    if challenger_used:
        response["decision"]["challenger"] = {
            "model": "prithivMLmods-ViT",
            "real": round(chall_real, 4),
            "fake": round(chall_fake, 4),
        }

    return response


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

    # Champion
    champ_real, champ_fake = _run_champion(analysis_img)

    # Decision
    challenger_used = False
    chall_real, chall_fake = None, None

    if not quality_ok:
        if champ_fake >= 0.95:
            label, final_real, final_fake = "Fake", champ_real, champ_fake
        elif champ_fake <= 0.05:
            label, final_real, final_fake = "Real", champ_real, champ_fake
        else:
            label, final_real, final_fake = "Uncertain", champ_real, champ_fake
    elif champ_fake >= CONFIDENT_FAKE:
        label, final_real, final_fake = "Fake", champ_real, champ_fake
    elif champ_fake <= CONFIDENT_REAL:
        label, final_real, final_fake = "Real", champ_real, champ_fake
    else:
        challenger_used = True
        chall_real, chall_fake = _run_challenger(analysis_img)
        disagreement = abs(champ_fake - chall_fake)
        in_midrange = UNCERTAIN_LOW < champ_fake < UNCERTAIN_HIGH

        if in_midrange and disagreement > DISAGREE_THRESH:
            label = "Uncertain"
            final_fake = _blend_logits(champ_fake, chall_fake)
            final_real = 1 - final_fake
        else:
            final_fake = _blend_logits(champ_fake, chall_fake)
            final_real = 1 - final_fake
            label = "Fake" if final_fake > 0.5 else "Real"

    # Forensics
    features = compute_all_features(analysis_img)
    deviations = compute_deviations(features)
    anomaly_flags = sum(1 for v in deviations.values() if abs(v) > 1.5)

    result = {
        "prediction": label,
        "confidence": round(max(final_real, final_fake), 4),
        "p_fake": round(final_fake, 4),
        "face_found": face_found,
        "champion_fake": round(champ_fake, 4),
        "anomaly_flags": anomaly_flags,
    }

    if challenger_used:
        result["challenger_fake"] = round(chall_fake, 4)

    return result


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

    # ---- Aggregate ----
    predictions = [r["prediction"] for r in frame_results]
    p_fakes = [r["p_fake"] for r in frame_results]
    confidences = [r["confidence"] for r in frame_results]

    fake_count = predictions.count("Fake")
    real_count = predictions.count("Real")
    uncertain_count = predictions.count("Uncertain")
    total = len(predictions)

    # Majority vote
    if fake_count > real_count and fake_count > uncertain_count:
        overall_label = "Fake"
    elif real_count > fake_count and real_count > uncertain_count:
        overall_label = "Real"
    else:
        overall_label = "Uncertain"

    avg_p_fake = sum(p_fakes) / len(p_fakes)
    avg_confidence = sum(confidences) / len(confidences)

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
        "vote": {
            "fake": fake_count,
            "real": real_count,
            "uncertain": uncertain_count,
        },
        "temporal_consistency": {
            "p_fake_mean": round(avg_p_fake, 4),
            "p_fake_std": round(p_fake_std, 4),
        },
        "per_frame": frame_results,
        "calibration": {
            "samples": 40,
            "accuracy": "90.0%",
            "false_positive_rate": "0.0%",
        },
        "warnings": warnings,
    }