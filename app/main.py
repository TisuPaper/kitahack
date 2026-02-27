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
import time
import uuid
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

# ---- Permissions-Policy header (needed when /live is embedded in an iframe) ----
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class _PermissionsPolicyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Permissions-Policy"] = (
            "camera=*, microphone=*, display-capture=*, picture-in-picture=*"
        )
        return response

app.add_middleware(_PermissionsPolicyMiddleware)

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
    """Returns (cropped_image, face_found, face_meta).

    face_meta includes normalised bbox (0-1 fractions of original image size)
    so the frontend can draw the box regardless of display scale.
    """
    rgb = np.array(pil_img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return pil_img, False, {}

    img_h, img_w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
    )

    if len(faces) == 0:
        return pil_img, False, {}

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = int(0.25 * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)

    face_meta = {
        "face_width": int(w),
        "face_height": int(h),
        # Normalised crop box (with padding) — used by frontend to draw overlay
        "bbox": {
            "x": round(x1 / img_w, 4),
            "y": round(y1 / img_h, 4),
            "w": round((x2 - x1) / img_w, 4),
            "h": round((y2 - y1) / img_h, 4),
        },
    }
    return Image.fromarray(rgb[y1:y2, x1:x2]), True, face_meta


# =========================================================================
# Ensemble parameters — TUNED via grid search on 80 FF++ C23 videos
# (40 real + 10 each of Deepfakes/Face2Face/FaceSwap/NeuralTextures)
#
# Strategy: two-stage logit stacking with fallback tiebreaker.
#
# Stage 1 — Primary (Champion + Challenger only):
#   z = 0.25*logit(champ) + 1.0*logit(chall) + 2.5
#   p_fake = sigmoid(z)
#
# Stage 2 — Fallback tiebreaker (only when primary is uncertain):
#   If UNCERTAIN_LOW <= p_fake <= UNCERTAIN_HIGH, blend in Fallback:
#   z = 0.25*logit(champ) + 1.0*logit(chall) + 0.5*logit(fall) + 2.5
#
# Why: on Celeb-DF fakes, Champion+Challenger often land in 0.35-0.65;
# EfficientNet (trained on Celeb-DF) confidently says Fake, fixing the call.
# On FF++ reals misclassified as Fake, EfficientNet says Real, correcting it.
#
# Validated showcase:
#   Celeb-synthesis/id0_id4_0004.mp4 (FAKE):
#     primary avg=0.44 -> Real (wrong)
#     tiebreaker avg=0.71 -> Fake (correct, fallback p_fake ~0.92)
# =========================================================================
STACKING_WEIGHT_CHAMPION   = 0.25
STACKING_WEIGHT_CHALLENGER = 1.0
STACKING_WEIGHT_FALLBACK   = 0.5   # used only when primary is uncertain
STACKING_BIAS              = 2.5

UNCERTAIN_LOW  = 0.35   # primary p_fake in this band → call fallback tiebreaker
UNCERTAIN_HIGH = 0.65

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


def _stacking_blend(champ_fake: float, chall_fake: float, fall_fake: float) -> tuple[float, bool]:
    """Two-stage logit stacking with fallback tiebreaker.

    Stage 1: primary score from Champion + Challenger.
    Stage 2: if primary is uncertain (UNCERTAIN_LOW–UNCERTAIN_HIGH) AND
             a fallback model is loaded, blend in Fallback to break the tie.

    Returns (p_fake, tiebreaker_used).
    """
    z_primary = (STACKING_WEIGHT_CHAMPION   * _logit(champ_fake)
                 + STACKING_WEIGHT_CHALLENGER * _logit(chall_fake)
                 + STACKING_BIAS)
    p_primary = _sigmoid(z_primary)

    if UNCERTAIN_LOW <= p_primary <= UNCERTAIN_HIGH and fallback_model is not None:
        z_tiebreak = (STACKING_WEIGHT_CHAMPION   * _logit(champ_fake)
                      + STACKING_WEIGHT_CHALLENGER * _logit(chall_fake)
                      + STACKING_WEIGHT_FALLBACK   * _logit(fall_fake)
                      + STACKING_BIAS)
        return _sigmoid(z_tiebreak), True

    return p_primary, False


# ---- Confidence / verdict helpers ----

# Confidence band thresholds (calibrated on FF++ C23 eval data):
#   HIGH   — p_fake < 0.20 (confident real)  or p_fake > 0.80 (confident fake)
#   MEDIUM — p_fake in [0.20, 0.35) or (0.65, 0.80]  (model leans one way)
#   LOW    — p_fake in [0.35, 0.65]  (uncertain zone, tiebreaker territory)

def _confidence_band(p_fake: float) -> str:
    dist = abs(p_fake - 0.5)
    if dist > 0.30:
        return "HIGH"
    elif dist > 0.15:
        return "MEDIUM"
    return "LOW"


def _verdict(p_fake: float, quality_ok: bool) -> tuple[str, bool]:
    """Returns (verdict: REAL|FAKE|UNCERTAIN, uncertain: bool)."""
    if not quality_ok:
        return "UNCERTAIN", True
    if p_fake > 0.65:
        return "FAKE", False
    elif p_fake < 0.35:
        return "REAL", False
    return "UNCERTAIN", True


def _build_reasons_image(
    p_fake: float,
    face_found: bool,
    low_res: bool,
    small_face: bool,
    champ_fake: float,
    chall_fake: float,
    tiebreaker_used: bool,
) -> list[str]:
    reasons: list[str] = []
    if not face_found:
        reasons.append("no_face_detected")
    if low_res:
        reasons.append("low_resolution")
    if small_face:
        reasons.append("small_face")
    if tiebreaker_used:
        reasons.append("tiebreaker_used")
    disagreement = abs(champ_fake - chall_fake)
    if disagreement > 0.30:
        reasons.append("models_disagree")
    elif disagreement < 0.15:
        reasons.append("models_agree")
    if 0.35 <= p_fake <= 0.65:
        reasons.append("borderline_score")
    if p_fake < 0.20 or p_fake > 0.80:
        reasons.append("high_confidence")
    return reasons


def _build_reasons_video(
    avg_p_fake: float,
    champ_avg: float,
    chall_avg: float,
    p_fake_std: float,
    faces_found: int,
    total_frames: int,
    any_tiebreaker: bool,
) -> list[str]:
    reasons: list[str] = []
    if 0.35 <= avg_p_fake <= 0.65:
        reasons.append("borderline_score")
    disagreement = abs(champ_avg - chall_avg)
    if disagreement > 0.30:
        reasons.append("models_disagree")
    elif disagreement < 0.15:
        reasons.append("models_agree")
    if any_tiebreaker:
        reasons.append("tiebreaker_used")
    if p_fake_std > 0.25:
        reasons.append("high_variance")
    if avg_p_fake < 0.20 or avg_p_fake > 0.80:
        reasons.append("high_confidence")
    if faces_found < total_frames * 0.5:
        reasons.append("low_face_rate")
    if p_fake_std < 0.10:
        reasons.append("consistent_prediction")
    return reasons



_REASON_TO_WHY = {
    "no_face_detected":   "No face was detected clearly",
    "low_resolution":     "Image quality is too low for reliable detection",
    "small_face":         "The face in the image is too small",
    "models_disagree":    "Our detection models gave conflicting results",
    "borderline_score":   "The score is right on the boundary",
    "high_variance":      "Predictions varied a lot across video frames",
    "low_face_rate":      "A face could not be found in most frames",
    "tiebreaker_used":    "An extra model was called in to break a tie",
}

_NEXT_STEPS = {
    "FAKE": [
        "Don't share OTP or bank info",
        "Verify identity via an official channel",
        "Report or block if suspicious",
    ],
    "REAL": [
        "No strong manipulation detected",
        "Still verify the source if it's sensitive",
    ],
    "UNCERTAIN": [
        "Try again with better lighting or a closer face",
        "Upload a short video (5-10 s) for more data",
        "Avoid screenshots with overlays or heavy text",
    ],
}


def _build_advice(verdict: str, reasons: list[str]) -> dict:
    """Return a user-friendly advice dict from verdict + reason tags."""
    # Pick the first applicable human-readable "why" line
    why_parts: list[str] = []
    for r in reasons:
        if r in _REASON_TO_WHY:
            why_parts.append(_REASON_TO_WHY[r])
    if not why_parts:
        if verdict == "FAKE":
            why_parts.append("Strong signs of manipulation detected")
        elif verdict == "REAL":
            why_parts.append("No signs of manipulation detected")
        else:
            why_parts.append("The result is inconclusive")

    # Append agreement / confidence qualifier
    if "models_agree" in reasons and "high_confidence" in reasons:
        why_parts.append("all models agree with high confidence")
    elif "models_agree" in reasons:
        why_parts.append("all models agree on this result")
    elif "high_confidence" in reasons:
        why_parts.append("detection confidence is high")
    elif "consistent_prediction" in reasons:
        why_parts.append("predictions are consistent across frames")

    why = " — ".join(why_parts[:2])
    return {
        "why": why,
        "next_steps": _NEXT_STEPS.get(verdict, _NEXT_STEPS["UNCERTAIN"]),
    }


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
    t_start = time.monotonic()
    request_id = str(uuid.uuid4())

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    orig_w, orig_h = img.size
    low_res = orig_w < 128 or orig_h < 128

    # ---- Face crop + quality gate ----
    img, face_found, face_meta = crop_largest_face(img)
    small_face = (
        face_found
        and (face_meta.get("face_width", 0) < MIN_FACE_SIZE
             or face_meta.get("face_height", 0) < MIN_FACE_SIZE)
    )
    quality_ok = face_found and not small_face

    quality_warning: str | None = None
    if not face_found:
        quality_warning = "No face detected — running on full image. Accuracy is significantly lower."
    elif small_face:
        quality_warning = (
            f"Small face ({face_meta['face_width']}×{face_meta['face_height']}px). "
            "Results may be less reliable."
        )
    elif low_res:
        quality_warning = f"Low resolution ({orig_w}×{orig_h}). Results may be less reliable."

    # ---- Inference with per-model timing ----
    t0 = time.monotonic()
    champ_real, champ_fake = _run_champion(img)
    t_champ = int((time.monotonic() - t0) * 1000)

    t0 = time.monotonic()
    chall_real, chall_fake = _run_challenger(img)
    t_chall = int((time.monotonic() - t0) * 1000)

    t_fall = 0
    fall_fake = 0.0
    if fallback_model is not None:
        t0 = time.monotonic()
        _fr, fall_fake = _run_fallback(img)
        t_fall = int((time.monotonic() - t0) * 1000)

    final_fake, tiebreaker_used = _stacking_blend(champ_fake, chall_fake, fall_fake)

    # Suppress timing for fallback when it wasn't actually invoked in tiebreaker
    if not tiebreaker_used:
        t_fall = 0

    verdict, uncertain = _verdict(final_fake, quality_ok)
    confidence_band = _confidence_band(final_fake)
    disagreement = round(abs(champ_fake - chall_fake), 4)

    if not quality_ok:
        decision_path = "low_quality"
    elif tiebreaker_used:
        decision_path = "tiebreaker_used"
    else:
        decision_path = "primary_ensemble"

    reasons = _build_reasons_image(
        final_fake, face_found, low_res, small_face,
        champ_fake, chall_fake, tiebreaker_used,
    )

    models_list = [
        {"name": "FaceForge-Xception", "role": "champion",  "p_fake": round(champ_fake, 4), "used": True},
        {"name": "ViT-FF++TwoStage",   "role": "challenger", "p_fake": round(chall_fake, 4), "used": True},
    ]
    if fallback_model is not None:
        models_list.append(
            {"name": "EffNetB4-CelebDF", "role": "fallback", "p_fake": round(fall_fake, 4), "used": tiebreaker_used}
        )

    t_total = int((time.monotonic() - t_start) * 1000)

    return {
        "request_id": request_id,
        "media_type": "image",

        "verdict": verdict,
        "confidence_band": confidence_band,
        "final_p_fake": round(final_fake, 4),
        "uncertain": uncertain,

        "decision_path": decision_path,
        "reasons": reasons,
        "advice": _build_advice(verdict, reasons),

        "signals": {
            "face_found": face_found,
            "face_bbox": face_meta.get("bbox"),
            "quality_warning": quality_warning,
            "disagreement": disagreement,
        },

        "models": models_list,

        "privacy": {"stored_media": False},

        "timing_ms": {
            "total": t_total,
            "champion": t_champ,
            "challenger": t_chall,
            "fallback": t_fall,
        },
    }


# =========================================================================
# VIDEO DETECTION
# =========================================================================

import tempfile
import os

MAX_FRAMES = 10  # Extract up to this many evenly-spaced frames


def _extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list[tuple[Image.Image, float]]:
    """Extract evenly-spaced frames from a video file.

    Returns a list of (PIL.Image, timestamp_seconds) tuples.
    Skips the outer 10% of the video to avoid intro/outro artefacts.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    start = int(total_frames * 0.10)
    end = int(total_frames * 0.90)
    if end <= start:
        start, end = 0, total_frames

    n = min(max_frames, end - start)
    if n <= 0:
        cap.release()
        return []

    indices = [start + int(i * (end - start) / n) for i in range(n)]

    frames: list[tuple[Image.Image, float]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_sec = round(idx / fps, 2)
            frames.append((Image.fromarray(rgb), t_sec))

    cap.release()
    return frames


def _analyze_single_frame(img: Image.Image) -> dict:
    """Run the full ensemble pipeline on a single frame.

    Returns a rich dict including per-model scores for video-level aggregation.
    """
    img_cropped, face_found, face_meta = crop_largest_face(img)

    quality_ok = (
        face_found
        and face_meta.get("face_width", 0) >= MIN_FACE_SIZE
        and face_meta.get("face_height", 0) >= MIN_FACE_SIZE
    )

    champ_real, champ_fake = _run_champion(img_cropped)
    chall_real, chall_fake = _run_challenger(img_cropped)

    fall_fake = 0.0
    if fallback_model is not None:
        _fr, fall_fake = _run_fallback(img_cropped)

    final_fake, tiebreaker_used = _stacking_blend(champ_fake, chall_fake, fall_fake)
    verdict, uncertain = _verdict(final_fake, quality_ok)

    return {
        "prediction": verdict,
        "p_fake": round(final_fake, 4),
        "face_found": face_found,
        "tiebreaker_used": tiebreaker_used,
        "champ_fake": round(champ_fake, 4),
        "chall_fake": round(chall_fake, 4),
        "fall_fake": round(fall_fake, 4),
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    t_start = time.monotonic()
    request_id = str(uuid.uuid4())

    if file.content_type and file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a video file, not an image.")

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames_with_times = _extract_frames(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not frames_with_times:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")

    # ---- Analyze each frame ----
    frame_results = []
    for i, (frame, t_sec) in enumerate(frames_with_times):
        result = _analyze_single_frame(frame)
        result["frame_index"] = i
        result["t_sec"] = t_sec
        frame_results.append(result)

    total = len(frame_results)
    p_fakes = [r["p_fake"] for r in frame_results]

    avg_p_fake = float(np.mean(p_fakes))
    median_p_fake = float(np.median(p_fakes))
    variance = float(np.var(p_fakes))
    p_fake_std = float(np.std(p_fakes))

    uncertain_count = sum(1 for r in frame_results if r["prediction"] == "UNCERTAIN")
    confident_fake_frames = sum(1 for p in p_fakes if p > 0.65)
    uncertain_frame_rate = round(uncertain_count / total, 3)

    # Per-model averages
    champ_avg = round(float(np.mean([r["champ_fake"] for r in frame_results])), 4)
    chall_avg = round(float(np.mean([r["chall_fake"] for r in frame_results])), 4)
    fall_avg  = round(float(np.mean([r["fall_fake"]  for r in frame_results])), 4)

    # Top suspicious frames (highest p_fake, above 0.50)
    top_suspicious = sorted(
        [{"t_sec": r["t_sec"], "p_fake": r["p_fake"]} for r in frame_results if r["p_fake"] > 0.50],
        key=lambda x: x["p_fake"], reverse=True,
    )[:5]

    # Video-level verdict: if majority of frames are quality-uncertain, propagate
    quality_ok = uncertain_count <= total * 0.5
    verdict, is_uncertain = _verdict(avg_p_fake, quality_ok)
    confidence_band = _confidence_band(avg_p_fake)

    any_tiebreaker = any(r["tiebreaker_used"] for r in frame_results)
    faces_found = sum(1 for r in frame_results if r["face_found"])

    if not quality_ok:
        decision_path = "low_quality"
    elif any_tiebreaker:
        decision_path = "tiebreaker_used"
    else:
        decision_path = "primary_ensemble"

    reasons = _build_reasons_video(
        avg_p_fake, champ_avg, chall_avg, p_fake_std, faces_found, total, any_tiebreaker,
    )

    warnings: list[str] = []
    if p_fake_std > 0.25:
        warnings.append(
            f"High variance across frames (std={p_fake_std:.3f}). "
            "Inconsistent predictions may indicate partial manipulation."
        )
    if faces_found < total * 0.5:
        warnings.append(
            f"Face detected in only {faces_found}/{total} frames. Results may be less reliable."
        )

    t_total = int((time.monotonic() - t_start) * 1000)

    return {
        "request_id": request_id,
        "media_type": "video",

        "verdict": verdict,
        "confidence_band": confidence_band,
        "final_p_fake": round(avg_p_fake, 4),
        "uncertain": is_uncertain,

        "sampling": {
            "frames_used": total,
            "strategy": "middle_80_even",
        },

        "video_stats": {
            "mean_p_fake": round(avg_p_fake, 4),
            "median_p_fake": round(median_p_fake, 4),
            "variance": round(variance, 4),
            "uncertain_frame_rate": uncertain_frame_rate,
            "confident_fake_frames": confident_fake_frames,
        },

        "decision_path": decision_path,
        "reasons": reasons,
        "advice": _build_advice(verdict, reasons),

        "models_summary": {
            "champion_avg": champ_avg,
            "challenger_avg": chall_avg,
            "fallback_avg": fall_avg,
        },

        "top_suspicious_frames": top_suspicious,

        "privacy": {"stored_media": False},

        "timing_ms": {"total": t_total},

        "warnings": warnings,
    }


# =========================================================================
# AUDIO DETECTION (Multi-Model Ensemble)
# =========================================================================

from app.audio_model_loader import audio_models
from app.audio_inference import predict_ensemble

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}


_AUDIO_MODEL_NAMES = {"cnn-lstm": "CNN-LSTM", "tcn": "TCN", "tcn-lstm": "TCN-LSTM"}


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    """Analyze an audio clip for deepfake voice detection.

    Runs three models (CNN-LSTM, TCN, TCN-LSTM) as a majority-vote ensemble
    and returns a structured response consistent with /predict and /predict-video.
    """
    t_start = time.monotonic()
    request_id = str(uuid.uuid4())

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

    suffix = ext or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw = predict_ensemble(tmp_path, audio_models)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Build structured response ──────────────────────────────────────────
    avg_p_fake   = raw["probabilities"]["fake"]
    verdict, uncertain = _verdict(avg_p_fake, True)
    confidence_band    = _confidence_band(avg_p_fake)

    ens = raw.get("ensemble", {})
    models_voted_fake  = ens.get("models_voted_fake", 0)
    total_models       = ens.get("total_models", 0)
    models_voted_real  = total_models - models_voted_fake

    reasons: list[str] = []
    if models_voted_fake == total_models or models_voted_real == total_models:
        reasons.append("models_agree")
    else:
        reasons.append("models_disagree")
    if avg_p_fake < 0.20 or avg_p_fake > 0.80:
        reasons.append("high_confidence")
    if 0.35 <= avg_p_fake <= 0.65:
        reasons.append("borderline_score")

    models_list = []
    for key, detail in raw.get("model_details", {}).items():
        if "error" not in detail:
            mv = detail.get("prediction", "").upper()
            models_list.append({
                "name":    _AUDIO_MODEL_NAMES.get(key, key),
                "p_fake":  detail.get("fake_probability", 0.5),
                "verdict": mv if mv in ("REAL", "FAKE") else "UNCERTAIN",
                "chunks":  detail.get("chunks_analyzed", 0),
            })

    t_total = int((time.monotonic() - t_start) * 1000)

    return {
        "request_id":      request_id,
        "media_type":      "audio",

        "verdict":         verdict,
        "confidence_band": confidence_band,
        "final_p_fake":    round(avg_p_fake, 4),
        "uncertain":       uncertain,

        "decision_path":   "majority_vote",
        "reasons":         reasons,
        "advice":          _build_advice(verdict, reasons),

        "models": models_list,

        "ensemble_summary": {
            "voted_fake": models_voted_fake,
            "voted_real": models_voted_real,
            "total":      total_models,
        },

        "privacy":    {"stored_media": False},
        "timing_ms":  {"total": t_total},
    }