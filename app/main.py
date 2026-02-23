# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model_loader import model, preprocess, device

from PIL import Image
import io
import torch

import cv2
import numpy as np


app = FastAPI(title="Deepfake Detector API")


# ---- Face detector (Haar cascade) ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_largest_face(pil_img: Image.Image) -> tuple[Image.Image, bool]:
    """
    Returns (cropped_image, face_found).
    If no face found, returns original image and False.
    """
    rgb = np.array(pil_img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return pil_img, False

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return pil_img, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = int(0.25 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(rgb.shape[1], x + w + pad)
    y2 = min(rgb.shape[0], y + h + pad)

    face_rgb = rgb[y1:y2, x1:x2]
    return Image.fromarray(face_rgb), True


CONFIDENCE_THRESHOLD = 0.60


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
    if orig_w < 128 or orig_h < 128:
        warnings.append(
            f"Low resolution ({orig_w}x{orig_h}). Results may be less reliable."
        )

    # Crop face
    img, face_found = crop_largest_face(img)
    if not face_found:
        warnings.append(
            "No face detected — running on full image. "
            "Accuracy is lower without a proper face crop."
        )

    # Preprocess + inference
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    real_prob = float(probs[0][0].item())
    fake_prob = float(probs[0][1].item())
    confidence = float(conf.item())

    # 0=Real, 1=Fake
    if confidence < CONFIDENCE_THRESHOLD:
        label = "Uncertain"
        warnings.append(
            f"Confidence ({confidence:.1%}) is below threshold "
            f"({CONFIDENCE_THRESHOLD:.0%}). Prediction is unreliable."
        )
    else:
        label = "Real" if pred.item() == 0 else "Fake"

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "real": round(real_prob, 4),
            "fake": round(fake_prob, 4),
        },
        "face_found": face_found,
        "warnings": warnings,
    }