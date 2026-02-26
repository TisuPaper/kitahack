# app/audio_model_loader.py
"""
Loads all 3 audio deepfake detection models for ensemble prediction.
Models: CNN-LSTM, TCN, TCN-LSTM
"""

import os
import torch
from app.audio_detection.models import CNN_LSTM, TCN, TCN_LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
INPUT_DIM = 1  # Single-channel raw audio

def _load_audio_model(model_class, model_name):
    """Load a single audio model from .pth file."""
    path = os.path.join(MODELS_DIR, f"{model_name}_audio_classifier.pth")
    if not os.path.exists(path):
        print(f"  ⚠️  Audio model not found: {path}")
        return None

    model = model_class(input_dim=INPUT_DIM, num_classes=2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    print(f"  ✅ Loaded: {model_name}")
    return model


def load_all_audio_models():
    """Load all 3 audio models and return a dict."""
    print("🎙️ Loading audio detection models...")
    models = {}

    configs = [
        ("cnn-lstm", CNN_LSTM),
        ("tcn", TCN),
        ("tcn-lstm", TCN_LSTM),
    ]

    for name, cls in configs:
        m = _load_audio_model(cls, name)
        if m is not None:
            models[name] = m

    print(f"🎙️ Loaded {len(models)}/3 audio models")
    return models


# Load at import time (same pattern as model_loader.py)
audio_models = load_all_audio_models()
