# app/audio_inference.py
"""
Multi-model audio deepfake inference.
Runs all 3 models (CNN-LSTM, TCN, TCN-LSTM) as an ensemble
and returns aggregated predictions.
"""

import os
import torch
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_DURATION = 2  # seconds
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32000 samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resample(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Linear-interpolation resample to SAMPLE_RATE."""
    if src_sr == SAMPLE_RATE:
        return audio
    new_len = int(round(len(audio) * SAMPLE_RATE / src_sr))
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio)


def _load_with_av(file_path: str) -> np.ndarray:
    """Decode any format soundfile can't handle (WebM, Opus, MP3 …) using PyAV."""
    import av as _av
    samples: list[np.ndarray] = []
    sr: int = SAMPLE_RATE
    with _av.open(file_path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        sr = stream.sample_rate or SAMPLE_RATE
        resampler = _av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=SAMPLE_RATE
        )
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()     # shape (1, N) float32
                samples.append(arr[0])
        # Flush resampler
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            samples.append(arr[0])
    if not samples:
        return np.zeros(CLIP_LENGTH, dtype=np.float32)
    return np.concatenate(samples).astype(np.float32)


def _load_audio(file_path: str) -> np.ndarray:
    """Load audio file and return 16 kHz mono float32 array.

    Tries soundfile first (fast, handles WAV/FLAC/OGG).
    Falls back to PyAV for formats soundfile cannot read (WebM/Opus, MP3, MP4 …).
    """
    try:
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return _resample(audio.astype(np.float32), sr)
    except Exception:
        pass  # soundfile can't decode this format — try PyAV

    return _load_with_av(file_path)


def predict_single_model(audio: np.ndarray, model, chunk_size: int = CLIP_LENGTH, max_chunks: int = 10):
    """
    Predict with a single model using chunk-based majority vote.
    Takes pre-loaded audio array (not file path) for efficiency.
    Limits to max_chunks evenly-spaced chunks for speed.
    """
    total_chunks = len(audio) // chunk_size
    if total_chunks <= 0:
        # Audio shorter than one chunk — pad and analyze
        chunk = np.pad(audio, (0, max(0, chunk_size - len(audio))))
        total_chunks = 1
        chunk_indices = [0]
    elif total_chunks <= max_chunks:
        chunk_indices = list(range(total_chunks))
    else:
        # Sample evenly-spaced chunks
        chunk_indices = [int(i * total_chunks / max_chunks) for i in range(max_chunks)]

    chunk_results = []
    for ci in chunk_indices:
        start = ci * chunk_size
        chunk = audio[start:start + chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(waveform)
            # Model output: 0 = FAKE, 1 = REAL (from LabelEncoder)
            # We remap it here so 1 = FAKE and 0 = REAL everywhere else
            model_pred = outputs.argmax(dim=1).item()
            is_fake = 1 if model_pred == 0 else 0

            probs = torch.softmax(outputs, dim=1)
            p_fake = float(probs[0][0])
            p_real = float(probs[0][1])

        chunk_results.append({
            "prediction": is_fake,
            "p_real": round(p_real, 4),
            "p_fake": round(p_fake, 4),
        })

    if not chunk_results:
        return 0, 0.5, []

    # Now standard boolean logic applies: 1 is Fake
    fake_count = sum(1 for c in chunk_results if c["prediction"] == 1)
    fake_ratio = fake_count / len(chunk_results)
    avg_fake_prob = np.mean([c["p_fake"] for c in chunk_results])

    final_pred = 1 if fake_ratio > 0.5 else 0
    confidence = avg_fake_prob if final_pred == 1 else (1 - avg_fake_prob)

    return final_pred, float(confidence), chunk_results


def predict_ensemble(file_path: str, audio_models: dict):
    """
    Run ALL audio models and aggregate results.
    Loads audio ONCE and shares across all models for speed.
    """
    # Load audio once (not per model!)
    audio = _load_audio(file_path)

    model_results = {}
    ensemble_votes = []

    for model_name, model in audio_models.items():
        try:
            pred, conf, chunks = predict_single_model(audio, model)
            fake_chunks = sum(1 for c in chunks if c["prediction"] == 1)

            model_results[model_name] = {
                "prediction": "Fake" if pred == 1 else "Real",
                "confidence": round(conf, 4),
                "fake_probability": round(
                    float(np.mean([c["p_fake"] for c in chunks])) if chunks else 0.5, 4
                ),
                "chunks_analyzed": len(chunks),
                "chunks_fake": fake_chunks,
                "chunks_real": len(chunks) - fake_chunks,
            }
            ensemble_votes.append(pred)
        except Exception as e:
            model_results[model_name] = {
                "prediction": "Error",
                "error": str(e),
            }

    # Ensemble: majority vote across models
    if ensemble_votes:
        fake_model_count = sum(ensemble_votes)
        total_models = len(ensemble_votes)
        ensemble_pred = 1 if fake_model_count > total_models / 2 else 0

        valid_results = [r for r in model_results.values() if "confidence" in r]
        avg_confidence = float(np.mean([r["confidence"] for r in valid_results])) if valid_results else 0.5
        avg_fake_prob = float(np.mean([r["fake_probability"] for r in valid_results])) if valid_results else 0.5
    else:
        ensemble_pred = 0
        avg_confidence = 0.5
        avg_fake_prob = 0.5
        fake_model_count = 0
        total_models = 0

    return {
        "prediction": "Fake" if ensemble_pred == 1 else "Real",
        "confidence": round(avg_confidence, 4),
        "probabilities": {
            "real": round(1 - avg_fake_prob, 4),
            "fake": round(avg_fake_prob, 4),
        },
        "ensemble": {
            "models_voted_fake": fake_model_count,
            "models_voted_real": total_models - fake_model_count,
            "total_models": total_models,
            "decision": "majority_vote",
        },
        "model_details": model_results,
    }
