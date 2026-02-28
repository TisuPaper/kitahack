# app/fraud_detection/stt.py
"""
Speech-to-text: Whisper (default, runs on VM) or optional Gemini.
Whisper model is cached after first load so subsequent calls are fast.
"""

import os
from app.fraud_detection import config as cfg

_whisper_model = None


def _get_whisper_model():
    """Load Whisper once and cache for the lifetime of the process."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "Whisper not installed. Run: pip install openai-whisper"
            )
        _whisper_model = whisper.load_model(cfg.WHISPER_MODEL)
    return _whisper_model


def transcribe_whisper(audio_path: str) -> str:
    """Transcribe audio file using cached Whisper model."""
    model = _get_whisper_model()
    result = model.transcribe(audio_path, fp16=False, language=None)
    return (result.get("text") or "").strip()


def transcribe_gemini(audio_path: str) -> str:
    """Transcribe using Gemini API (requires GEMINI_API_KEY)."""
    if not cfg.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "Google Generative AI not installed. Run: pip install google-generativeai"
        )
    genai.configure(api_key=cfg.GEMINI_API_KEY)
    ext = os.path.splitext(audio_path)[1].lower()
    mime = {
        ".wav": "audio/wav", ".mp3": "audio/mp3",
        ".ogg": "audio/ogg", ".webm": "audio/webm", ".m4a": "audio/mp4",
    }.get(ext, "audio/wav")
    uploaded = genai.upload_file(path=audio_path, mime_type=mime)
    model = genai.GenerativeModel(cfg.GEMINI_MODEL)
    response = model.generate_content(
        ["Transcribe this audio exactly. Output only the spoken text, no commentary.", uploaded]
    )
    return (response.text or "").strip()


def transcribe(audio_path: str) -> str:
    """Run STT: Gemini if configured, else Whisper."""
    if cfg.USE_GEMINI_FOR_STT and cfg.GEMINI_API_KEY:
        return transcribe_gemini(audio_path)
    return transcribe_whisper(audio_path)
