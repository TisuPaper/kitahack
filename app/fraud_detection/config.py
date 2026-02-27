# app/fraud_detection/config.py
"""
Configuration for fraud detection (env vars, paths).
"""

import os

# Gemini API key (required for analysis). Set in env: GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Optional: use Gemini for transcription instead of Whisper (default: Whisper)
USE_GEMINI_FOR_STT = os.environ.get("FRAUD_USE_GEMINI_STT", "").lower() in ("1", "true", "yes")

# Model names
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # base, small, medium, large
