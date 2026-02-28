# app/fraud_detection/__init__.py
"""
AI fraud detection pipeline: audio → speech-to-text → PII filter → Gemini analysis.
"""

from app.fraud_detection.pipeline import run_fraud_pipeline

__all__ = ["run_fraud_pipeline"]
