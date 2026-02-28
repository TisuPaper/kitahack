# app/fraud_detection/gemini_analyze.py
"""
Send filtered transcript to Gemini for fraud/scam analysis.
Returns strictly validated output with risk_score, confidence, evidence, scam_type.
"""

import json
from app.fraud_detection import config as cfg

_VALID_RISK_LEVELS = {"low", "medium", "high"}
_VALID_SCAM_TYPES = {
    "phishing", "tech_support", "investment", "romance",
    "impersonation", "parcel", "loan", "job", "other", "none",
}


def _extract_json(text: str) -> str:
    """Strip markdown fences if Gemini wraps the JSON."""
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        first_nl = text.find("\n", start)
        if first_nl != -1:
            start = first_nl + 1
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    return text


def _validate(data: dict) -> dict:
    """Enforce schema and sanitise values."""
    # risk_level
    risk_level = str(data.get("risk_level", "")).lower()
    if risk_level not in _VALID_RISK_LEVELS:
        risk_level = "medium"

    # risk_score (0–100)
    try:
        risk_score = int(data.get("risk_score", 0))
        risk_score = max(0, min(100, risk_score))
    except (TypeError, ValueError):
        risk_score = {"low": 15, "medium": 50, "high": 80}.get(risk_level, 50)

    # confidence (0.0–1.0)
    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = round(max(0.0, min(1.0, confidence)), 3)
    except (TypeError, ValueError):
        confidence = 0.5

    # scam_type
    scam_type = str(data.get("scam_type", "none")).lower()
    if scam_type not in _VALID_SCAM_TYPES:
        scam_type = "other"

    # summary
    summary = str(data.get("summary", "")).strip()[:500] or "No summary provided."

    # indicators: list[str]
    raw_indicators = data.get("indicators", [])
    indicators = [str(i).strip() for i in raw_indicators if str(i).strip()] if isinstance(raw_indicators, list) else []

    # evidence: list[{quote, reason}]
    raw_evidence = data.get("evidence", [])
    evidence: list[dict] = []
    if isinstance(raw_evidence, list):
        for e in raw_evidence:
            if isinstance(e, dict):
                q = str(e.get("quote", "")).strip()
                r = str(e.get("reason", "")).strip()
                if q and r:
                    evidence.append({"quote": q[:200], "reason": r[:200]})

    # recommendation
    recommendation = str(data.get("recommendation", "")).strip()[:400] or "Manual review recommended."

    return {
        "risk_level":      risk_level,
        "risk_score":      risk_score,
        "confidence":      confidence,
        "scam_type":       scam_type,
        "summary":         summary,
        "indicators":      indicators,
        "evidence":        evidence,
        "recommendation":  recommendation,
    }


def analyze_for_fraud(filtered_transcript: str, rule_context: str = "") -> dict:
    """
    Call Gemini to analyze transcript for fraud/scam indicators.
    Optionally include rule_context (pre-computed rule-based signals) for better accuracy.
    Returns strictly validated dict.
    """
    if not cfg.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    genai.configure(api_key=cfg.GEMINI_API_KEY)
    model = genai.GenerativeModel(cfg.GEMINI_MODEL)

    context_block = (
        f"\n\nRule-based signals already detected (use as additional context):\n{rule_context}"
        if rule_context else ""
    )

    prompt = f"""You are an expert fraud and scam detection assistant. Analyze the transcript below.
Personal/privacy details have already been redacted (shown as [PHONE], [EMAIL], [OTP], etc.).
{context_block}

Respond with ONLY a JSON object — no explanation, no markdown — using this exact schema:
{{
  "risk_level": "low" | "medium" | "high",
  "risk_score": <integer 0-100>,
  "confidence": <float 0.0-1.0>,
  "scam_type": "phishing" | "tech_support" | "investment" | "romance" | "impersonation" | "parcel" | "loan" | "job" | "other" | "none",
  "summary": "<one sentence summary>",
  "indicators": ["<indicator 1>", "<indicator 2>"],
  "evidence": [
    {{"quote": "<exact short quote from transcript>", "reason": "<why this is suspicious>"}}
  ],
  "recommendation": "<one actionable sentence for the user>"
}}

Transcript:
{filtered_transcript.strip() or "(empty — possibly silence or no speech detected)"}
"""

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "quota" in err_msg or "rate" in err_msg or "resource_exhausted" in err_msg:
            return _validate({
                "risk_level":   "medium",
                "risk_score":   50,
                "confidence":   0.5,
                "scam_type":    "other",
                "summary":      "Fraud analysis temporarily unavailable (API quota exceeded). Try again later or check your Gemini plan.",
                "indicators":   [],
                "evidence":     [],
                "recommendation": "Manual review recommended. Retry after a short delay or check https://ai.google.dev/gemini-api/docs/rate-limits",
            })
        raise

    raw = _extract_json(response.text or "")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _validate({
            "risk_level": "unknown",
            "summary": raw[:300] if raw else "No response from model.",
        })

    return _validate(data)
