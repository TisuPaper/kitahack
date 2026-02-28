# app/fraud_detection/pipeline.py
"""
Hybrid fraud-detection pipeline:
  audio → STT → PII filter → Rule engine + Playbook matching + Gemini → final score
"""

from app.fraud_detection.stt import transcribe
from app.fraud_detection.pii_filter import filter_pii
from app.fraud_detection.rules import score_rules
from app.fraud_detection.playbook import match_playbooks
from app.fraud_detection.gemini_analyze import analyze_for_fraud

# Weights for final hybrid score (must sum to 1.0)
_W_RULES    = 0.35
_W_PLAYBOOK = 0.20
_W_GEMINI   = 0.45


def _level_from_score(score: int) -> str:
    if score >= 65:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def run_fraud_pipeline(audio_path: str) -> dict:
    """
    Full pipeline. Returns:
      transcript_raw, transcript_filtered, redacted (typed list),
      rule_signals, playbook_matches, gemini_analysis,
      final_risk_score (0-100), final_risk_level, final_scam_type,
      evidence (merged), error.
    """
    result: dict = {
        "transcript_raw":      "",
        "transcript_filtered": "",
        "redacted":            [],
        "rule_signals":        {},
        "playbook_matches":    [],
        "gemini_analysis":     None,
        "final_risk_score":    0,
        "final_risk_level":    "low",
        "final_scam_type":     "none",
        "evidence":            [],
        "error":               None,
    }

    # ── 1. Speech-to-Text ────────────────────────────────────────────────
    try:
        result["transcript_raw"] = transcribe(audio_path)
    except Exception as e:
        result["error"] = f"Speech-to-text failed: {e}"
        return result

    # ── 2. PII Filter (typed placeholders) ───────────────────────────────
    try:
        filtered, redacted = filter_pii(result["transcript_raw"])
        result["transcript_filtered"] = filtered
        result["redacted"] = redacted
    except Exception as e:
        result["error"] = f"PII filter failed: {e}"
        return result

    # ── 3. Rule-based scoring (runs on raw transcript) ───────────────────
    try:
        rule_out = score_rules(result["transcript_raw"])
        result["rule_signals"] = rule_out
    except Exception as e:
        rule_out = {"rule_score": 0, "matched_rules": [], "evidence": [], "summary": ""}
        result["rule_signals"] = rule_out

    # ── 4. Playbook matching (runs on filtered transcript) ────────────────
    try:
        pb_matches = match_playbooks(result["transcript_filtered"])
        result["playbook_matches"] = [
            {
                "scam_type":       m.scam_type,
                "label":           m.label,
                "similarity":      m.similarity,
                "matched_phrases": m.matched_phrases,
            }
            for m in pb_matches
        ]
    except Exception as e:
        pb_matches = []
        result["playbook_matches"] = []

    playbook_score = min(100, int((pb_matches[0].similarity if pb_matches else 0) * 100))

    # ── 5. Gemini analysis (filtered text + rule context) ─────────────────
    try:
        rule_ctx = result["rule_signals"].get("summary", "")
        gemini_out = analyze_for_fraud(result["transcript_filtered"], rule_context=rule_ctx)
        result["gemini_analysis"] = gemini_out
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "rate" in err:
            summary_msg = "Fraud analysis temporarily unavailable (API quota exceeded). Try again later."
        else:
            summary_msg = f"Gemini unavailable: {e}"
        gemini_out = {
            "risk_level": "medium", "risk_score": 50, "confidence": 0.5,
            "scam_type": "other", "summary": summary_msg,
            "indicators": [], "evidence": [], "recommendation": "Manual review required.",
        }
        result["gemini_analysis"] = gemini_out

    # ── 6. Combine signals into final hybrid score ────────────────────────
    rule_score    = rule_out.get("rule_score", 0)
    gemini_score  = gemini_out.get("risk_score", 50)

    hybrid_score = int(
        _W_RULES    * rule_score +
        _W_PLAYBOOK * playbook_score +
        _W_GEMINI   * gemini_score
    )
    hybrid_score = max(0, min(100, hybrid_score))

    result["final_risk_score"] = hybrid_score
    result["final_risk_level"] = _level_from_score(hybrid_score)

    # Scam type: prefer Gemini unless rules/playbook strongly disagree
    gemini_type = gemini_out.get("scam_type", "none")
    playbook_type = pb_matches[0].scam_type if pb_matches else "none"
    result["final_scam_type"] = gemini_type if gemini_type != "none" else playbook_type

    # ── 7. Merge evidence from all three sources ──────────────────────────
    evidence: list[dict] = []
    evidence.extend(rule_out.get("evidence", []))
    for m in (result["playbook_matches"] or []):
        for phrase in m.get("matched_phrases", []):
            evidence.append({"quote": phrase, "reason": f"Matches {m['label']} playbook"})
    evidence.extend(gemini_out.get("evidence", []))

    # Deduplicate by quote
    seen: set[str] = set()
    deduped: list[dict] = []
    for e in evidence:
        key = e.get("quote", "")[:60].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    result["evidence"] = deduped[:10]

    return result
