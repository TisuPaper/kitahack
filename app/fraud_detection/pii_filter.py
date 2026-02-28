# app/fraud_detection/pii_filter.py
"""
Filter personal and privacy-sensitive words from transcripts before sending to Gemini.
Uses typed placeholders ([PHONE], [EMAIL], etc.) so Gemini retains context.
Overlapping spans are merged so no double-replacement occurs.
"""

import os
import re
from pathlib import Path

# Each entry: (regex_pattern, placeholder_label)
_TYPED_PII_PATTERNS: list[tuple[str, str]] = [
    # Card numbers (16–19 digits, with or without separators)
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "CARD"),
    # Malaysia IC/NRIC: YYMMDD-PB-XXXX
    (r"\b\d{6}[- ]\d{2}[- ]\d{4}\b", "NRIC"),
    # SSN-like
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    # Email
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "EMAIL"),
    # Malaysia phone: 01x-xxxxxxx, +601x, 03-xxxxxxxx
    (r"\b(?:\+?60|0)\d{1,2}[- ]?\d{3,4}[- ]?\d{4}\b", "PHONE"),
    # Generic international phone
    (r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b", "PHONE"),
    # OTP / password / PIN keywords + following digits
    (r"\b(?:OTP|one[ -]time[ -]password|passcode|pin)\b(?:\s+(?:is|was|:))?\s*\d{4,8}", "OTP"),
    (r"\b(?:password|kata[ -]laluan)\b", "PASSWORD"),
    # Bank account / routing
    (r"\b(?:bank account|account number|no[.] akaun|nombor akaun|routing number)\b", "ACCOUNT"),
    # Name self-disclosure: "my name is X", "I am X", "call me X"
    (r"\b(?:my name is|I am|call me|nama saya|saya ialah)\s+[A-Za-z]+(?:\s+[A-Za-z]+)?", "NAME"),
    # Malaysia IC explicit mention
    (r"\b(?:IC number|no[.] IC|nombor IC|mykad)\b", "NRIC"),
]


def _load_custom_patterns(path: str) -> list[tuple[str, str]]:
    """Load extra patterns from a file (format: LABEL|regex, one per line)."""
    extra: list[tuple[str, str]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    label, pattern = line.split("|", 1)
                    extra.append((pattern.strip(), label.strip().upper()))
                else:
                    extra.append((line, "REDACTED"))
    except OSError:
        pass
    return extra


def _get_patterns() -> list[tuple[str, str]]:
    patterns = _TYPED_PII_PATTERNS.copy()
    env_file = os.environ.get("FRAUD_PII_PATTERNS_FILE")
    if env_file:
        patterns.extend(_load_custom_patterns(env_file))
    return patterns


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Sort by start, merge overlapping spans (keep earlier label)."""
    if not spans:
        return []
    spans.sort(key=lambda x: x[0])
    merged: list[tuple[int, int, str]] = [spans[0]]
    for start, end, label in spans[1:]:
        prev_start, prev_end, prev_label = merged[-1]
        if start < prev_end:          # overlapping
            merged[-1] = (prev_start, max(prev_end, end), prev_label)
        else:
            merged.append((start, end, label))
    return merged


def filter_pii(text: str, patterns: list[tuple[str, str]] | None = None) -> tuple[str, list[dict]]:
    """
    Redact PII using typed placeholders. Returns (filtered_text, redacted_list).
    redacted_list: [{"original": "...", "label": "PHONE"}, ...]
    Overlapping spans are merged (earliest label wins).
    """
    if patterns is None:
        patterns = _get_patterns()

    spans: list[tuple[int, int, str]] = []
    for pattern, label in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), label))

    merged = _merge_spans(spans)

    redacted: list[dict] = []
    out = text
    # Replace from end so positions stay valid
    for start, end, label in reversed(merged):
        original = out[start:end]
        redacted.append({"original": original, "label": label})
        out = out[:start] + f"[{label}]" + out[end:]

    # Re-order redacted to match original text order
    redacted.reverse()
    return out.strip(), redacted
