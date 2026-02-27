# app/fraud_detection/rules.py
"""
Rule-based heuristic scoring engine.
Fast, deterministic, offline — no API needed.
Each rule adds to a cumulative risk_score (0–100 after clamping).
Also returns matched rule names and evidence quotes.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RuleMatch:
    rule:   str
    weight: int
    quotes: list[str] = field(default_factory=list)


# Each entry: (rule_id, weight, [patterns], description)
# Weight: positive = raises risk, max total ~100
_RULES: list[tuple[str, int, list[str], str]] = [
    # OTP / PIN harvesting
    (
        "otp_request", 35,
        [r"\b(OTP|one[- ]time[- ]password|passcode|verification code|kod pengesahan)\b",
         r"\b(give me|tell me|share|send|berikan|bagi saya|send me).{0,20}(OTP|code|pin|passcode)\b"],
        "Requests OTP or verification code",
    ),
    # Urgent money transfer
    (
        "urgent_transfer", 30,
        [r"\b(transfer|hantar|pindah|wire).{0,20}(now|immediately|segera|sekarang|right away)\b",
         r"\b(urgent|segera|cepat|immediately).{0,20}(transfer|pay|bayar)\b",
         r"\b(transfer|pay|bayar).{0,20}(account|akaun|number|nombor)\b"],
        "Urgently requests money transfer",
    ),
    # Bank/police/government impersonation
    (
        "impersonation", 25,
        [r"\b(I am|I'm|calling from|dari).{0,30}(PDRM|police|polis|bank negara|BNM|MCMC|LHDN|IRB|Maybank|CIMB|RHB|HSBC|FBI|interpol)\b",
         r"\b(officer|pegawai|detective|agent|penolong).{0,20}(calling|menghubungi|dari)\b",
         r"\b(your account|akaun anda).{0,30}(suspended|frozen|blocked|dibekukan|digantung)\b"],
        "Impersonates authority (bank/police/government)",
    ),
    # Remote access requests
    (
        "remote_access", 25,
        [r"\b(AnyDesk|TeamViewer|AmmyAdmin|UltraViewer|remote.{0,10}access|screen.{0,10}share)\b",
         r"\b(download|install|pasang|muat turun).{0,30}(app|application|software|TeamViewer|AnyDesk)\b"],
        "Requests remote access to device",
    ),
    # Personal data harvesting
    (
        "data_harvest", 20,
        [r"\b(IC number|MyKad|passport|nombor IC|identity card)\b",
         r"\b(full name|nama penuh|date of birth|tarikh lahir|mother.{0,10}maiden)\b",
         r"\b(security question|soalan keselamatan|secret word)\b"],
        "Requests personal identity data",
    ),
    # Lottery / prize scams
    (
        "lottery_scam", 20,
        [r"\b(you.{0,10}(won|win|selected|chosen)|congratulations.{0,20}prize)\b",
         r"\b(lucky (draw|winner)|undian bertuah|hadiah)\b",
         r"\b(claim.{0,20}prize|collect.{0,10}winnings|tuntut hadiah)\b"],
        "Lottery or prize scam pattern",
    ),
    # Investment / get-rich promises
    (
        "investment_scam", 20,
        [r"\b(guaranteed (return|profit|income)|pulangan dijamin)\b",
         r"\b(double.{0,10}money|gandakan wang|high return|passive income)\b",
         r"\b(crypto|bitcoin|forex|trading).{0,30}(profit|return|invest)\b"],
        "Fraudulent investment promise",
    ),
    # Parcel / delivery scams (common in Malaysia)
    (
        "parcel_scam", 15,
        [r"\b(parcel|pakej|package).{0,20}(detained|held|seized|ditahan)\b",
         r"\b(customs|kastam|immigration).{0,20}(package|parcel|item)\b",
         r"\b(Pos Malaysia|J&T|DHL|FedEx).{0,20}(problem|issue|detained)\b"],
        "Parcel detention scam",
    ),
    # Pressure / fear tactics
    (
        "pressure_tactics", 15,
        [r"\b(arrest|tangkap|arrested|warrant).{0,30}(you|your name|nama anda)\b",
         r"\b(last (chance|warning|notice)|peluang terakhir|amaran terakhir)\b",
         r"\b(do not tell|jangan beritahu|keep this (secret|confidential))\b"],
        "Uses pressure, fear, or secrecy tactics",
    ),
    # Loan scams
    (
        "loan_scam", 15,
        [r"\b(easy loan|pinjaman mudah|no (document|collateral)|instant (loan|approval))\b",
         r"\b(pay (processing|admin|insurance) fee.{0,20}(first|dahulu))\b"],
        "Upfront fee or easy-loan scam",
    ),
]


def score_rules(transcript: str) -> dict:
    """
    Score transcript against all rules.
    Returns:
      rule_score:   int 0–100
      matched:      list of RuleMatch
      evidence:     list of {quote, reason}
      summary:      human-readable string for Gemini context
    """
    matched: list[RuleMatch] = []
    total_weight = 0

    for rule_id, weight, patterns, description in _RULES:
        quotes: list[str] = []
        for pat in patterns:
            for m in re.finditer(pat, transcript, re.IGNORECASE):
                # Grab a short snippet with some context
                start = max(0, m.start() - 20)
                end   = min(len(transcript), m.end() + 20)
                snippet = transcript[start:end].strip()
                if snippet and snippet not in quotes:
                    quotes.append(snippet)
        if quotes:
            matched.append(RuleMatch(rule=rule_id, weight=weight, quotes=quotes[:3]))
            total_weight += weight

    rule_score = min(100, total_weight)

    evidence = []
    for rm in matched:
        for q in rm.quotes:
            evidence.append({"quote": q, "reason": _rule_description(rm.rule)})

    matched_ids = [rm.rule for rm in matched]
    summary_lines = [f"- {_rule_description(r)} (weight +{w})" for r, w, _, _ in _RULES if r in matched_ids]
    summary = "\n".join(summary_lines) if summary_lines else "No rule-based signals detected."

    return {
        "rule_score": rule_score,
        "matched_rules": matched_ids,
        "evidence": evidence,
        "summary": summary,
    }


def _rule_description(rule_id: str) -> str:
    for r, _, _, desc in _RULES:
        if r == rule_id:
            return desc
    return rule_id
