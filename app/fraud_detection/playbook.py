# app/fraud_detection/playbook.py
"""
Scam Playbook Matching — RAG-lite without embeddings.
Compares transcript against known scam scripts using normalised token overlap.
No external API or ML model needed; fast enough for real-time use.
"""

import re
from dataclasses import dataclass


@dataclass
class PlaybookMatch:
    scam_type:   str
    label:       str
    similarity:  float          # 0.0 – 1.0
    matched_phrases: list[str]


# Each entry: (scam_type, label, key phrases)
# Phrases are intentionally partial so they match paraphrases.
_PLAYBOOKS: list[tuple[str, str, list[str]]] = [
    (
        "impersonation",
        "Police / Bank Impersonation",
        [
            "calling from the police", "PDRM", "bank negara", "your account is frozen",
            "akaun anda telah dibekukan", "we have a warrant", "cooperate with investigation",
            "do not tell anyone", "this is confidential", "calling from MCMC",
            "jangan beritahu sesiapa", "saya pegawai", "I am an officer",
        ],
    ),
    (
        "tech_support",
        "Tech Support / Remote Access",
        [
            "your computer has a virus", "call Microsoft", "call Apple support",
            "download AnyDesk", "download TeamViewer", "give me remote access",
            "I can fix your computer", "your device is compromised",
            "install this software", "screen sharing",
        ],
    ),
    (
        "investment",
        "Investment / Crypto Scam",
        [
            "guaranteed profit", "guaranteed return", "double your money",
            "high return investment", "passive income", "invest in crypto",
            "forex trading profit", "bitcoin opportunity", "risk-free investment",
            "pulangan dijamin", "gandakan wang anda",
        ],
    ),
    (
        "phishing",
        "OTP / Credential Phishing",
        [
            "give me the OTP", "tell me the verification code", "what is your OTP",
            "share the code", "I need the one-time password", "berikan OTP",
            "kod pengesahan", "verify your account", "confirm your identity",
            "click this link", "enter your password",
        ],
    ),
    (
        "parcel",
        "Parcel / Customs Scam",
        [
            "your parcel is detained", "customs has held your package",
            "pay the customs fee", "release your parcel", "package is blocked",
            "Pos Malaysia", "pakej anda ditahan", "kastam menahan",
            "illegal items found", "bahan terlarang",
        ],
    ),
    (
        "romance",
        "Romance / Pig-Butchering Scam",
        [
            "I love you", "we met online", "I want to send you a gift",
            "gift is stuck at customs", "I am a soldier", "I am working abroad",
            "help me with money", "send me money and I will repay",
            "I have feelings for you",
        ],
    ),
    (
        "loan",
        "Loan Scam",
        [
            "easy loan", "instant approval", "no documents needed",
            "pay processing fee first", "pay insurance first",
            "pinjaman mudah", "bayar dulu", "tiada dokumen diperlukan",
            "guaranteed loan approval",
        ],
    ),
    (
        "job",
        "Job / Task Scam",
        [
            "work from home earn money", "like and subscribe for money",
            "easy task high pay", "part-time online job", "commission per task",
            "deposit to unlock earnings", "buy gift cards", "purchase Google Play cards",
            "kerja dari rumah", "tugas mudah wang banyak",
        ],
    ),
]


def _tokenise(text: str) -> set[str]:
    """Lowercase words only, remove stop words."""
    _STOP = {"a", "an", "the", "is", "are", "was", "were", "i", "you", "we",
             "he", "she", "it", "my", "your", "their", "this", "that", "and",
             "or", "but", "in", "on", "at", "to", "of", "for", "with", "me",
             "saya", "anda", "yang", "dan", "di", "ke", "ini", "itu"}
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return tokens - _STOP


def match_playbooks(transcript: str, top_n: int = 3) -> list[PlaybookMatch]:
    """
    Compare transcript against all playbooks.
    Returns top_n matches sorted by similarity descending.
    """
    t_tokens = _tokenise(transcript)
    if not t_tokens:
        return []

    results: list[PlaybookMatch] = []

    for scam_type, label, phrases in _PLAYBOOKS:
        phrase_tokens = [_tokenise(p) for p in phrases]
        all_playbook_tokens = set().union(*phrase_tokens)
        if not all_playbook_tokens:
            continue

        # Jaccard similarity between transcript and full playbook vocab
        overlap = t_tokens & all_playbook_tokens
        union   = t_tokens | all_playbook_tokens
        similarity = len(overlap) / len(union) if union else 0.0

        # Find specific phrase-level matches for evidence
        matched_phrases: list[str] = []
        for phrase, p_tokens in zip(phrases, phrase_tokens):
            if p_tokens and p_tokens.issubset(t_tokens):
                matched_phrases.append(phrase)

        # Boost similarity if we found exact phrase matches
        if matched_phrases:
            similarity = max(similarity, 0.25 * len(matched_phrases))

        similarity = min(1.0, round(similarity, 3))
        results.append(PlaybookMatch(
            scam_type=scam_type,
            label=label,
            similarity=similarity,
            matched_phrases=matched_phrases[:5],
        ))

    results.sort(key=lambda x: -x.similarity)
    return [r for r in results[:top_n] if r.similarity > 0.02]
