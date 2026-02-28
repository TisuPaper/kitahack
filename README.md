# Realytic — Deepfake & Fraud Detector

A privacy-preserving, multi-modal deepfake detection system run at Trusted Execution Environment (TEE), covering image, video, and audio deepfakes, plus a fraud analysis pipeline that transcribes speech, redacts private data, and uses a hybrid risk engine to detect scams accross multiple platforms (web/native app).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution & Features](#2-solution--features)
3. [Architecture](#3-architecture)
4. [Tech Stack](#4-tech-stack)
5. [System Overview](#5-system-overview)
6. [Visual Deepfake Detection (Image & Video)](#6-visual-deepfake-detection-image--video)
7. [Audio Deepfake Detection](#7-audio-deepfake-detection)
8. [Call Fraud Detection (Advanced)](#8-call-fraud-detection-advanced)
9. [API Reference](#9-api-reference)
10. [Challenges Faced](#10-challenges-faced)
11. [Future Roadmap](#11-future-roadmap)

---

## 1. Problem Statement

- **Deepfake-enabled fraud is now a real financial threat, not a “future risk.** — Fraud losses are already massive (US$12.5B reported in 2024), and deepfakes are rapidly scaling—occurring as frequently as one attempt every five minutes—making impersonation scams harder and faster than ever.
- **People can’t reliably tell what’s real, and the tools aren’t simple across devices.** — People can’t reliably tell what’s real, and the tools aren’t simple across devices. Even among consumers familiar with generative AI, 59% say they have a hard time distinguishing human vs AI media, yet verification isn’t “one-tap” across both phone and PC during live situations (calls, streams, messages).
- **Privacy and trust** — Call recordings and transcripts can contain OTPs, NRIC, bank details, and phone numbers—so people hesitate to use detection tools if they must upload sensitive content to external servers.

---

## 2. Solution & Features

Realytic is a **multi-modal deepfake and call-fraud detection system** that runs in a Trusted Execution Environment (TEE) without storing user's data. It gives users confidence scores and actionable advice instead of a simple real/fake label, and keeps sensitive data out of third-party APIs.

**Highlights:**

- **Image & video deepfake** — Ensemble of Xception, ViT, and EfficientNet-B4 with a tiebreaker for uncertain cases; confidence bands (HIGH/MEDIUM/LOW) and per-model signals; face crop and frame sampling for video.
- **Audio deepfake** — Three models (CNN-LSTM, TCN, TCN-LSTM) with chunk-level majority vote; supports common formats (WAV, MP3, WebM, etc.).
- **Call fraud pipeline** — Speech-to-text (Whisper) → typed PII redaction → rule engine + playbook matching + Gemini; hybrid risk score (rules + playbook + LLM) and merged evidence; scam type and recommendation.
- **Live capture (web)** — Share screen or tab; real-time face and audio analysis。
- **Privacy-first** — PII is redacted before any LLM call; backend can run on a confidential VM (TEE); no storage of uploaded media by default.
- **Cross-platform** — Flutter app on iOS, Android, and web; same API for upload and live flows.

---

## 3. Architecture

The client (Flutter app on mobile or web) sends media to a FastAPI backend running inside a **Trusted Execution Environment (TEE)**. The backend routes by media type, runs ensemble deepfake models and (for audio) a fraud pipeline, then returns scores and evidence.


![Kitahack 2026 (1)](https://github.com/user-attachments/assets/bf1f36cf-e4f4-41ce-8a53-35c4c981e53f)

- **Client:** Flutter (iOS, Android, Web). Uploads images/videos/audio or streams live capture; displays verdicts, confidence, and fraud risk (e.g. risk level, scam type, evidence).
- **TEE / Backend:** FastAPI. **Image** → face crop (OpenCV) → visual deepfake ensemble (Xception + ViT, with EfficientNet tiebreaker when uncertain). **Video** → frame extraction → same visual pipeline. **Audio** → deepfake ensemble (CNN-LSTM, TCN, TCN-LSTM) and, in parallel, **fraud pipeline**: Speech-to-Text → PII redaction → Rule engine + Playbook matching + Gemini → hybrid risk score and evidence.
- **Outputs:** Deepfake verdict + confidence band; fraud risk level, scam type, and merged evidence from rules, playbooks, and Gemini.

---

## 4. Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Flutter (Dart) — mobile (iOS/Android) and web|
| **Backend** | FastAPI (Python 3.x), Uvicorn |
| **Visual deepfake** | OpenCV (face detection/crop), PyTorch — FaceForge (Xception), ViT (Vision Transformer), EfficientNet-B4 (tiebreaker) |
| **Audio deepfake** | PyTorch — CNN-LSTM, TCN, TCN-LSTM; 16 kHz mono; soundfile / PyAV for decoding |
| **Fraud pipeline** | Whisper (local STT), custom PII filter (typed redaction), rule engine (bilingual EN/MY), playbook matcher (token overlap), Google Gemini (LLM) |
| **Deployment** | Backend on confidential VM  (TEE)|

---

## 5. System Overview


```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                                                                 │
│  POST /predict          → Image deepfake detection             │
│  POST /predict-video    → Video deepfake detection             │
│  POST /predict-audio    → Audio deepfake detection             │
│  POST /analyze-fraud    → Call fraud analysis pipeline         │
└─────────────────────────────────────────────────────────────────┘
```

| Capability | Models | Approach |
|------------|--------|----------|
| Image deepfake | FaceForge (XceptionNet) + ViT + EfficientNet-B4 | Logit-stacking ensemble + fallback tiebreaker |
| Video deepfake | Same 3 models, frame-sampled | Per-frame ensemble, video-level aggregation |
| Audio deepfake | CNN-LSTM + TCN + TCN-LSTM | Chunk-based majority vote |
| Call fraud | Whisper STT + PII filter + Rules + Playbooks + Gemini | Hybrid risk engine (3 signals → final score) |

---

## 6. Visual Deepfake Detection (Image & Video)

### Models

| Model | Architecture | Role |
|-------|--------------|------|
| **Champion** | FaceForge (XceptionNet) | Strong on reals and common fakes |
| **Challenger** | ViT (fine-tuned FF++ two-stage) | Balanced across forgery types |
| **Fallback** | EfficientNet-B4 (fine-tuned Celeb-DF) | Tiebreaker for uncertain cases |

### Ensemble Strategy

**Stage 1 — Primary (logit stacking):**

```
z = 0.25 × logit(champion_p_fake) + 1.0 × logit(challenger_p_fake) + 2.5
final_p_fake = sigmoid(z)
```

**Stage 2 — Fallback tiebreaker (only when uncertain):**

When `0.35 ≤ final_p_fake ≤ 0.65`, the Fallback model is blended in:

```
z = 0.25×logit(champ) + 1.0×logit(chall) + 0.5×logit(fallback) + 2.5
```

**Why two stages?** On Celeb-DF fakes, Champion+Challenger often land in 0.35–0.65. EfficientNet (trained on Celeb-DF) confidently identifies these as fake — breaking the tie correctly.

### Confidence Band

| `final_p_fake` | Verdict | Band |
|---|---|---|
| < 0.20 | REAL | HIGH |
| [0.20, 0.35) | REAL | MEDIUM |
| [0.35, 0.65] | UNCERTAIN | LOW |
| (0.65, 0.80] | FAKE | MEDIUM |
| > 0.80 | FAKE | HIGH |

### Video Processing

- Up to 10 evenly-spaced frames from the middle 80% of the video.
- Each frame scored independently; final verdict from `mean(final_p_fake)`.
- If >50% of frames have no detectable face → whole video is `UNCERTAIN`.

---

## 7. Audio Deepfake Detection

Detects **AI-synthesised or cloned voices** using three independently-trained models operating on raw audio waveforms.

### Models

| Model | Architecture | Description |
|-------|--------------|-------------|
| **CNN-LSTM** | Convolutional + LSTM | Captures local patterns and temporal sequences |
| **TCN** | Temporal Convolutional Network | Dilated causal convolutions for long-range context |
| **TCN-LSTM** | TCN + LSTM hybrid | Combines both temporal approaches |

All three models are trained on 16 kHz mono audio and classify 2-second chunks as **Real** or **Fake**.

### Inference Pipeline

```
Audio file
    → decode (soundfile / PyAV fallback for WebM, MP3, etc.)
    → resample to 16 kHz mono
    → split into 2-second chunks (max 10 chunks, evenly spaced)
    → each model scores each chunk independently
    → chunk-level majority vote → per-model prediction
    → ensemble: majority vote across 3 models → final verdict
```

### Supported Audio Formats

`.wav` · `.mp3` · `.ogg` · `.flac` · `.m4a` · `.webm` · `.aac`

### API Response (`POST /predict-audio`)

```json
{
  "request_id": "uuid",
  "media_type": "audio",
  "verdict": "FAKE",
  "confidence_band": "HIGH",
  "final_p_fake": 0.87,
  "uncertain": false,
  "decision_path": "majority_vote",
  "reasons": ["models_agree", "high_confidence"],
  "advice": {
    "why": "Strong signs of manipulation detected — all models agree with high confidence",
    "next_steps": ["Don't share OTP or bank info", "Verify identity via an official channel"]
  },
  "models": [
    {"name": "CNN-LSTM",  "p_fake": 0.91, "verdict": "FAKE", "chunks": 5},
    {"name": "TCN",       "p_fake": 0.84, "verdict": "FAKE", "chunks": 5},
    {"name": "TCN-LSTM",  "p_fake": 0.85, "verdict": "FAKE", "chunks": 5}
  ],
  "ensemble_summary": {"voted_fake": 3, "voted_real": 0, "total": 3},
  "privacy": {"stored_media": false},
  "timing_ms": {"total": 340}
}
```

---

## 8. Call Fraud Detection (Advanced)

Detects **phone scams and social engineering attacks** from call audio. Combines three independent signals into a single hybrid risk score — no single point of failure.

> **Privacy-first design:** Audio is transcribed **locally** (Whisper, no cloud STT). Sensitive data is **redacted before any LLM call**. Only the sanitised transcript is sent to Gemini.

### Pipeline

```
Audio
  │
  ▼
[1] Speech-to-Text (Whisper, local)
  │   → transcript_raw
  │
  ▼
[2] PII Firewall (typed redaction, offline)
  │   → replaces phone/email/OTP/card/NRIC/name → [PHONE], [OTP], etc.
  │   → transcript_filtered  +  redacted[]
  │
  ├──[3a] Rule Engine (offline, deterministic)
  │         → matches 10 rule categories, produces rule_score (0–100)
  │         → extracts evidence quotes
  │
  ├──[3b] Playbook Matching (offline, RAG-lite)
  │         → token overlap vs 8 known scam scripts
  │         → returns similarity scores + matched phrases
  │
  └──[3c] Gemini Analysis (LLM)
            → classifies scam_type, extracts indicators + evidence
            → returns risk_score, confidence, recommendation
  │
  ▼
[4] Hybrid Risk Engine
      final_risk_score = 0.35×rule_score + 0.20×playbook_score + 0.45×gemini_score
      → final_risk_level: low / medium / high
      → merged evidence from all 3 sources
```

### What Makes This Unique

**1. PII Firewall with typed placeholders**

Instead of a generic `[REDACTED]`, each sensitive item is labelled by type so Gemini retains analytical context:

| Detected | Replaced with |
|----------|--------------|
| Phone number | `[PHONE]` |
| Email address | `[EMAIL]` |
| OTP / verification code | `[OTP]` |
| Card number | `[CARD]` |
| Malaysia IC / NRIC | `[NRIC]` |
| Name disclosure | `[NAME]` |
| Password / passcode | `[PASSWORD]` |
| Bank account | `[ACCOUNT]` |

Overlapping patterns are merged (no double-replacement). Malaysian-specific formats (NRIC, `+60` phones, Malay phrases) are covered.

**2. Rule-based heuristics (offline, zero-latency)**

10 rule categories with weighted scoring:

| Rule | Weight | Detects |
|------|--------|---------|
| `otp_request` | 35 | "Give me the OTP / verification code" |
| `urgent_transfer` | 30 | "Transfer now / segera pindah" |
| `impersonation` | 25 | Bank Negara / PDRM / MCMC callers |
| `remote_access` | 25 | AnyDesk / TeamViewer requests |
| `data_harvest` | 20 | IC/MyKad/passport data requests |
| `lottery_scam` | 20 | Prize/lucky draw announcements |
| `investment_scam` | 20 | Guaranteed returns / crypto |
| `parcel_scam` | 15 | Detained package / customs fee |
| `pressure_tactics` | 15 | Arrest warrant / secrecy demands |
| `loan_scam` | 15 | Upfront fee / instant loan |

Rules are bilingual (English + Malay) and run entirely **on-device** — no API call, near-instant.

**3. Scam Playbook Matching (RAG-lite, offline)**

Token overlap similarity against 8 known scam script templates (current):
- Police / Bank Impersonation
- Tech Support / Remote Access
- Investment / Crypto
- OTP / Credential Phishing
- Parcel / Customs
- Romance / Pig-Butchering
- Loan Scam
- Job / Task Scam

**4. Gemini LLM Reasoning**

Receives the PII-filtered transcript + rule context. Returns:
- `scam_type` (enum: phishing / impersonation / investment / etc.)
- `risk_score` 0–100
- `confidence` 0.0–1.0
- `indicators` (list of detected fraud signals)
- `evidence` (quoted snippets with reasons)
- `recommendation` (one actionable sentence)

**5. Hybrid Risk Score**

```
final_risk_score = 0.35 × rule_score
                 + 0.20 × playbook_score
                 + 0.45 × gemini_score
```

| Score | Risk Level |
|-------|-----------|
| 0–34 | low |
| 35–64 | medium |
| 65–100 | high |

### API Response (`POST /analyze-fraud`)

```json
{
  "request_id": "uuid",
  "transcript_raw": "Give me your OTP right now, this is Bank Negara officer calling.",
  "transcript_filtered": "Give me your [OTP] right now, this is [NAME] officer calling.",
  "redacted": [
    {"original": "OTP", "label": "OTP"},
    {"original": "Bank Negara officer", "label": "NAME"}
  ],
  "risk_level": "high",
  "risk_score": 82,
  "scam_type": "impersonation",
  "signals": {
    "rule_score": 60,
    "matched_rules": ["otp_request", "impersonation"],
    "playbook_matches": [
      {"scam_type": "phishing", "label": "OTP / Credential Phishing", "similarity": 0.45, "matched_phrases": ["Give me the OTP"]},
      {"scam_type": "impersonation", "label": "Police / Bank Impersonation", "similarity": 0.30, "matched_phrases": ["Bank Negara"]}
    ],
    "gemini": {
      "risk_level": "high",
      "risk_score": 95,
      "confidence": 0.92,
      "scam_type": "impersonation",
      "summary": "Caller impersonates a Bank Negara officer and requests OTP.",
      "indicators": ["Authority impersonation", "OTP harvesting", "Urgency"],
      "recommendation": "Hang up immediately — Bank Negara never calls to request OTPs."
    }
  },
  "evidence": [
    {"quote": "Give me your OTP right now", "reason": "Requests OTP or verification code"},
    {"quote": "Bank Negara officer calling", "reason": "Impersonates authority (bank/police/government)"}
  ],
  "privacy": {"stored_media": false},
  "timing_ms": {"total": 2100}
}
```

---

## 9. API Reference

### `POST /predict` — Image deepfake

```json
{
  "request_id": "uuid",
  "media_type": "image",
  "verdict": "FAKE",
  "confidence_band": "HIGH",
  "final_p_fake": 0.91,
  "uncertain": false,
  "decision_path": "primary_ensemble",
  "reasons": ["models_agree", "high_confidence"],
  "signals": {
    "face_found": true,
    "face_bbox": {"x": 0.12, "y": 0.08, "w": 0.45, "h": 0.60},
    "quality_warning": null,
    "disagreement": 0.08
  },
  "models": [
    {"name": "FaceForge-Xception", "role": "champion",   "p_fake": 0.94, "used": true},
    {"name": "ViT-FF++TwoStage",   "role": "challenger", "p_fake": 0.86, "used": true},
    {"name": "EffNetB4-CelebDF",   "role": "fallback",   "p_fake": 0.77, "used": false}
  ],
  "privacy": {"stored_media": false},
  "timing_ms": {"total": 620}
}
```

`decision_path`: `primary_ensemble` · `tiebreaker_used` · `low_quality`

`reasons` tags: `no_face_detected` · `low_resolution` · `small_face` · `models_agree` · `models_disagree` · `borderline_score` · `high_confidence` · `tiebreaker_used`

---

### `POST /predict-video` — Video deepfake

```json
{
  "request_id": "uuid",
  "media_type": "video",
  "verdict": "UNCERTAIN",
  "confidence_band": "LOW",
  "final_p_fake": 0.55,
  "uncertain": true,
  "sampling": {"frames_used": 10, "strategy": "middle_80_even"},
  "video_stats": {
    "mean_p_fake": 0.55,
    "median_p_fake": 0.52,
    "variance": 0.08,
    "uncertain_frame_rate": 0.40,
    "confident_fake_frames": 2
  },
  "decision_path": "tiebreaker_used",
  "reasons": ["borderline_score", "models_disagree", "tiebreaker_used"],
  "models_summary": {"champion_avg": 0.10, "challenger_avg": 0.48, "fallback_avg": 0.70},
  "top_suspicious_frames": [
    {"t_sec": 12.4, "p_fake": 0.84},
    {"t_sec": 18.0, "p_fake": 0.79}
  ],
  "privacy": {"stored_media": false},
  "timing_ms": {"total": 4200}
}
```

Additional `reasons` for video: `high_variance` · `low_face_rate` · `consistent_prediction`

---

### `POST /predict-audio` — Audio deepfake

See [Section 7](#7-audio-deepfake-detection) for full response schema.

---

### `POST /analyze-fraud` — Call fraud

See [Section 8](#8-call-fraud-detection-advanced) for full response schema.

---

## 10. Challenges Faced

- **Uncertain band in visual detection** — Champion + Challenger often output mid-range scores (0.35–0.65) on certain fakes (e.g. Celeb-DF). **Approach:** A third model (EfficientNet-B4, trained on Celeb-DF) is invoked only when the primary ensemble is uncertain, acting as a tiebreaker and significantly improving accuracy on those cases.
- **Video quality and face visibility** — Low resolution, blur, or few visible faces lead to inconsistent per-frame scores and "UNCERTAIN" verdicts. **Approach:** Video-level aggregation (e.g. mean score), clear confidence bands, and in-app tips (e.g. "Try better lighting or a closer face") so users understand why the result is uncertain.
- **PII in call audio** — Sending raw transcripts to an LLM would leak phone numbers, OTPs, NRIC. **Approach:** Typed PII redaction (e.g. `[PHONE]`, `[OTP]`, `[NRIC]`) before any LLM call; merge overlapping spans; keep evidence and scam analysis useful without exposing real identifiers.
- **Latency vs accuracy** — Multiple models and the fraud pipeline (STT → rules → playbook → Gemini) add latency. **Approach:** Run audio deepfake and fraud in one request; cache Whisper model; optional one-time analysis in live mode so the UI doesn't hammer the backend every few seconds.

---

## 11. Future Roadmap

- **Models & data** — Add or swap visual/audio models; fine-tune on more diverse deepfake datasets (e.g. additional face-forgery and voice-clone corpora).
- **Detection coverage & precision** — **Image/video/audio:** detect more deepfake types (e.g. more forgery methods and generators) and improve precision so real vs fake is more accurate, including on edge cases (low quality, compression, partial faces). **Fraud:** detect more scam types and patterns (e.g. new playbooks, regional variants); improve precision to reduce false positives and missed scams.
- **Fraud pipeline** — Support more languages and scam playbooks; optional streaming STT for long calls; tune hybrid weights (rules vs playbook vs Gemini) from production feedback.
- **Live capture** — Improve UX for live capturing (e.g. when browser/OS support improves); native mobile screen capture where applicable.
- Implement full features detector accross multiple platform. 
- **Explainability** — Enhance the explainable part so users get more detail on **why** content is classified as real or fake (e.g. which visual or audio cues drove the verdict, per-model contributions, and for video/audio which segments or frames were most suspicious); surface this in the UI and API for better trust and decision-making.
- **Ops** — Audit logs for verdicts and fraud signals; dashboards for model usage (e.g. how often the tiebreaker is used); A/B tests on confidence thresholds.
- **Deployment** — One-click backend + frontend deploy (e.g. Docker, cloud runbooks); rate limiting and auth for public APIs.

---

**Team 3004** — Lim Fang Yee, Lee Wai Yee