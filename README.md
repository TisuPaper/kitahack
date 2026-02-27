# Realitic — AI Media & Call Fraud Detection

A multi-modal AI fraud detection system covering **image**, **video**, and **audio deepfakes**, plus a **call fraud analysis** pipeline that transcribes speech, redacts private data, and uses a hybrid risk engine to detect scams.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Visual Deepfake Detection (Image & Video)](#2-visual-deepfake-detection-image--video)
3. [Audio Deepfake Detection](#3-audio-deepfake-detection)
4. [Call Fraud Detection (Advanced)](#4-call-fraud-detection-advanced)
5. [API Reference](#5-api-reference)
6. [Running the Backend](#6-running-the-backend)
7. [Evaluation & Tuning](#7-evaluation--tuning)

---

## 1. System Overview

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

## 2. Visual Deepfake Detection (Image & Video)

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

### Showcase Cases

**Case A: Real video corrected by Fallback**
- Video: `FaceForensics++/original/000.mp4` (Real)
- Without tiebreaker: Challenger uncertain → primary score 0.54 → **Fake (wrong)**
- With tiebreaker: Fallback says real → blended score 0.19 → **Real (correct)**

**Case B: Fake video corrected by Fallback**
- Video: `Celeb-synthesis/id0_id4_0004.mp4` (Fake)
- Without tiebreaker: Champion says real → primary score 0.44 → **Real (wrong)**
- With tiebreaker: Fallback (Celeb-DF trained) says fake → blended score 0.71 → **Fake (correct)**

---

## 3. Audio Deepfake Detection

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

## 4. Call Fraud Detection (Advanced)

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

Token overlap similarity against 8 known scam script templates:
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

### Setup (Quick)

```bash
# 1. Install deps
pip install -r app/requirements-fraud.txt

# 2. Install ffmpeg (for Whisper)
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Linux

# 3. Set Gemini API key (get one at aistudio.google.com/apikey)
echo "GEMINI_API_KEY=your_key" > app/.env.local

# 4. Run backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Test
curl -X POST http://localhost:8000/analyze-fraud -F "file=@call.wav"
```

Full setup guide: [`app/docs/SETUP_FRAUD_DETECTION.md`](app/docs/SETUP_FRAUD_DETECTION.md)

---

## 5. API Reference

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

See [Section 3](#3-audio-deepfake-detection) for full response schema.

---

### `POST /analyze-fraud` — Call fraud

See [Section 4](#4-call-fraud-detection-advanced) for full response schema.

---

## 6. Running the Backend

```bash
# From project root
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or
make run-backend
```

**Model locations:**

| Model | Source |
|-------|--------|
| Champion (FaceForge) | HuggingFace `huzaifanasirrr/faceforge-detector` |
| Challenger (ViT) | `models/vit_finetuned_twostage/` |
| Fallback (EfficientNet) | `models/efficientnet_finetuned_ffpp/` |
| Audio models | `models/cnn-lstm_audio_classifier.pth`, `tcn_audio_classifier.pth`, `tcn-lstm_audio_classifier.pth` |

**Environment variables (optional):**

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Required for `/analyze-fraud` |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `WHISPER_MODEL` | `base` | Whisper size: `tiny`/`base`/`small`/`medium`/`large` |
| `FRAUD_USE_GEMINI_STT` | off | Use Gemini for transcription instead of Whisper |
| `FRAUD_PII_PATTERNS_FILE` | — | Path to extra PII regex patterns (`LABEL\|regex` per line) |

Set via `app/.env.local` (loaded automatically on startup).

---

## 7. Evaluation & Tuning

| Script | Purpose |
|--------|---------|
| `evaluate_ensemble.py` | Run all 3 visual models on FF++ C23, grid-search ensemble weights |
| `analyze_results.py` | Analyse saved results (distributions, thresholds, calibrated blending) |
| `optimize_video_level.py` | Optimise video-level stacking coefficients and threshold |

Raw results (`eval_raw_results.json`, `eval_celebdf_results.json`) can be reused without re-running inference.

---

## Summary

| Component | Unique Aspect |
|-----------|--------------|
| **Visual ensemble** | Two-stage logit stacking; Fallback only activates when primary is uncertain |
| **Fallback tiebreaker** | Celeb-DF trained EfficientNet corrects errors in the uncertain band |
| **Audio deepfake** | Three independent temporal models; chunk-level majority vote |
| **PII Firewall** | Typed redaction ([PHONE], [OTP] …) preserves context for LLM while protecting privacy |
| **Rule engine** | 10 bilingual (EN+MY) heuristic rules; offline, zero-latency |
| **Playbook matching** | Token-overlap similarity vs 8 known scam scripts; no ML model needed |
| **Hybrid risk score** | Rules 35% + Playbook 20% + Gemini 45% → final score 0–100 |
| **Evidence output** | Quoted snippets with reasons from all three sources for explainable detection |
