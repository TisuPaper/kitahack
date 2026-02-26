# Kitahack — Deepfake Detection API

A multi-model deepfake detection system for **images** and **videos**, with an optional **fallback tiebreaker** when the primary ensemble is uncertain.

---

## Overview

The detector runs **three** image-based models in parallel:

| Model | Architecture | Source | Role |
|-------|--------------|--------|------|
| **Champion** | FaceForge (XceptionNet) | HuggingFace | Strong on reals and some fakes (e.g. Deepfakes) |
| **Challenger** | ViT (Vision Transformer) | Fine-tuned (FF++ / two-stage) | Balanced across forgery types |
| **Fallback** | EfficientNet-B4 | Fine-tuned (Celeb-DF) | Used only when the primary decision is **uncertain** |

The **primary decision** comes from a **logit-stacking** blend of Champion and Challenger. When that blend falls in an **uncertain range**, the **Fallback** is used as a **tiebreaker** to push the final score toward Real or Fake.

---

## How It Works

### 1. Primary ensemble (logit stacking)

For each frame (or image), we get `p_fake` from Champion and Challenger and combine them with a learned formula:

```
z = 0.25 × logit(champion_p_fake) + 1.0 × logit(challenger_p_fake) + 2.5
final_p_fake = sigmoid(z)
```

- **Decision:** `Real` if `final_p_fake ≤ 0.5`, else `Fake`.
- Weights and bias were tuned on 80 FF++ C23 videos (40 real + 40 fake).

### 2. Fallback tiebreaker (when uncertain)

If the primary score is in the **uncertain band** (e.g. `0.35 ≤ final_p_fake ≤ 0.65`), we **blend in the Fallback** model so it can break the tie:

```
z = 0.25×logit(champ) + 1.0×logit(chall) + 0.5×logit(fallback) + 2.5
final_p_fake = sigmoid(z)
```

- **When Fallback helps**
  - **Real video, primary says Fake:** Fallback (trained on Celeb-DF reals) often has low `p_fake` and pulls the score back to Real.
  - **Fake video, primary says Real:** On **Celeb-DF** fakes, Fallback (trained on Celeb-DF) often has high `p_fake` and pushes the score to Fake.

So: **uncertain → use Fallback as tiebreaker**; otherwise use only Champion + Challenger.

### 3. Confidence band

After computing `final_p_fake`, the system assigns a **confidence band** based on distance from the 0.5 decision boundary (calibrated on FF++ C23 eval data):

| `final_p_fake` | Verdict | Confidence band |
|---|---|---|
| < 0.20 | REAL | HIGH |
| [0.20, 0.35) | REAL | MEDIUM |
| [0.35, 0.65] | UNCERTAIN | LOW |
| (0.65, 0.80] | FAKE | MEDIUM |
| > 0.80 | FAKE | HIGH |

The LOW band coincides exactly with the tiebreaker zone — when confidence is LOW it means the primary ensemble was genuinely split.

### 4. Video-level decision

- We extract up to 10 evenly spaced frames (middle 80% of the video).
- Each frame gets a `final_p_fake` (with or without tiebreaker as above), a per-frame verdict, and a timestamp.
- **Video verdict:** `_verdict(mean(final_p_fake))` using the same thresholds as above.
- If more than 50% of frames have quality issues (no face / small face), the whole video is `UNCERTAIN`.

---

## Showcase Cases

### Case A: Real video — primary says Fake, Fallback corrects to Real

- **Dataset:** FaceForensics++ C23  
- **Video:** `original/000.mp4` (ground truth: **Real**)
- **Without tiebreaker:** Champion says real (p_fake ~0.01), Challenger is unsure (~0.18–0.38). Primary score ≈ 0.54 → **Fake** (wrong).
- **With tiebreaker:** All frames are in the uncertain band. Fallback says real (p_fake ~0.02–0.05). Blended score ≈ 0.19 → **Real** (correct).

Path (if FF++ is cached):  
`~/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/FaceForensics++_C23/original/000.mp4`

### Case B: Fake video — primary says Real, Fallback corrects to Fake

- **Dataset:** Celeb-DF v2  
- **Video:** `Celeb-synthesis/id0_id4_0004.mp4` (ground truth: **Fake**)
- **Without tiebreaker:** Champion says real (p_fake ~0.001–0.02), Challenger uncertain (~0.14–0.43). Primary score ≈ 0.44 → **Real** (wrong).
- **With tiebreaker:** Frames in uncertain band; Fallback (trained on Celeb-DF) says fake (p_fake ~0.72–0.96). Blended score ≈ 0.71 → **Fake** (correct).

Path (if Celeb-DF is cached):  
`~/.cache/kagglehub/datasets/reubensuju/celeb-df-v2/versions/1/Celeb-synthesis/id0_id4_0004.mp4`

---

## API

### `POST /predict` — image

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
    {"name": "FaceForge-Xception", "role": "champion",  "p_fake": 0.94, "used": true},
    {"name": "ViT-FF++TwoStage",   "role": "challenger", "p_fake": 0.86, "used": true},
    {"name": "EffNetB4-CelebDF",   "role": "fallback",   "p_fake": 0.77, "used": false}
  ],

  "privacy": {"stored_media": false},
  "timing_ms": {"total": 620, "champion": 220, "challenger": 260, "fallback": 0}
}
```

**`decision_path` values:** `primary_ensemble` · `tiebreaker_used` · `low_quality`

**`reasons` tags:** `no_face_detected` · `low_resolution` · `small_face` · `models_agree` · `models_disagree` · `borderline_score` · `high_confidence` · `tiebreaker_used`

---

### `POST /predict-video` — video

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

  "models_summary": {
    "champion_avg": 0.10,
    "challenger_avg": 0.48,
    "fallback_avg": 0.70
  },

  "top_suspicious_frames": [
    {"t_sec": 12.4, "p_fake": 0.84},
    {"t_sec": 18.0, "p_fake": 0.79}
  ],

  "privacy": {"stored_media": false},
  "timing_ms": {"total": 4200},
  "warnings": []
}
```

**Additional `reasons` tags for video:** `high_variance` · `low_face_rate` · `consistent_prediction`

---

### `POST /predict-audio`

Audio deepfake detection (separate pipeline; not covered in this README).

---

See `/docs` when the server is running for full OpenAPI spec.

---

## Running the API

```bash
# From project root
python -m uvicorn app.main:app --reload
```

Models are loaded from:

- **Champion:** HuggingFace `huzaifanasirrr/faceforge-detector` (or local cache).
- **Challenger:** `models/vit_finetuned_twostage/` (or HuggingFace fallback).
- **Fallback:** `models/efficientnet_finetuned_ffpp/` (optional; if missing, only Champion + Challenger are used).

---

## Evaluation and Tuning

- **`evaluate_ensemble.py`** — Runs all three models on FF++ C23, saves raw per-frame probabilities and runs a grid search over ensemble weights.
- **`analyze_results.py`** — Analyzes saved results (distributions, per-model thresholds, calibrated blending, stacking).
- **`optimize_video_level.py`** — Optimizes video-level stacking coefficients and threshold.

Raw results (e.g. `eval_raw_results.json`, `eval_celebdf_results.json`) can be re-used without re-running inference.

---

## Summary

| Idea | Description |
|------|-------------|
| **Primary ensemble** | Champion + Challenger combined via logit stacking (weights 0.25 / 1.0, bias 2.5). |
| **Uncertain band** | When primary `final_p_fake` ∈ [0.35, 0.65] the decision is treated as uncertain. |
| **Fallback tiebreaker** | In the uncertain band, Fallback (EfficientNet) is blended in with weight 0.5 to break the tie. |
| **Confidence band** | `HIGH` (p < 0.20 or > 0.80) · `MEDIUM` (0.20–0.35 or 0.65–0.80) · `LOW` (0.35–0.65). |
| **Reason tags** | Rule-based tags explain *why* a verdict was reached without needing an AI explanation. |
| **Why tiebreaker helps** | On **FF++ reals**, Fallback pulls mistaken "Fake" back to Real. On **Celeb-DF fakes**, Fallback pushes mistaken "Real" to Fake. |
