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

### 3. Video-level decision

- We extract up to 10 evenly spaced frames (middle 80% of the video).
- Each frame gets a `final_p_fake` (with or without tiebreaker as above).
- **Video label:** compare **mean**(`final_p_fake`) to 0.5: Real if mean ≤ 0.5, Fake otherwise.

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

- **`POST /predict`** — image upload → prediction (Real / Fake / Uncertain), confidence, per-model outputs, optional forensic features.
- **`POST /predict-video`** — video upload → same for video (frame-level + video-level decision, temporal stats).
- **`POST /predict-audio`** — audio deepfake detection (separate pipeline; not covered in this README).

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
| **Primary ensemble** | Champion + Challenger combined via logit stacking; no Fallback in the formula by default. |
| **Uncertain band** | When primary `final_p_fake` is in [0.35, 0.65], the decision is treated as uncertain. |
| **Fallback tiebreaker** | In the uncertain band, blend in Fallback (EfficientNet) with weight 0.5 to get the final `final_p_fake`. |
| **Why it helps** | On **FF++ reals**, Fallback often pulls mistaken “Fake” back to Real. On **Celeb-DF fakes**, Fallback (trained on Celeb-DF) often pushes mistaken “Real” to Fake. |

This README describes the **design**: the current codebase may implement only the primary two-model stacking by default; the tiebreaker logic can be added where `final_p_fake` is computed (image and video paths) using the uncertain band and the three-model formula above.
