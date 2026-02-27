"""
Evaluate all 3 deepfake detection models on FF++ C23 dataset,
then grid-search for the optimal ensemble weights and blending strategy.

Usage:
    python evaluate_ensemble.py

Outputs per-model accuracy, per-forgery-type breakdown, and the best
ensemble configuration found.
"""

import os
import sys
import json
import math
import time
import itertools
import cv2
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/FaceForensics++_C23"
)
VIDEOS_PER_CLASS = 40          # real videos to sample
VIDEOS_PER_FORGERY = 10        # fake videos per forgery type
FRAMES_PER_VIDEO = 5           # frames extracted per video
FORGERY_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
MIN_FACE_SIZE = 80

# ---------------------------------------------------------------------------
# Face cropper (same as main.py)
# ---------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_largest_face(pil_img):
    rgb = np.array(pil_img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return pil_img, False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return pil_img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return pil_img, False
    pad = int(0.25 * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
    return Image.fromarray(rgb[y1:y2, x1:x2]), True


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path, n_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    start = int(total * 0.10)
    end = int(total * 0.90)
    if end <= start:
        start, end = 0, total
    n = min(n_frames, end - start)
    if n <= 0:
        cap.release()
        return []
    indices = [start + int(i * (end - start) / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            cropped, found = crop_largest_face(pil)
            if found:
                frames.append(cropped)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Load models (reuses app/model_loader.py logic)
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

sys.path.insert(0, os.path.dirname(__file__))
from app.model_loader import (
    champion, champion_preprocess,
    challenger_model, challenger_processor,
    fallback_model, fallback_processor,
    device,
)

assert fallback_model is not None, (
    "EfficientNet fallback model not loaded! "
    "Check models/efficientnet_finetuned_ffpp/ exists."
)
print(f"\nAll 3 models loaded on {device}.\n")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_champion(img):
    x = champion_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(champion(x), dim=1)
    return float(probs[0][1])  # p_fake


def run_challenger(img):
    inputs = challenger_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(challenger_model(**inputs).logits, dim=1)
    id2label = challenger_model.config.id2label
    label_map = {"Realism": "fake_inv", "Deepfake": "fake", "Real": "real", "Fake": "fake"}
    p_fake = 0.0
    for idx_str, ln in id2label.items():
        c = label_map.get(ln, ln.lower())
        pv = float(probs[0][int(idx_str)])
        if c == "fake":
            p_fake = pv
    return p_fake


def run_fallback(img):
    inputs = fallback_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(fallback_model(**inputs).logits, dim=1)
    id2label = fallback_model.config.id2label
    label_map = {"Realism": "real", "Deepfake": "fake", "Real": "real", "Fake": "fake"}
    p_fake = 0.0
    for idx_str, ln in id2label.items():
        c = label_map.get(ln, ln.lower())
        pv = float(probs[0][int(idx_str)])
        if c == "fake":
            p_fake = pv
    return p_fake


# ---------------------------------------------------------------------------
# Blending functions
# ---------------------------------------------------------------------------
def _logit(p):
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def blend_logits(probs, weights):
    return _sigmoid(sum(w * _logit(p) for p, w in zip(probs, weights)))


def blend_linear(probs, weights):
    return sum(w * p for p, w in zip(probs, weights))


def blend_geometric(probs, weights):
    log_p = sum(w * math.log(max(p, 1e-10)) for p, w in zip(probs, weights))
    return math.exp(log_p)


def blend_max(probs, weights):
    return max(w * p for p, w in zip(probs, weights)) / max(weights)


BLEND_FUNCTIONS = {
    "logit": blend_logits,
    "linear": blend_linear,
    "geometric": blend_geometric,
}


# ---------------------------------------------------------------------------
# Collect dataset samples
# ---------------------------------------------------------------------------
print("=" * 70)
print("COLLECTING DATASET SAMPLES")
print("=" * 70)

samples = []  # list of (video_path, label, forgery_type)

# Real videos
real_dir = os.path.join(DATASET_ROOT, "original")
real_videos = sorted([f for f in os.listdir(real_dir) if f.endswith(".mp4")])
for v in real_videos[:VIDEOS_PER_CLASS]:
    samples.append((os.path.join(real_dir, v), 0, "original"))
print(f"  Real: {min(VIDEOS_PER_CLASS, len(real_videos))} videos")

# Fake videos per forgery type
for forgery in FORGERY_TYPES:
    fdir = os.path.join(DATASET_ROOT, forgery)
    if not os.path.exists(fdir):
        print(f"  {forgery}: MISSING")
        continue
    vids = sorted([f for f in os.listdir(fdir) if f.endswith(".mp4")])
    for v in vids[:VIDEOS_PER_FORGERY]:
        samples.append((os.path.join(fdir, v), 1, forgery))
    print(f"  {forgery}: {min(VIDEOS_PER_FORGERY, len(vids))} videos")

total_videos = len(samples)
print(f"\n  Total: {total_videos} videos, ~{total_videos * FRAMES_PER_VIDEO} frames expected\n")


# ---------------------------------------------------------------------------
# Run inference on all samples
# ---------------------------------------------------------------------------
print("=" * 70)
print("RUNNING INFERENCE (this will take a while on CPU)")
print("=" * 70)

results = []  # list of dicts per frame

t0 = time.time()
for si, (vpath, label, forgery) in enumerate(samples):
    frames = extract_frames(vpath, FRAMES_PER_VIDEO)
    if not frames:
        continue

    for fi, img in enumerate(frames):
        p_champ = run_champion(img)
        p_chall = run_challenger(img)
        p_fall = run_fallback(img)

        results.append({
            "video": os.path.basename(vpath),
            "forgery": forgery,
            "label": label,
            "frame_idx": fi,
            "champion": p_champ,
            "challenger": p_chall,
            "fallback": p_fall,
        })

    elapsed = time.time() - t0
    done = si + 1
    eta = (elapsed / done) * (total_videos - done) if done > 0 else 0
    print(
        f"  [{done}/{total_videos}] {forgery:16s} | "
        f"{len(frames)} frames | "
        f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
    )

elapsed_total = time.time() - t0
print(f"\nInference complete: {len(results)} frames in {elapsed_total:.1f}s\n")


# ---------------------------------------------------------------------------
# Save raw results for later analysis (cache for standalone analyze/optimize)
# ---------------------------------------------------------------------------
RAW_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_raw_results.json")
with open(RAW_RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Raw results saved to {RAW_RESULTS_PATH}\n")

# ---------------------------------------------------------------------------
# Optional: run analysis and optimization in-process (direct pass, no re-read)
# ---------------------------------------------------------------------------
try:
    import analyze_results
    analyze_results.analyze(results)
except Exception as e:
    print(f"analyze_results skipped: {e}\n")
try:
    import optimize_video_level
    optimize_video_level.optimize(results)
except Exception as e:
    print(f"optimize_video_level skipped: {e}\n")


# ---------------------------------------------------------------------------
# Individual model evaluation
# ---------------------------------------------------------------------------
print("=" * 70)
print("INDIVIDUAL MODEL PERFORMANCE (frame-level, threshold=0.5)")
print("=" * 70)

labels = np.array([r["label"] for r in results])

for model_name in ["champion", "challenger", "fallback"]:
    preds = np.array([1 if r[model_name] > 0.5 else 0 for r in results])
    acc = np.mean(preds == labels) * 100
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1)

    print(f"\n  {model_name.upper()} (XceptionNet/ViT/EfficientNet)")
    print(f"    Accuracy:  {acc:.1f}%")
    print(f"    Precision: {precision:.1f}%  Recall: {recall:.1f}%  F1: {f1:.1f}%")
    print(f"    TP={tp} TN={tn} FP={fp} FN={fn}")

    # Per-forgery breakdown
    for forgery in ["original"] + FORGERY_TYPES:
        mask = np.array([r["forgery"] == forgery for r in results])
        if mask.sum() == 0:
            continue
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        sub_acc = np.mean(sub_preds == sub_labels) * 100
        print(f"      {forgery:20s}: {sub_acc:.1f}% ({mask.sum()} frames)")


# ---------------------------------------------------------------------------
# Video-level evaluation (majority vote across frames)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("INDIVIDUAL MODEL PERFORMANCE (video-level, majority vote)")
print("=" * 70)

video_groups = defaultdict(list)
for r in results:
    key = (r["video"], r["forgery"], r["label"])
    video_groups[key].append(r)

for model_name in ["champion", "challenger", "fallback"]:
    correct = 0
    total = 0
    forgery_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for (vid, forgery, label), frames in video_groups.items():
        fake_votes = sum(1 for f in frames if f[model_name] > 0.5)
        pred = 1 if fake_votes > len(frames) / 2 else 0
        total += 1
        if pred == label:
            correct += 1
        forgery_stats[forgery]["total"] += 1
        if pred == label:
            forgery_stats[forgery]["correct"] += 1

    acc = correct / max(total, 1) * 100
    print(f"\n  {model_name.upper()}: {acc:.1f}% ({correct}/{total} videos)")
    for forgery in ["original"] + FORGERY_TYPES:
        s = forgery_stats[forgery]
        if s["total"] > 0:
            facc = s["correct"] / s["total"] * 100
            print(f"      {forgery:20s}: {facc:.1f}% ({s['correct']}/{s['total']})")


# ---------------------------------------------------------------------------
# Grid search: best ensemble weights + blending strategy
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GRID SEARCH: OPTIMAL ENSEMBLE WEIGHTS")
print("=" * 70)

# Generate weight triplets that sum to 1.0 (step=0.05)
STEP = 0.05
weight_grid = []
for w1 in np.arange(0.0, 1.0 + STEP / 2, STEP):
    for w2 in np.arange(0.0, 1.0 - w1 + STEP / 2, STEP):
        w3 = 1.0 - w1 - w2
        if w3 >= -0.001:
            w3 = max(0.0, w3)
            weight_grid.append((round(w1, 2), round(w2, 2), round(w3, 2)))

print(f"  Testing {len(weight_grid)} weight combinations x {len(BLEND_FUNCTIONS)} blending strategies")
print(f"  = {len(weight_grid) * len(BLEND_FUNCTIONS)} total configurations\n")

# Pre-extract raw probs for speed
champ_probs = [r["champion"] for r in results]
chall_probs = [r["challenger"] for r in results]
fall_probs = [r["fallback"] for r in results]
true_labels = [r["label"] for r in results]

best_overall = {"acc": 0, "f1": 0}
all_configs = []

for blend_name, blend_fn in BLEND_FUNCTIONS.items():
    best_for_blend = {"acc": 0}

    for w_champ, w_chall, w_fall in weight_grid:
        weights = [w_champ, w_chall, w_fall]

        # Frame-level predictions
        preds = []
        for i in range(len(results)):
            probs = [champ_probs[i], chall_probs[i], fall_probs[i]]
            try:
                p_fake = blend_fn(probs, weights)
            except (ValueError, ZeroDivisionError, OverflowError):
                p_fake = 0.5
            preds.append(1 if p_fake > 0.5 else 0)

        preds = np.array(preds)
        labs = np.array(true_labels)

        acc = np.mean(preds == labs) * 100
        tp = np.sum((preds == 1) & (labs == 1))
        fp = np.sum((preds == 1) & (labs == 0))
        fn = np.sum((preds == 0) & (labs == 1))
        prec = tp / max(tp + fp, 1) * 100
        rec = tp / max(tp + fn, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1)

        config = {
            "blend": blend_name,
            "w_champion": w_champ,
            "w_challenger": w_chall,
            "w_fallback": w_fall,
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "fp": int(fp),
            "fn": int(fn),
        }
        all_configs.append(config)

        if f1 > best_for_blend.get("f1", 0) or (
            f1 == best_for_blend.get("f1", 0) and acc > best_for_blend.get("acc", 0)
        ):
            best_for_blend = config.copy()

        if f1 > best_overall.get("f1", 0) or (
            f1 == best_overall.get("f1", 0) and acc > best_overall.get("acc", 0)
        ):
            best_overall = config.copy()

    print(
        f"  {blend_name:12s} best: "
        f"w=[{best_for_blend['w_champion']:.2f}, {best_for_blend['w_challenger']:.2f}, {best_for_blend['w_fallback']:.2f}] "
        f"acc={best_for_blend['accuracy']:.1f}% "
        f"f1={best_for_blend['f1']:.1f}% "
        f"fp={best_for_blend['fp']} fn={best_for_blend['fn']}"
    )


# ---------------------------------------------------------------------------
# Also test video-level ensemble with best frame-level weights
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP 10 CONFIGURATIONS (by F1, then Accuracy)")
print("=" * 70)

all_configs.sort(key=lambda c: (c["f1"], c["accuracy"]), reverse=True)
for i, c in enumerate(all_configs[:10]):
    print(
        f"  #{i+1}: {c['blend']:12s} "
        f"w=[{c['w_champion']:.2f}, {c['w_challenger']:.2f}, {c['w_fallback']:.2f}] "
        f"acc={c['accuracy']:.1f}% prec={c['precision']:.1f}% "
        f"rec={c['recall']:.1f}% f1={c['f1']:.1f}% "
        f"fp={c['fp']} fn={c['fn']}"
    )


# ---------------------------------------------------------------------------
# Video-level evaluation of the top configs
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VIDEO-LEVEL EVALUATION OF TOP 5 CONFIGS")
print("=" * 70)

for rank, cfg in enumerate(all_configs[:5]):
    blend_fn = BLEND_FUNCTIONS[cfg["blend"]]
    weights = [cfg["w_champion"], cfg["w_challenger"], cfg["w_fallback"]]

    correct = 0
    total = 0
    forgery_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for (vid, forgery, label), frames in video_groups.items():
        fake_votes = 0
        for f in frames:
            probs = [f["champion"], f["challenger"], f["fallback"]]
            try:
                p_fake = blend_fn(probs, weights)
            except (ValueError, ZeroDivisionError, OverflowError):
                p_fake = 0.5
            if p_fake > 0.5:
                fake_votes += 1
        pred = 1 if fake_votes > len(frames) / 2 else 0
        total += 1
        if pred == label:
            correct += 1
        forgery_stats[forgery]["total"] += 1
        if pred == label:
            forgery_stats[forgery]["correct"] += 1

    vid_acc = correct / max(total, 1) * 100
    print(
        f"\n  #{rank+1} {cfg['blend']} "
        f"w=[{cfg['w_champion']:.2f}, {cfg['w_challenger']:.2f}, {cfg['w_fallback']:.2f}]"
    )
    print(f"    Video-level accuracy: {vid_acc:.1f}% ({correct}/{total})")
    for forgery in ["original"] + FORGERY_TYPES:
        s = forgery_stats[forgery]
        if s["total"] > 0:
            facc = s["correct"] / s["total"] * 100
            print(f"      {forgery:20s}: {facc:.1f}% ({s['correct']}/{s['total']})")


# ---------------------------------------------------------------------------
# Also test threshold tuning for the best config
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("THRESHOLD TUNING FOR BEST CONFIG")
print("=" * 70)

best_cfg = all_configs[0]
blend_fn = BLEND_FUNCTIONS[best_cfg["blend"]]
weights = [best_cfg["w_champion"], best_cfg["w_challenger"], best_cfg["w_fallback"]]

# Compute blended p_fake for all frames
blended_probs = []
for i in range(len(results)):
    probs = [champ_probs[i], chall_probs[i], fall_probs[i]]
    try:
        blended_probs.append(blend_fn(probs, weights))
    except (ValueError, ZeroDivisionError, OverflowError):
        blended_probs.append(0.5)

blended_probs = np.array(blended_probs)
labs = np.array(true_labels)

print(f"  Config: {best_cfg['blend']} w=[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]")
print(f"  {'Threshold':>10s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'FP':>4s}  {'FN':>4s}")
print(f"  {'-'*52}")

best_thresh = 0.5
best_thresh_f1 = 0.0
for thresh in np.arange(0.30, 0.75, 0.025):
    preds = (blended_probs > thresh).astype(int)
    acc = np.mean(preds == labs) * 100
    tp = np.sum((preds == 1) & (labs == 1))
    fp = np.sum((preds == 1) & (labs == 0))
    fn = np.sum((preds == 0) & (labs == 1))
    prec = tp / max(tp + fp, 1) * 100
    rec = tp / max(tp + fn, 1) * 100
    f1 = 2 * prec * rec / max(prec + rec, 1)
    marker = " <--" if f1 > best_thresh_f1 else ""
    if f1 > best_thresh_f1:
        best_thresh_f1 = f1
        best_thresh = thresh
    print(f"  {thresh:10.3f}  {acc:5.1f}%  {prec:5.1f}%  {rec:5.1f}%  {f1:5.1f}%  {fp:4d}  {fn:4d}{marker}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL RECOMMENDED CONFIGURATION")
print("=" * 70)

print(f"""
  Blending strategy : {best_cfg['blend']}
  Champion weight   : {best_cfg['w_champion']:.2f}  (FaceForge XceptionNet)
  Challenger weight  : {best_cfg['w_challenger']:.2f}  (ViT Two-Stage)
  Fallback weight   : {best_cfg['w_fallback']:.2f}  (EfficientNet-B4)
  Best threshold    : {best_thresh:.3f}
  
  Frame-level: acc={best_cfg['accuracy']:.1f}% f1={best_cfg['f1']:.1f}%
  
  To apply these weights, update the following in app/main.py:
    ensemble_weights = [{best_cfg['w_champion']}, {best_cfg['w_challenger']}, {best_cfg['w_fallback']}]
""")

# Save summary
summary = {
    "best_config": best_cfg,
    "best_threshold": round(float(best_thresh), 3),
    "top_10": all_configs[:10],
    "individual_models": {},
}

for model_name in ["champion", "challenger", "fallback"]:
    preds = np.array([1 if r[model_name] > 0.5 else 0 for r in results])
    acc = np.mean(preds == labels) * 100
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    prec = tp / max(tp + fp, 1) * 100
    rec = tp / max(tp + fn, 1) * 100
    f1 = 2 * prec * rec / max(prec + rec, 1)
    summary["individual_models"][model_name] = {
        "accuracy": round(float(acc), 2),
        "precision": round(float(prec), 2),
        "recall": round(float(rec), 2),
        "f1": round(float(f1), 2),
    }

SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "eval_summary.json")
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Full summary saved to {SUMMARY_PATH}")
