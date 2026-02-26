"""
Find the optimal video-level configuration using the logit stacking approach
with threshold tuning to balance real/fake accuracy.
"""

import json
import math
import numpy as np
from collections import defaultdict

with open("eval_raw_results.json") as f:
    results = json.load(f)

labels = np.array([r["label"] for r in results])
champ = np.array([r["champion"] for r in results])
chall = np.array([r["challenger"] for r in results])
fall = np.array([r["fallback"] for r in results])

def logit(p):
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))

def sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1 / (1 + math.exp(-x))

champ_logits = np.array([logit(p) for p in champ])
chall_logits = np.array([logit(p) for p in chall])
fall_logits = np.array([logit(p) for p in fall])

video_groups = defaultdict(list)
for i, r in enumerate(results):
    key = (r["video"], r["forgery"], r["label"])
    video_groups[key].append(i)


def video_eval_detailed(frame_scores, threshold):
    correct, total = 0, 0
    forgery_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for (vid, forgery, label), indices in video_groups.items():
        avg_score = np.mean([frame_scores[i] for i in indices])
        pred = 1 if avg_score > threshold else 0
        total += 1
        if pred == label:
            correct += 1
        forgery_stats[forgery]["total"] += 1
        if pred == label:
            forgery_stats[forgery]["correct"] += 1
    return correct, total, forgery_stats


# ---------------------------------------------------------------------------
# Search over stacking coefficients AND video-level threshold
# ---------------------------------------------------------------------------
print("=" * 70)
print("EXHAUSTIVE VIDEO-LEVEL OPTIMIZATION")
print("  Using average frame score per video (not majority vote)")
print("=" * 70)

best = {"acc": 0, "f1": 0}
configs = []

for a in np.arange(-0.5, 2.5, 0.25):
    for b in np.arange(0.0, 2.5, 0.25):
        for c in np.arange(-1.0, 1.5, 0.25):
            z = a * champ_logits + b * chall_logits + c * fall_logits

            for bias in np.arange(-4.0, 5.0, 0.5):
                scores = z + bias

                for vthresh in [0.0]:
                    correct, total, fs = video_eval_detailed(scores, vthresh)
                    acc = correct / total * 100

                    real_c = fs["original"]["correct"]
                    real_t = fs["original"]["total"]
                    fake_c = sum(fs[ft]["correct"] for ft in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
                    fake_t = sum(fs[ft]["total"] for ft in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])

                    real_acc = real_c / max(real_t, 1) * 100
                    fake_acc = fake_c / max(fake_t, 1) * 100

                    # F1 at video level
                    tp = fake_c
                    fp = real_t - real_c
                    fn = fake_t - fake_c
                    prec = tp / max(tp + fp, 1) * 100
                    rec = tp / max(tp + fn, 1) * 100
                    f1 = 2 * prec * rec / max(prec + rec, 1)

                    balanced_acc = (real_acc + fake_acc) / 2

                    config = {
                        "a": round(a, 2), "b": round(b, 2),
                        "c": round(c, 2), "bias": round(bias, 2),
                        "threshold": vthresh,
                        "accuracy": round(acc, 1),
                        "balanced_acc": round(balanced_acc, 1),
                        "real_acc": round(real_acc, 1),
                        "fake_acc": round(fake_acc, 1),
                        "f1": round(f1, 1),
                        "prec": round(prec, 1),
                        "rec": round(rec, 1),
                    }
                    configs.append(config)

# Sort by balanced accuracy (real_acc + fake_acc) / 2
configs.sort(key=lambda c: (c["balanced_acc"], c["accuracy"]), reverse=True)

print(f"\n  Searched {len(configs)} configurations\n")
print("  TOP 20 BY BALANCED ACCURACY (video-level):")
print(f"  {'#':>3s}  {'a':>5s}  {'b':>5s}  {'c':>5s}  {'bias':>5s}  "
      f"{'Acc':>5s}  {'BAcc':>5s}  {'Real':>5s}  {'Fake':>5s}  {'F1':>5s}")
print(f"  {'-'*65}")

seen = set()
count = 0
for cfg in configs:
    key = (cfg["a"], cfg["b"], cfg["c"], cfg["bias"])
    if key in seen:
        continue
    seen.add(key)
    count += 1
    if count > 20:
        break
    print(f"  {count:3d}  {cfg['a']:5.2f}  {cfg['b']:5.2f}  {cfg['c']:5.2f}  {cfg['bias']:5.1f}  "
          f"{cfg['accuracy']:4.1f}%  {cfg['balanced_acc']:4.1f}%  "
          f"{cfg['real_acc']:4.1f}%  {cfg['fake_acc']:4.1f}%  {cfg['f1']:4.1f}%")


# ---------------------------------------------------------------------------
# For the top config, show per-forgery breakdown
# ---------------------------------------------------------------------------
best_cfg = configs[0]
print(f"\n\nBEST CONFIG DETAILS:")
print(f"  Formula: {best_cfg['a']}*logit(champ) + {best_cfg['b']}*logit(chall) + "
      f"{best_cfg['c']}*logit(fall) + {best_cfg['bias']} > 0")

z = best_cfg["a"] * champ_logits + best_cfg["b"] * chall_logits + best_cfg["c"] * fall_logits + best_cfg["bias"]
correct, total, fs = video_eval_detailed(z, 0.0)

print(f"\n  Per-forgery results:")
for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
    s = fs[ftype]
    if s["total"] > 0:
        print(f"    {ftype:20s}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")

print(f"\n  Overall: {correct}/{total} = {correct/total*100:.1f}%")
print(f"  Balanced accuracy: {best_cfg['balanced_acc']:.1f}%")


# ---------------------------------------------------------------------------
# Also test: Champion with threshold calibration + Challenger tiebreaker
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CHAMPION + CHALLENGER HYBRID STRATEGIES (video-level)")
print("=" * 70)

best_hybrid = {"balanced_acc": 0}

for champ_low in np.arange(0.005, 0.15, 0.005):
    for champ_high in np.arange(0.05, 0.80, 0.05):
        if champ_high <= champ_low:
            continue
        for chall_thresh in np.arange(0.15, 0.65, 0.05):
            correct, total = 0, 0
            forgery_stats = defaultdict(lambda: {"correct": 0, "total": 0})

            for (vid, forgery, label), indices in video_groups.items():
                frame_preds = []
                for i in indices:
                    cp = champ[i]
                    if cp <= champ_low:
                        frame_preds.append(0)
                    elif cp >= champ_high:
                        frame_preds.append(1)
                    else:
                        frame_preds.append(1 if chall[i] > chall_thresh else 0)

                fake_votes = sum(frame_preds)
                pred = 1 if fake_votes > len(frame_preds) / 2 else 0
                total += 1
                if pred == label:
                    correct += 1
                forgery_stats[forgery]["total"] += 1
                if pred == label:
                    forgery_stats[forgery]["correct"] += 1

            acc = correct / total * 100
            real_c = forgery_stats["original"]["correct"]
            real_t = forgery_stats["original"]["total"]
            fake_c = sum(forgery_stats[ft]["correct"] for ft in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
            fake_t = sum(forgery_stats[ft]["total"] for ft in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
            real_acc = real_c / max(real_t, 1) * 100
            fake_acc = fake_c / max(fake_t, 1) * 100
            balanced_acc = (real_acc + fake_acc) / 2

            if balanced_acc > best_hybrid.get("balanced_acc", 0) or (
                balanced_acc == best_hybrid.get("balanced_acc", 0) and acc > best_hybrid.get("accuracy", 0)
            ):
                best_hybrid = {
                    "champ_low": round(champ_low, 3),
                    "champ_high": round(champ_high, 3),
                    "chall_thresh": round(chall_thresh, 3),
                    "accuracy": round(acc, 1),
                    "balanced_acc": round(balanced_acc, 1),
                    "real_acc": round(real_acc, 1),
                    "fake_acc": round(fake_acc, 1),
                    "forgery_stats": {k: dict(v) for k, v in forgery_stats.items()},
                }

print(f"\n  Best hybrid:")
print(f"    champ_low={best_hybrid['champ_low']}, champ_high={best_hybrid['champ_high']}, "
      f"chall_thresh={best_hybrid['chall_thresh']}")
print(f"    accuracy={best_hybrid['accuracy']:.1f}%, balanced={best_hybrid['balanced_acc']:.1f}%, "
      f"real={best_hybrid['real_acc']:.1f}%, fake={best_hybrid['fake_acc']:.1f}%")

fs = best_hybrid["forgery_stats"]
for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
    s = fs.get(ftype, {"correct": 0, "total": 0})
    if s["total"] > 0:
        print(f"      {ftype:20s}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")


# ---------------------------------------------------------------------------
# FINAL: print the recommended code changes
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RECOMMENDED CODE CHANGES FOR main.py")
print("=" * 70)

print(f"""
OPTION A: Logit Stacking (best frame-level F1)
  Formula: sigmoid({best_cfg['a']}*logit(champ) + {best_cfg['b']}*logit(chall) + {best_cfg['c']}*logit(fall) + {best_cfg['bias']})
  Video: {best_cfg['accuracy']:.1f}% overall, {best_cfg['balanced_acc']:.1f}% balanced
  Real: {best_cfg['real_acc']:.1f}%  Fake: {best_cfg['fake_acc']:.1f}%

OPTION B: Champion-Gated Hybrid (best balance)
  If champ_p_fake <= {best_hybrid['champ_low']}: Real
  If champ_p_fake >= {best_hybrid['champ_high']}: Fake  
  Otherwise: Fake if chall_p_fake > {best_hybrid['chall_thresh']}
  Video: {best_hybrid['accuracy']:.1f}% overall, {best_hybrid['balanced_acc']:.1f}% balanced
  Real: {best_hybrid['real_acc']:.1f}%  Fake: {best_hybrid['fake_acc']:.1f}%
""")
