"""
Deep analysis of evaluation results: probability distributions,
per-model threshold calibration, and advanced ensemble strategies.

Can be run with in-memory results (e.g. from evaluate_ensemble) or by loading
eval_raw_results.json when run as a script.
"""

import json
import math
import numpy as np
from collections import defaultdict
from itertools import product


def load_results(path: str = "eval_raw_results.json") -> list:
    """Load raw results from JSON (used when running this script standalone)."""
    with open(path) as f:
        return json.load(f)


def analyze(results: list) -> None:
    """Run full analysis on the given results list (from evaluate_ensemble or file)."""
    labels = np.array([r["label"] for r in results])
    forgeries = [r["forgery"] for r in results]
    champ = np.array([r["champion"] for r in results])
    chall = np.array([r["challenger"] for r in results])
    fall = np.array([r["fallback"] for r in results])
    
    real_mask = labels == 0
    fake_mask = labels == 1
    
    # ---------------------------------------------------------------------------
    # 1. Probability distribution analysis
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("PROBABILITY DISTRIBUTIONS")
    print("=" * 70)
    
    for name, probs in [("Champion", champ), ("Challenger", chall), ("Fallback", fall)]:
        print(f"\n  {name}:")
        print(f"    Reals:  mean={probs[real_mask].mean():.4f}  "
              f"std={probs[real_mask].std():.4f}  "
              f"min={probs[real_mask].min():.4f}  "
              f"max={probs[real_mask].max():.4f}  "
              f"median={np.median(probs[real_mask]):.4f}")
        print(f"    Fakes:  mean={probs[fake_mask].mean():.4f}  "
              f"std={probs[fake_mask].std():.4f}  "
              f"min={probs[fake_mask].min():.4f}  "
              f"max={probs[fake_mask].max():.4f}  "
              f"median={np.median(probs[fake_mask]):.4f}")
    
        # Per-forgery breakdown
        for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            mask = np.array([f == ftype for f in forgeries])
            if mask.sum() == 0:
                continue
            p = probs[mask]
            print(f"      {ftype:20s}: mean={p.mean():.4f} std={p.std():.4f} "
                  f"[{p.min():.4f} - {p.max():.4f}]")
    
    
    # ---------------------------------------------------------------------------
    # 2. Per-model optimal threshold (maximize F1)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-MODEL OPTIMAL THRESHOLDS (frame-level F1)")
    print("=" * 70)
    
    def eval_at_threshold(probs, labels, thresh):
        preds = (probs > thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        acc = (tp + tn) / len(labels) * 100
        prec = tp / max(tp + fp, 1) * 100
        rec = tp / max(tp + fn, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1)
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    
    optimal_thresholds = {}
    for name, probs in [("champion", champ), ("challenger", chall), ("fallback", fall)]:
        best_f1 = 0
        best_thresh = 0.5
        best_result = None
        for t in np.arange(0.01, 0.99, 0.01):
            r = eval_at_threshold(probs, labels, t)
            if r["f1"] > best_f1:
                best_f1 = r["f1"]
                best_thresh = t
                best_result = r
    
        optimal_thresholds[name] = best_thresh
        print(f"\n  {name.upper()}: optimal threshold = {best_thresh:.2f}")
        print(f"    acc={best_result['acc']:.1f}% prec={best_result['prec']:.1f}% "
              f"rec={best_result['rec']:.1f}% f1={best_result['f1']:.1f}%")
        print(f"    TP={best_result['tp']} TN={best_result['tn']} "
              f"FP={best_result['fp']} FN={best_result['fn']}")
    
    
    # ---------------------------------------------------------------------------
    # 3. Calibrated logit blending: shift logits per-model so that their
    #    optimal threshold maps to logit=0, then blend in logit space
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CALIBRATED LOGIT BLENDING")
    print("=" * 70)
    
    def logit(p):
        p = max(1e-7, min(1 - 1e-7, p))
        return math.log(p / (1 - p))
    
    def sigmoid(x):
        if x > 500: return 1.0
        if x < -500: return 0.0
        return 1 / (1 + math.exp(-x))
    
    # The idea: for each model, subtract logit(optimal_threshold) from the logit
    # so the decision boundary is at 0. Then blend shifted logits.
    cal_shifts = {}
    for name, thresh in optimal_thresholds.items():
        cal_shifts[name] = logit(thresh)
        print(f"  {name}: threshold={thresh:.2f} -> logit_shift={cal_shifts[name]:.4f}")
    
    # Calibrated probabilities
    champ_cal = np.array([sigmoid(logit(p) - cal_shifts["champion"]) for p in champ])
    chall_cal = np.array([sigmoid(logit(p) - cal_shifts["challenger"]) for p in chall])
    fall_cal = np.array([sigmoid(logit(p) - cal_shifts["fallback"]) for p in fall])
    
    # Now grid search on calibrated probs
    print("\n  Grid searching calibrated ensemble weights...")
    
    STEP = 0.05
    best_overall = {"f1": 0}
    all_configs = []
    
    def blend_logits_cal(p_list, w_list):
        val = sum(w * logit(p) for p, w in zip(p_list, w_list))
        return sigmoid(val)
    
    def blend_linear_cal(p_list, w_list):
        return sum(w * p for p, w in zip(p_list, w_list))
    
    STRATEGIES = {
        "cal_logit": blend_logits_cal,
        "cal_linear": blend_linear_cal,
    }
    
    for strat_name, blend_fn in STRATEGIES.items():
        best_for_strat = {"f1": 0}
    
        for w1 in np.arange(0.0, 1.0 + STEP / 2, STEP):
            for w2 in np.arange(0.0, 1.0 - w1 + STEP / 2, STEP):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < -0.001:
                    continue
                w3 = max(0.0, w3)
                weights = [w1, w2, w3]
    
                preds = []
                for i in range(len(results)):
                    p_list = [champ_cal[i], chall_cal[i], fall_cal[i]]
                    try:
                        p_fake = blend_fn(p_list, weights)
                    except:
                        p_fake = 0.5
                    preds.append(1 if p_fake > 0.5 else 0)
    
                preds = np.array(preds)
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))
                tn = np.sum((preds == 0) & (labels == 0))
                acc = (tp + tn) / len(labels) * 100
                prec = tp / max(tp + fp, 1) * 100
                rec = tp / max(tp + fn, 1) * 100
                f1 = 2 * prec * rec / max(prec + rec, 1)
    
                config = {
                    "strategy": strat_name,
                    "w_champion": round(w1, 2),
                    "w_challenger": round(w2, 2),
                    "w_fallback": round(w3, 2),
                    "accuracy": round(acc, 2),
                    "precision": round(prec, 2),
                    "recall": round(rec, 2),
                    "f1": round(f1, 2),
                    "fp": int(fp), "fn": int(fn),
                }
                all_configs.append(config)
    
                if f1 > best_for_strat.get("f1", 0):
                    best_for_strat = config.copy()
                if f1 > best_overall.get("f1", 0) or (
                    f1 == best_overall.get("f1", 0) and acc > best_overall.get("accuracy", 0)
                ):
                    best_overall = config.copy()
    
        print(
            f"\n  {strat_name}: best w=[{best_for_strat['w_champion']:.2f}, "
            f"{best_for_strat['w_challenger']:.2f}, {best_for_strat['w_fallback']:.2f}] "
            f"acc={best_for_strat['accuracy']:.1f}% f1={best_for_strat['f1']:.1f}%"
        )
    
    
    # ---------------------------------------------------------------------------
    # 4. Weighted voting with per-model calibrated thresholds
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WEIGHTED VOTING WITH CALIBRATED THRESHOLDS")
    print("=" * 70)
    
    for w1 in np.arange(0.0, 1.0 + STEP / 2, STEP):
        for w2 in np.arange(0.0, 1.0 - w1 + STEP / 2, STEP):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -0.001:
                continue
            w3 = max(0.0, w3)
            weights = [w1, w2, w3]
    
            # Each model votes fake/real using its calibrated threshold
            votes = np.zeros(len(results))
            for i in range(len(results)):
                v = 0.0
                if champ[i] > optimal_thresholds["champion"]:
                    v += w1
                if chall[i] > optimal_thresholds["challenger"]:
                    v += w2
                if fall[i] > optimal_thresholds["fallback"]:
                    v += w3
                votes[i] = v
    
            preds = (votes > 0.5).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))
            acc = (tp + tn) / len(labels) * 100
            prec = tp / max(tp + fp, 1) * 100
            rec = tp / max(tp + fn, 1) * 100
            f1 = 2 * prec * rec / max(prec + rec, 1)
    
            config = {
                "strategy": "weighted_vote",
                "w_champion": round(w1, 2),
                "w_challenger": round(w2, 2),
                "w_fallback": round(w3, 2),
                "accuracy": round(acc, 2),
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1": round(f1, 2),
                "fp": int(fp), "fn": int(fn),
            }
            all_configs.append(config)
            if f1 > best_overall.get("f1", 0) or (
                f1 == best_overall.get("f1", 0) and acc > best_overall.get("accuracy", 0)
            ):
                best_overall = config.copy()
    
    
    # ---------------------------------------------------------------------------
    # 5. "Any model says fake" / "Majority says fake" / "All say fake"
    # ---------------------------------------------------------------------------
    print("\nVOTING RULES WITH CALIBRATED THRESHOLDS:")
    thresholds = optimal_thresholds
    
    for rule_name, min_votes in [("ANY (1/3)", 1), ("MAJORITY (2/3)", 2), ("ALL (3/3)", 3)]:
        preds = []
        for i in range(len(results)):
            votes = 0
            if champ[i] > thresholds["champion"]: votes += 1
            if chall[i] > thresholds["challenger"]: votes += 1
            if fall[i] > thresholds["fallback"]: votes += 1
            preds.append(1 if votes >= min_votes else 0)
        preds = np.array(preds)
        r = eval_at_threshold(preds, labels, 0.5)
        print(f"  {rule_name}: acc={r['acc']:.1f}% prec={r['prec']:.1f}% "
              f"rec={r['rec']:.1f}% f1={r['f1']:.1f}% "
              f"FP={r['fp']} FN={r['fn']}")
    
        config = {
            "strategy": f"vote_{min_votes}of3",
            "w_champion": 0.33, "w_challenger": 0.33, "w_fallback": 0.33,
            "accuracy": round(r["acc"], 2),
            "precision": round(r["prec"], 2),
            "recall": round(r["rec"], 2),
            "f1": round(r["f1"], 2),
            "fp": int(r["fp"]), "fn": int(r["fn"]),
        }
        all_configs.append(config)
        if r["f1"] > best_overall.get("f1", 0):
            best_overall = config.copy()
    
    
    # ---------------------------------------------------------------------------
    # 6. Champion-gated approach: Champion decides unless uncertain,
    #    then defer to Challenger
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHAMPION-GATED APPROACH")
    print("=" * 70)
    print("  Champion classifies confidently outside [low, high]; "
          "uncertain goes to challenger+fallback")
    
    best_gated = {"f1": 0}
    for low in np.arange(0.02, 0.30, 0.02):
        for high in np.arange(0.10, 0.70, 0.02):
            if high <= low:
                continue
            preds = []
            for i in range(len(results)):
                if champ[i] <= low:
                    preds.append(0)  # Champion confident Real
                elif champ[i] >= high:
                    preds.append(1)  # Champion confident Fake
                else:
                    # Uncertain: defer to challenger
                    p = chall[i]
                    preds.append(1 if p > optimal_thresholds["challenger"] else 0)
            preds = np.array(preds)
            r = eval_at_threshold(preds, labels, 0.5)
            if r["f1"] > best_gated.get("f1", 0):
                best_gated = {
                    "strategy": "champion_gated",
                    "low": round(low, 2),
                    "high": round(high, 2),
                    **{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()},
                }
    
    print(f"\n  Best gated config: low={best_gated['low']:.2f}, high={best_gated['high']:.2f}")
    print(f"    acc={best_gated['acc']:.1f}% prec={best_gated['prec']:.1f}% "
          f"rec={best_gated['rec']:.1f}% f1={best_gated['f1']:.1f}%")
    
    all_configs.append({
        "strategy": "champion_gated",
        "w_champion": best_gated["low"],
        "w_challenger": best_gated["high"],
        "w_fallback": 0.0,
        "accuracy": best_gated["acc"],
        "precision": best_gated["prec"],
        "recall": best_gated["rec"],
        "f1": best_gated["f1"],
        "fp": best_gated["fp"], "fn": best_gated["fn"],
    })
    if best_gated["f1"] > best_overall.get("f1", 0):
        best_overall = all_configs[-1].copy()
        best_overall["gated_low"] = best_gated["low"]
        best_overall["gated_high"] = best_gated["high"]
    
    
    # ---------------------------------------------------------------------------
    # 7. Stacking: use model outputs as features, simple logistic-like rule
    #    p_fake_final = sigmoid(a*logit(champ) + b*logit(chall) + c*logit(fall) + bias)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEARNED LOGIT STACKING (grid search a, b, c, bias)")
    print("=" * 70)
    
    champ_logits = np.array([logit(p) for p in champ])
    chall_logits = np.array([logit(p) for p in chall])
    fall_logits = np.array([logit(p) for p in fall])
    
    best_stack = {"f1": 0}
    for a in np.arange(-0.5, 2.0, 0.25):
        for b in np.arange(-0.5, 2.0, 0.25):
            for c in np.arange(-0.5, 2.0, 0.25):
                for bias in np.arange(-3.0, 3.0, 0.5):
                    z = a * champ_logits + b * chall_logits + c * fall_logits + bias
                    preds = (z > 0).astype(int)
                    tp = np.sum((preds == 1) & (labels == 1))
                    fp = np.sum((preds == 1) & (labels == 0))
                    fn = np.sum((preds == 0) & (labels == 1))
                    tn = np.sum((preds == 0) & (labels == 0))
                    acc = (tp + tn) / len(labels) * 100
                    prec = tp / max(tp + fp, 1) * 100
                    rec = tp / max(tp + fn, 1) * 100
                    f1 = 2 * prec * rec / max(prec + rec, 1)
                    if f1 > best_stack.get("f1", 0) or (
                        f1 == best_stack.get("f1", 0) and acc > best_stack.get("acc", 0)
                    ):
                        best_stack = {
                            "a": round(a, 2), "b": round(b, 2),
                            "c": round(c, 2), "bias": round(bias, 2),
                            "acc": round(acc, 2), "prec": round(prec, 2),
                            "rec": round(rec, 2), "f1": round(f1, 2),
                            "fp": int(fp), "fn": int(fn),
                        }
    
    print(f"  Best stacking: a={best_stack['a']}, b={best_stack['b']}, "
          f"c={best_stack['c']}, bias={best_stack['bias']}")
    print(f"    acc={best_stack['acc']:.1f}% prec={best_stack['prec']:.1f}% "
          f"rec={best_stack['rec']:.1f}% f1={best_stack['f1']:.1f}%")
    print(f"    FP={best_stack['fp']} FN={best_stack['fn']}")
    
    all_configs.append({
        "strategy": "logit_stacking",
        "w_champion": best_stack["a"],
        "w_challenger": best_stack["b"],
        "w_fallback": best_stack["c"],
        "bias": best_stack["bias"],
        "accuracy": best_stack["acc"],
        "precision": best_stack["prec"],
        "recall": best_stack["rec"],
        "f1": best_stack["f1"],
        "fp": best_stack["fp"], "fn": best_stack["fn"],
    })
    if best_stack["f1"] > best_overall.get("f1", 0):
        best_overall = all_configs[-1].copy()
    
    
    # ---------------------------------------------------------------------------
    # 8. Video-level evaluation of top approaches
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VIDEO-LEVEL COMPARISON OF ALL BEST STRATEGIES")
    print("=" * 70)
    
    video_groups = defaultdict(list)
    for i, r in enumerate(results):
        key = (r["video"], r["forgery"], r["label"])
        video_groups[key].append(i)
    
    
    def video_eval(frame_preds):
        correct, total = 0, 0
        forgery_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for (vid, forgery, label), indices in video_groups.items():
            fake_votes = sum(frame_preds[i] for i in indices)
            pred = 1 if fake_votes > len(indices) / 2 else 0
            total += 1
            if pred == label:
                correct += 1
            forgery_stats[forgery]["total"] += 1
            if pred == label:
                forgery_stats[forgery]["correct"] += 1
        return correct, total, forgery_stats
    
    
    # Evaluate each approach at video level
    print("\n  1. Champion only (calibrated threshold):")
    preds_champ = (champ > optimal_thresholds["champion"]).astype(int)
    c, t, fs = video_eval(preds_champ)
    print(f"     {c}/{t} = {c/t*100:.1f}%")
    for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        s = fs[ftype]
        if s["total"] > 0:
            print(f"       {ftype:20s}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")
    
    print("\n  2. Challenger only (calibrated threshold):")
    preds_chall = (chall > optimal_thresholds["challenger"]).astype(int)
    c, t, fs = video_eval(preds_chall)
    print(f"     {c}/{t} = {c/t*100:.1f}%")
    for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        s = fs[ftype]
        if s["total"] > 0:
            print(f"       {ftype:20s}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")
    
    print(f"\n  3. Logit stacking (a={best_stack['a']}, b={best_stack['b']}, "
          f"c={best_stack['c']}, bias={best_stack['bias']}):")
    z = best_stack["a"] * champ_logits + best_stack["b"] * chall_logits + best_stack["c"] * fall_logits + best_stack["bias"]
    preds_stack = (z > 0).astype(int)
    c, t, fs = video_eval(preds_stack)
    print(f"     {c}/{t} = {c/t*100:.1f}%")
    for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        s = fs[ftype]
        if s["total"] > 0:
            print(f"       {ftype:20s}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")
    
    print(f"\n  4. Champion-gated (low={best_gated['low']}, high={best_gated['high']}):")
    preds_gated = []
    for i in range(len(results)):
        if champ[i] <= best_gated["low"]:
            preds_gated.append(0)
        elif champ[i] >= best_gated["high"]:
            preds_gated.append(1)
        else:
            preds_gated.append(1 if chall[i] > optimal_thresholds["challenger"] else 0)
    preds_gated = np.array(preds_gated)
    c, t, fs = video_eval(preds_gated)
    print(f"     {c}/{t} = {c/t*100:.1f}%")
    for ftype in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        s = fs[ftype]
        if s["total"] > 0:
            print(f"       {ftype:20s}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")
    
    
    # ---------------------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERALL BEST CONFIGURATION")
    print("=" * 70)
    
    all_configs.sort(key=lambda c: (c["f1"], c["accuracy"]), reverse=True)
    top = all_configs[0]
    print(f"\n  Strategy: {top['strategy']}")
    for k, v in top.items():
        if k != "strategy":
            print(f"    {k}: {v}")
    
    print(f"\n  Optimal per-model thresholds:")
    for name, thresh in optimal_thresholds.items():
        print(f"    {name}: {thresh:.2f}")
    
    print(f"\n  Logit stacking coefficients:")
    print(f"    champion_coeff = {best_stack['a']}")
    print(f"    challenger_coeff = {best_stack['b']}")
    print(f"    fallback_coeff = {best_stack['c']}")
    print(f"    bias = {best_stack['bias']}")
    
    # Save
    summary = {
        "optimal_thresholds": optimal_thresholds,
        "logit_stacking": best_stack,
        "champion_gated": best_gated,
        "best_overall": top,
        "top_20": all_configs[:20],
    }
    with open("eval_detailed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed summary saved to eval_detailed_summary.json")


if __name__ == "__main__":
    results = load_results()
    analyze(results)
