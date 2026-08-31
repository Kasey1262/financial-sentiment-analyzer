"""
evaluate.py — small hand-labeled evaluation for the Financial Sentiment Analyzer.

Runs every example in labeled_samples.json through analyze_sentiment(),
then reports:
  - overall accuracy
  - a 3x3 confusion matrix (true vs predicted)
  - per-class precision / recall / F1
  - a confidence-vs-correctness check (is the model more confident when it's right?)

HONESTY NOTE:
  The 'label' field in labeled_samples.json must be YOUR OWN judgment.
  Review/override every label before trusting these numbers — otherwise it
  isn't really a hand-labeled evaluation. Several items are deliberately
  ambiguous; those are yours to decide.

Run from the repo root (same folder as analyzer.py), with your .env in place:
    python3 evaluate.py
"""
import json
import time
from collections import defaultdict

from analyzer import analyze_sentiment

CLASSES = ["bullish", "bearish", "neutral"]
DATA_FILE = "labeled_samples.json"
SLEEP_BETWEEN_CALLS = 0.3   # be gentle on the API; raise if you hit rate limits


def load_samples():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data
    missing = [s["id"] for s in samples if not str(s.get("label", "")).strip()]
    if missing:
        print(f"WARNING: these items have a blank 'label': {missing}")
        print("Fill them in with your own judgment before trusting the results.\n")
    return samples


def run_predictions(samples):
    """Call the model on each sample; return a list of result dicts."""
    results = []
    print(f"Running {len(samples)} examples through analyze_sentiment()...\n")
    for i, ex in enumerate(samples, 1):
        true = str(ex.get("label", "")).strip().lower()
        out = analyze_sentiment(ex["text"])
        pred = str(out.get("sentiment", "error")).strip().lower()
        conf = out.get("confidence", 0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        correct = (pred == true) and (true in CLASSES)
        results.append({"id": ex["id"], "true": true, "pred": pred,
                        "conf": conf, "correct": correct})
        mark = "OK" if correct else ("--" if pred == "error" else "XX")
        print(f"[{i:2d}] {mark}  true={true:8s} pred={pred:8s} conf={conf:5.1f}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results


def confusion_matrix(results):
    """counts[true][pred] -> int. Predicted labels can include 'error'."""
    counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["true"]][r["pred"]] += 1
    return counts


def print_confusion(counts):
    pred_labels = CLASSES + ["error"]
    print("\nConfusion matrix (rows = your label, cols = model prediction)")
    header = "true \\ pred | " + " ".join(f"{c[:7]:>8s}" for c in pred_labels)
    print(header)
    print("-" * len(header))
    for t in CLASSES:
        row = counts.get(t, {})
        cells = " ".join(f"{row.get(p, 0):>8d}" for p in pred_labels)
        print(f"{t:11s}| {cells}")


def per_class_metrics(results):
    """Precision / recall / F1 per class (treating each class one-vs-rest)."""
    print("\nPer-class metrics")
    print(f"{'class':10s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    print("-" * 46)
    for c in CLASSES:
        tp = sum(1 for r in results if r["true"] == c and r["pred"] == c)
        fp = sum(1 for r in results if r["true"] != c and r["pred"] == c)
        fn = sum(1 for r in results if r["true"] == c and r["pred"] != c)
        support = sum(1 for r in results if r["true"] == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        print(f"{c:10s} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f} {support:>8d}")


def confidence_analysis(results):
    """Is the model more confident when it's actually right? (calibration signal)"""
    conf_right = [r["conf"] for r in results if r["correct"]]
    conf_wrong = [r["conf"] for r in results if not r["correct"] and r["pred"] != "error"]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("\nConfidence vs correctness")
    print(f"  avg confidence when CORRECT : {avg(conf_right):5.1f}  (n={len(conf_right)})")
    print(f"  avg confidence when WRONG   : {avg(conf_wrong):5.1f}  (n={len(conf_wrong)})")
    print("  (If 'correct' isn't clearly higher, the confidence score isn't well calibrated.)")


def main():
    samples = load_samples()
    results = run_predictions(samples)

    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_error = sum(1 for r in results if r["pred"] == "error")

    print("\n" + "=" * 46)
    print(f"Overall accuracy: {n_correct}/{n} = {n_correct / n:.1%}")
    if n_error:
        print(f"Parse errors (model returned no valid JSON): {n_error}")
    print("=" * 46)

    print_confusion(confusion_matrix(results))
    per_class_metrics(results)
    confidence_analysis(results)

    # Show the misses so you can eyeball WHY it was wrong (great interview material)
    misses = [r for r in results if not r["correct"]]
    if misses:
        print("\nMisses to review (id: your label -> model):")
        for r in misses:
            print(f"  #{r['id']}: {r['true']} -> {r['pred']} (conf {r['conf']:.0f})")


if __name__ == "__main__":
    main()
