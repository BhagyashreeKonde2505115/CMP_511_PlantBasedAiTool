"""
train_all.py
------------
Master training script. Runs all models sequentially and produces
a final comparison chart.

Usage:
    python train_all.py                  # Run all models
    python train_all.py --skip-baseline  # Skip SVM & RF (slow on large dataset)
    python train_all.py --model cnn      # Run only CNN
    python train_all.py --model resnet50
    python train_all.py --model vgg16
    python train_all.py --model baseline
"""

import argparse
import json
import os
import sys

from config import DATA_DIR, RESULTS_DIR, MODEL_DIR


def check_data_dir():
    if not os.path.isdir(DATA_DIR):
        print(f"\n[ERROR] Dataset not found at: {DATA_DIR}")
        print("Please:")
        print("  1. Run: python download_dataset.py")
        print("  2. OR update DATA_DIR in config.py")
        sys.exit(1)
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(classes) < 2:
        print(f"\n[ERROR] Expected class subfolders in {DATA_DIR}, found: {classes}")
        sys.exit(1)
    print(f"[OK] Dataset found — {len(classes)} classes in {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Detection — Train All Models")
    parser.add_argument("--model", choices=["all", "baseline", "cnn", "resnet50", "vgg16"],
                        default="all", help="Which model(s) to train")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip SVM & RF (recommended for large datasets without much RAM)")
    args = parser.parse_args()

    check_data_dir()

    all_results = {}

    # ── Baseline ───────────────────────────────────────────────────────────
    if args.model in ("all", "baseline") and not args.skip_baseline:
        from pipeline.baseline_models import run_baseline
        results, _ = run_baseline()
        all_results.update(results)

    # ── Custom CNN ─────────────────────────────────────────────────────────
    if args.model in ("all", "cnn"):
        from pipeline.cnn_model import run_cnn
        metrics, _ = run_cnn()
        all_results["Custom CNN"] = metrics

    # ── ResNet50 ───────────────────────────────────────────────────────────
    if args.model in ("all", "resnet50"):
        from pipeline.transfer_learning import run_transfer_learning
        metrics, _ = run_transfer_learning("resnet50")
        all_results["ResNet50"] = metrics

    # ── VGG16 ─────────────────────────────────────────────────────────────
    if args.model in ("all", "vgg16"):
        from pipeline.transfer_learning import run_transfer_learning
        metrics, _ = run_transfer_learning("vgg16")
        all_results["VGG16"] = metrics

    # ── Summary ────────────────────────────────────────────────────────────
    if all_results:
        print("\n" + "="*60)
        print("  FINAL COMPARISON")
        print("="*60)
        print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for name, m in all_results.items():
            print(f"  {name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>10.4f}")

        # Save results JSON
        results_path = os.path.join(RESULTS_DIR, "all_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  [Saved] {results_path}")

        # Comparison chart
        from utils.evaluation import plot_model_comparison
        plot_model_comparison(all_results)

        print("\n[ALL DONE] All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
