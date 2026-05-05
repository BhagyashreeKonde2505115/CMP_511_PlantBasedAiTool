"""
predict.py
----------
Command-line tool for predicting plant disease on a single image.

Usage:
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --model ResNet50
    python predict.py path/to/leaf.jpg --model "Custom CNN"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument("image", help="Path to leaf image")
    parser.add_argument("--model", default=None,
                        help="Model name (default: best available)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print("\n🌿 Plant Disease Detection")
    print("   Loading models...")
    predictor = Predictor()

    available = predictor.get_available_models()
    if not available:
        print("[ERROR] No trained models found. Run: python train_all.py")
        sys.exit(1)

    print(f"   Available models: {', '.join(available)}")
    print(f"   Analysing: {args.image}\n")

    result = predictor.predict(args.image, model_name=args.model)

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(1)

    print("=" * 50)
    print(f"  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']*100:.2f}%")
    print(f"  Model Used      : {result['model_used']}")
    print("=" * 50)
    print("\n  Top 5 Predictions:")
    for i, (cls, prob) in enumerate(result["top5"], 1):
        bar = "█" * int(prob * 30)
        print(f"  {i}. {cls:<40} {prob*100:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
