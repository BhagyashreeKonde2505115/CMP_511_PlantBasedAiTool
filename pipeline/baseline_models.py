"""
pipeline/baseline_models.py
Trains and evaluates SVM and Random Forest classifiers
using handcrafted image features (colour histogram + LBP texture).
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, RESULTS_DIR
from utils.data_loader import (
    get_all_image_paths, split_dataset, build_feature_matrix
)
from utils.evaluation import print_report, plot_confusion_matrix, compute_metrics


def train_svm(X_train, y_train):
    print("\n[SVM] Training Support Vector Machine...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train):
    print("\n[RF] Training Random Forest...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def run_baseline(data_dir=None):
    """
    Full baseline pipeline:
      1. Load images and extract handcrafted features
      2. Train SVM and Random Forest
      3. Evaluate on test set and save models
    Returns dict of metrics for both models.
    """
    from config import DATA_DIR
    data_dir = data_dir or DATA_DIR

    print("\n" + "="*60)
    print("  BASELINE ML — SVM & Random Forest")
    print("="*60)

    # 1. Load paths
    print("\n[1/4] Loading dataset...")
    image_paths, labels, class_names = get_all_image_paths(data_dir)
    print(f"  Total images : {len(image_paths)}")
    print(f"  Classes      : {len(class_names)}")

    # 2. Split
    print("\n[2/4] Splitting dataset...")
    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test = split_dataset(image_paths, labels)
    # Combine train + val for ML (we don't tune hyperparams per epoch)
    X_fit_p = X_train_p + X_val_p
    y_fit   = list(y_train) + list(y_val)
    print(f"  Train+Val: {len(X_fit_p)}  |  Test: {len(X_test_p)}")

    # 3. Feature extraction
    print("\n[3/4] Extracting handcrafted features...")
    X_fit,  y_fit  = build_feature_matrix(X_fit_p,  y_fit,  desc="Train+Val")
    X_test, y_test = build_feature_matrix(X_test_p, list(y_test), desc="Test")

    results = {}

    # 4a. SVM
    svm_model = train_svm(X_fit, y_fit)
    print("[SVM] Evaluating...")
    y_pred_svm = svm_model.predict(X_test)
    metrics_svm = print_report(y_test, y_pred_svm, class_names, "SVM")
    plot_confusion_matrix(y_test, y_pred_svm, class_names, "SVM")
    svm_path = os.path.join(MODEL_DIR, "svm_model.joblib")
    joblib.dump({"model": svm_model, "class_names": class_names}, svm_path)
    print(f"  [Saved] {svm_path}")
    results["SVM"] = metrics_svm

    # 4b. Random Forest
    rf_model = train_random_forest(X_fit, y_fit)
    print("[RF] Evaluating...")
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = print_report(y_test, y_pred_rf, class_names, "Random Forest")
    plot_confusion_matrix(y_test, y_pred_rf, class_names, "Random_Forest")
    rf_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    joblib.dump({"model": rf_model, "class_names": class_names}, rf_path)
    print(f"  [Saved] {rf_path}")
    results["Random Forest"] = metrics_rf

    print("\n[DONE] Baseline training complete.")
    return results, class_names


if __name__ == "__main__":
    run_baseline()
