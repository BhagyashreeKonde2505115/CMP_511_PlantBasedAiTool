"""
utils/evaluation.py
-------------------
Shared evaluation helpers: metrics, confusion matrix, training curves.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from config import RESULTS_DIR


def compute_metrics(y_true, y_pred, class_names=None):
    """Return dict of accuracy, precision, recall, f1."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_report(y_true, y_pred, class_names, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*60}")
    metrics = compute_metrics(y_true, y_pred, class_names)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="model", save=True):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) // 2),
                                    max(8, len(class_names) // 2)))
    sns.heatmap(
        cm, annot=(len(class_names) <= 20),
        fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"confusion_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] {path}")
    plt.close()


def plot_training_curves(history, model_name="model", save=True):
    """
    Plot train/val loss and accuracy from a history dict.
    history = {"train_loss": [...], "val_loss": [...],
                "train_acc": [...], "val_acc": [...]}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2E75B6")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#E05C3A", linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#2E75B6")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="#E05C3A", linestyle="--")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — {model_name}", fontsize=14)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"curves_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] {path}")
    plt.close()


def plot_model_comparison(results_dict, save=True):
    """
    Bar chart comparing all models across metrics.
    results_dict = {"Model Name": {"accuracy": 0.9, "f1": 0.89, ...}, ...}
    """
    models  = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = ["#2E75B6", "#1F3864", "#E05C3A", "#F5A623"]

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results_dict[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "model_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] {path}")
    plt.close()
