"""
pipeline/transfer_learning.py
Fine-tunes ResNet50 and VGG16 (pre trained on ImageNet)
for plant disease classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from config import (
    MODEL_DIR, TL_EPOCHS, TL_BATCH_SIZE,
    TL_LEARNING_RATE, TL_FINE_TUNE_LAYERS
)
from utils.data_loader import get_all_image_paths, split_dataset, get_dataloaders
from utils.evaluation import print_report, plot_confusion_matrix, plot_training_curves
from pipeline.cnn_model import train_epoch, eval_epoch


# ── Model Builders ─────────────────────────────────────────────────────────

def build_resnet50(num_classes, fine_tune_layers=TL_FINE_TUNE_LAYERS):
    """
    ResNet50 with ImageNet weights.
    Freezes all layers, then unfreezes the last `fine_tune_layers` layers.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N layers
    all_layers = list(model.children())
    for layer in all_layers[-fine_tune_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model


def build_vgg16(num_classes, fine_tune_layers=TL_FINE_TUNE_LAYERS):
    """
    VGG16 with ImageNet weights.
    Freezes convolutional layers, unfreezes last `fine_tune_layers` conv layers.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N conv layers
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    for layer in conv_layers[-fine_tune_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model


MODEL_REGISTRY = {
    "resnet50": build_resnet50,
    "vgg16":    build_vgg16,
}


# ── Training ───────────────────────────────────────────────────────────────

def train_transfer_model(backbone_name, train_loader, val_loader, num_classes,
                          epochs=TL_EPOCHS, lr=TL_LEARNING_RATE, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Backbone : {backbone_name.upper()}")
    print(f"  Device   : {device}")

    builder = MODEL_REGISTRY[backbone_name]
    model = builder(num_classes).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:03d}/{epochs}  "
              f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f}  |  "
              f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f}"
              + (" ✓" if val_acc == best_val_acc else ""))

    model.load_state_dict(best_state)
    return model, history


# ── Full Pipeline ──────────────────────────────────────────────────────────

def run_transfer_learning(backbone_name="resnet50", data_dir=None):
    from config import DATA_DIR
    data_dir = data_dir or DATA_DIR

    assert backbone_name in MODEL_REGISTRY, \
        f"Unknown backbone '{backbone_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"

    label = backbone_name.upper()
    print("\n" + "="*60)
    print(f"  TRANSFER LEARNING — {label}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1/4] Loading dataset...")
    image_paths, labels, class_names = get_all_image_paths(data_dir)
    print(f"  Total images: {len(image_paths)}  |  Classes: {len(class_names)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(image_paths, labels)

    print("\n[2/4] Building DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=TL_BATCH_SIZE
    )

    print("\n[3/4] Fine-tuning model...")
    model, history = train_transfer_model(
        backbone_name, train_loader, val_loader, len(class_names), device=device
    )

    print("\n[4/4] Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    _, _, y_pred, y_true = eval_epoch(model, test_loader, criterion, device)
    metrics = print_report(y_true, y_pred, class_names, label)
    plot_confusion_matrix(y_true, y_pred, class_names, label)
    plot_training_curves(history, label)

    # Save
    save_path = os.path.join(MODEL_DIR, f"{backbone_name}_model.pth")
    torch.save({
        "backbone":    backbone_name,
        "model_state": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "history":     history,
    }, save_path)
    print(f"  [Saved] {save_path}")

    print(f"\n[DONE] {label} training complete.")
    return metrics, class_names


if __name__ == "__main__":
    import sys
    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
    run_transfer_learning(backbone)
