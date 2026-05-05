"""
pipeline/cnn_model.py
Defines and trains a custom Convolutional Neural Network (CNN)
for plant disease classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    MODEL_DIR, CNN_EPOCHS, CNN_BATCH_SIZE,
    CNN_LEARNING_RATE, CNN_DROPOUT, IMG_SIZE
)
from utils.data_loader import get_all_image_paths, split_dataset, get_dataloaders
from utils.evaluation import print_report, plot_confusion_matrix, plot_training_curves


# ── Model Architecture ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool block."""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PlantDiseaseCNN(nn.Module):
    """
    Custom CNN for plant disease classification.
    Architecture:
        4 ConvBlocks (3→64→128→256→512) with MaxPool
        Global Average Pooling
        FC layers with dropout
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64),   # 224→112
            ConvBlock(64,  128),  # 112→56
            ConvBlock(128, 256),  # 56→28
            ConvBlock(256, 512),  # 28→14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 14→1
        self.classifier = nn.Sequential(
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(CNN_DROPOUT * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Training Loop ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_cnn(train_loader, val_loader, num_classes, epochs=CNN_EPOCHS,
              lr=CNN_LEARNING_RATE, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = PlantDiseaseCNN(num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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

def run_cnn(data_dir=None):
    from config import DATA_DIR
    data_dir = data_dir or DATA_DIR

    print("\n" + "="*60)
    print("  CUSTOM CNN")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load & split
    print("\n[1/4] Loading dataset...")
    image_paths, labels, class_names = get_all_image_paths(data_dir)
    print(f"  Total images: {len(image_paths)}  |  Classes: {len(class_names)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(image_paths, labels)

    # DataLoaders
    print("\n[2/4] Building DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=CNN_BATCH_SIZE
    )

    # Train
    print("\n[3/4] Training CNN...")
    model, history = train_cnn(train_loader, val_loader, len(class_names), device=device)

    # Evaluate
    print("\n[4/4] Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    _, _, y_pred, y_true = eval_epoch(model, test_loader, criterion, device)
    metrics = print_report(y_true, y_pred, class_names, "Custom CNN")
    plot_confusion_matrix(y_true, y_pred, class_names, "Custom_CNN")
    plot_training_curves(history, "Custom_CNN")

    # Save
    cnn_path = os.path.join(MODEL_DIR, "cnn_model.pth")
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "history": history,
    }, cnn_path)
    print(f"  [Saved] {cnn_path}")

    print("\n[DONE] CNN training complete.")
    return metrics, class_names


if __name__ == "__main__":
    run_cnn()
