"""
utils/data_loader.py
--------------------
Handles dataset loading, splitting, augmentation, and feature extraction
for both baseline ML models and deep learning models.
"""

import os
import random
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    DATA_DIR, IMG_SIZE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)


# ── Helpers ────────────────────────────────────────────────────────────────

def get_class_names(data_dir=DATA_DIR):
    """Return sorted list of class folder names."""
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])


def get_all_image_paths(data_dir=DATA_DIR):
    """
    Walk the dataset directory and return (path, label_index) pairs.
    Returns: (image_paths, labels, class_names)
    """
    class_names = get_class_names(data_dir)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    image_paths, labels = [], []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    return image_paths, labels, class_names


def split_dataset(image_paths, labels, train_ratio=TRAIN_RATIO,
                  val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """Stratified split into train / val / test sets."""
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        image_paths, labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_frac),
        stratify=y_tmp,
        random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── PyTorch Dataset ────────────────────────────────────────────────────────

def get_transforms(split="train"):
    """Return torchvision transforms for a given split."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, split="train"):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = get_transforms(split)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                    batch_size=32, num_workers=0):  # 0 = Windows safe
    """Build and return train / val / test DataLoaders."""
    train_ds = PlantDiseaseDataset(X_train, y_train, split="train")
    val_ds   = PlantDiseaseDataset(X_val,   y_val,   split="val")
    test_ds  = PlantDiseaseDataset(X_test,  y_test,  split="test")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ── Handcrafted Feature Extraction (for baseline ML) ──────────────────────

def extract_color_histogram(img_bgr, bins=64):
    """Compute per-channel colour histogram and concatenate."""
    hists = []
    for ch in range(3):
        hist = cv2.calcHist([img_bgr], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)
    return np.concatenate(hists)


def extract_lbp(img_gray, radius=1, n_points=8):
    """Local Binary Pattern texture descriptor."""
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                            range=(0, n_points + 2))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_features(image_path):
    """
    Extract handcrafted features from a single image for baseline ML.
    Returns a 1D numpy feature vector.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_feat   = extract_color_histogram(img, bins=64)   # 64*3 = 192
    lbp_feat     = extract_lbp(gray)                       # 10

    return np.concatenate([color_feat, lbp_feat])


def build_feature_matrix(image_paths, labels, desc="Extracting features"):
    """Extract handcrafted features for a list of images."""
    from tqdm import tqdm
    X, y = [], []
    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc=desc):
        try:
            feat = extract_features(path)
            X.append(feat)
            y.append(label)
        except Exception:
            pass   # skip corrupt images
    return np.array(X), np.array(y)
