"""
download_dataset.py
-------------------
Downloads the PlantVillage dataset from Kaggle.

Usage:
    python download_dataset.py

Requirements:
    - Kaggle API credentials configured at ~/.kaggle/kaggle.json
    - OR set environment variables: KAGGLE_USERNAME and KAGGLE_KEY

To get Kaggle credentials:
    1. Go to https://www.kaggle.com/account
    2. Click "Create New API Token"
    3. Place the downloaded kaggle.json in ~/.kaggle/
"""

import os
import sys
import zipfile
import shutil

def download_plantvillage():
    print("=" * 60)
    print("PlantVillage Dataset Downloader")
    print("=" * 60)

    # Check for kaggle credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            print("\n[ERROR] Kaggle credentials not found.")
            print("\nTo set up:")
            print("  1. Go to https://www.kaggle.com/account")
            print("  2. Click 'Create New API Token'")
            print("  3. Place kaggle.json in ~/.kaggle/")
            print("  OR set: KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            sys.exit(1)

    try:
        import kaggle
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"\nDownloading to: {data_dir}")
    print("This may take a few minutes...\n")

    os.system(f"kaggle datasets download -d emmarex/plantdisease -p {data_dir} --unzip")

    # Find and organise the dataset folder
    possible_dirs = [
        os.path.join(data_dir, "PlantVillage"),
        os.path.join(data_dir, "plantvillage"),
        os.path.join(data_dir, "plant_disease"),
    ]

    dataset_dir = None
    for d in possible_dirs:
        if os.path.isdir(d):
            dataset_dir = d
            break

    if dataset_dir is None:
        # Try to find any directory with class subfolders
        for item in os.listdir(data_dir):
            full_path = os.path.join(data_dir, item)
            if os.path.isdir(full_path):
                subdirs = [x for x in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, x))]
                if len(subdirs) > 5:
                    dataset_dir = full_path
                    break

    if dataset_dir:
        print(f"\n[OK] Dataset found at: {dataset_dir}")
        classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        print(f"[OK] Number of classes: {len(classes)}")
        print("\nSample classes:")
        for c in sorted(classes)[:5]:
            count = len(os.listdir(os.path.join(dataset_dir, c)))
            print(f"  {c}: {count} images")
        print(f"\nDataset ready. Update DATA_DIR in config.py if needed.")
    else:
        print("\n[WARNING] Dataset downloaded but directory structure unclear.")
        print(f"Please check: {data_dir}")
        print("Update DATA_DIR in config.py to point to the folder containing class subfolders.")


if __name__ == "__main__":
    download_plantvillage()
