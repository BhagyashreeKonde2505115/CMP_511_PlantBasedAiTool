"""
config.py
---------
Central configuration for the Plant Disease Detection project.
Edit DATA_DIR to point to your PlantVillage dataset folder.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"D:\Bhagyashree\CMP511\New folder\Code\PlantVillage"
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
UPLOAD_DIR = os.path.join(BASE_DIR, "webapp", "static", "uploads")

for _dir in [MODEL_DIR, RESULTS_DIR, UPLOAD_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Image settings ─────────────────────────────────────────────────────────
IMG_SIZE     = 224        # pixels (both width and height)
CHANNELS     = 3

# ── Dataset split ──────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

# ── Baseline ML ────────────────────────────────────────────────────────────
# Feature size for handcrafted features (colour hist + texture)
FEATURE_SIZE = 512        # colour histogram bins per channel * channels + LBP histogram

# ── CNN ────────────────────────────────────────────────────────────────────
CNN_EPOCHS         = 5
CNN_BATCH_SIZE     = 16
CNN_LEARNING_RATE  = 1e-3
CNN_DROPOUT        = 0.5

# ── Transfer Learning ──────────────────────────────────────────────────────
TL_EPOCHS          = 3
TL_BATCH_SIZE      = 16
TL_LEARNING_RATE   = 1e-4   # lower LR for fine-tuning
TL_FINE_TUNE_LAYERS = 10    # unfreeze last N layers of backbone

# ── Available transfer learning backbones ─────────────────────────────────
TL_MODELS = ["resnet50", "vgg16"]

# ── Web App ────────────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False
MAX_UPLOAD_MB = 16
