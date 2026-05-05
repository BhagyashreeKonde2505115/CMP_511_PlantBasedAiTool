"""
utils/predictor.py
------------------
Unified inference interface. Loads whichever trained models are available
and runs predictions on single images.
"""

import os
import io
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import MODEL_DIR, IMG_SIZE


# ── Transform for inference ────────────────────────────────────────────────

_infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_image(source):
    """Load a PIL image from a file path, bytes, or BytesIO."""
    if isinstance(source, (str, os.PathLike)):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    return img


# ── Model loaders ──────────────────────────────────────────────────────────

def _load_torch_model(path, backbone=None):
    """Load a saved PyTorch model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes  = checkpoint["num_classes"]

    if backbone is None or backbone == "cnn":
        from pipeline.cnn_model import PlantDiseaseCNN
        model = PlantDiseaseCNN(num_classes)
    elif backbone == "resnet50":
        from pipeline.transfer_learning import build_resnet50
        model = build_resnet50(num_classes)
    elif backbone == "vgg16":
        from pipeline.transfer_learning import build_vgg16
        model = build_vgg16(num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names


def _extract_features_for_sklearn(img_pil):
    """Extract handcrafted features from a PIL image for sklearn models."""
    import cv2
    # Convert PIL to OpenCV BGR
    img_np  = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Colour histogram
    hists = []
    for ch in range(3):
        h = cv2.calcHist([img_bgr], [ch], None, [64], [0, 256])
        hists.append(cv2.normalize(h, h).flatten())
    color_feat = np.concatenate(hists)

    # LBP texture
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-6)
    except Exception:
        lbp_hist = np.zeros(10)

    return np.concatenate([color_feat, lbp_hist]).reshape(1, -1)


class Predictor:
    """
    Loads all available trained models and exposes a predict() method.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.models  = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_all(model_dir)

    def _load_all(self, model_dir):
        if not os.path.isdir(model_dir):
            print(f"[Predictor] Model directory not found: {model_dir}")
            return

        # ── CNN ───────────────────────────────────────────────
        cnn_path = os.path.join(model_dir, "cnn_model.pth")
        if os.path.exists(cnn_path):
            try:
                model, cls = _load_torch_model(cnn_path, backbone="cnn")
                self.models["Custom CNN"] = (model.to(self._device), cls, "torch")
                print(f"[Predictor] Loaded: Custom CNN")
            except Exception as e:
                print(f"[Predictor] Warning — could not load CNN: {e}")

        # ── ResNet50 ──────────────────────────────────────────
        rn_path = os.path.join(model_dir, "resnet50_model.pth")
        if os.path.exists(rn_path):
            try:
                model, cls = _load_torch_model(rn_path, backbone="resnet50")
                self.models["ResNet50"] = (model.to(self._device), cls, "torch")
                print(f"[Predictor] Loaded: ResNet50")
            except Exception as e:
                print(f"[Predictor] Warning — could not load ResNet50: {e}")

        # ── VGG16 ─────────────────────────────────────────────
        vgg_path = os.path.join(model_dir, "vgg16_model.pth")
        if os.path.exists(vgg_path):
            try:
                model, cls = _load_torch_model(vgg_path, backbone="vgg16")
                self.models["VGG16"] = (model.to(self._device), cls, "torch")
                print(f"[Predictor] Loaded: VGG16")
            except Exception as e:
                print(f"[Predictor] Warning — could not load VGG16: {e}")

        # ── SVM ───────────────────────────────────────────────
        svm_path = os.path.join(model_dir, "svm_model.joblib")
        if os.path.exists(svm_path):
            try:
                import joblib
                pkg = joblib.load(svm_path)
                self.models["SVM"] = (pkg["model"], pkg["class_names"], "sklearn")
                print(f"[Predictor] Loaded: SVM")
            except Exception as e:
                print(f"[Predictor] Warning — could not load SVM: {e}")

        # ── Random Forest ─────────────────────────────────────
        rf_path = os.path.join(model_dir, "rf_model.joblib")
        if os.path.exists(rf_path):
            try:
                import joblib
                pkg = joblib.load(rf_path)
                self.models["Random Forest"] = (pkg["model"], pkg["class_names"], "sklearn")
                print(f"[Predictor] Loaded: Random Forest")
            except Exception as e:
                print(f"[Predictor] Warning — could not load RF: {e}")

        if not self.models:
            print("[Predictor] No trained models found in:", model_dir)

    def get_available_models(self):
        return list(self.models.keys())

    def predict(self, image_source, model_name=None):
        """
        Predict the disease class for a single image.
        Returns dict with: predicted_class, confidence, top5, model_used
        """
        if not self.models:
            return {"error": "No models loaded. Please copy model files to saved_models/ folder."}

        # Pick best available model
        if model_name is None:
            for pref in ["ResNet50", "VGG16", "Custom CNN", "Random Forest", "SVM"]:
                if pref in self.models:
                    model_name = pref
                    break

        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not available. Loaded: {self.get_available_models()}"}

        model_obj, class_names, kind = self.models[model_name]

        try:
            img = _load_image(image_source)

            if kind == "torch":
                tensor = _infer_transform(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    logits = model_obj(tensor)
                    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

                top5_idx   = probs.argsort()[::-1][:5]
                top5       = [(class_names[i], float(probs[i])) for i in top5_idx]
                pred_cls   = class_names[top5_idx[0]]
                confidence = float(probs[top5_idx[0]])

            else:  # sklearn
                feat     = _extract_features_for_sklearn(img)
                pred_idx = model_obj.predict(feat)[0]
                pred_cls = class_names[pred_idx]

                if hasattr(model_obj, "predict_proba"):
                    probs      = model_obj.predict_proba(feat)[0]
                    top5_idx   = probs.argsort()[::-1][:5]
                    top5       = [(class_names[i], float(probs[i])) for i in top5_idx]
                    confidence = float(probs[pred_idx])
                else:
                    top5       = [(pred_cls, 1.0)]
                    confidence = 1.0

            return {
                "predicted_class": pred_cls,
                "confidence":      confidence,
                "top5":            top5,
                "model_used":      model_name,
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
