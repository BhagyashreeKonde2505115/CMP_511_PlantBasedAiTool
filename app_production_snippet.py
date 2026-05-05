# ============================================================
#  ADD THIS BLOCK at the very TOP of webapp/app.py
#  (right after your existing imports)
#
#  Replace the 3 Google Drive links with your actual links.
#  How to get a direct Google Drive link:
#    1. Upload the .pth file to Google Drive
#    2. Right-click → Share → Anyone with the link
#    3. Copy the link — it looks like:
#       https://drive.google.com/file/d/FILE_ID/view?usp=sharing
#    4. Change it to:
#       https://drive.google.com/uc?export=download&id=FILE_ID
# ============================================================

import os
import urllib.request

MODEL_DOWNLOAD_URLS = {
    "saved_models/cnn_model.pth":       "https://drive.google.com/uc?export=download&id=YOUR_CNN_FILE_ID",
    "saved_models/resnet50_model.pth":  "https://drive.google.com/uc?export=download&id=YOUR_RESNET_FILE_ID",
    "saved_models/vgg16_model.pth":     "https://drive.google.com/uc?export=download&id=YOUR_VGG16_FILE_ID",
}

os.makedirs("saved_models", exist_ok=True)

for model_path, url in MODEL_DOWNLOAD_URLS.items():
    if not os.path.exists(model_path):
        print(f"Downloading {model_path} from Google Drive...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"  Done: {model_path}")
        except Exception as e:
            print(f"  WARNING: Could not download {model_path}: {e}")

# ============================================================
#  REPLACE the LAST LINE of webapp/app.py with this block:
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
