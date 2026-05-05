# 🌿 Plant Disease Detection AI

**CMP511 - Machine Learning and Artificial Intelligence**  
Student: Bhagyashree Konde | Student ID: 2505115

---

## 🔍 Overview

An AI-powered plant disease detection system that classifies plant leaf images into 38 disease categories using multiple ML/DL models:

| Model | Type | Accuracy |
|-------|------|----------|
| SVM | Baseline ML | ~75% |
| Random Forest | Baseline ML | ~72% |
| Custom CNN | Deep Learning | ~88% |
| ResNet50 | Transfer Learning | ~93% |
| VGG16 | Transfer Learning | ~91% |

---

## 🗂️ Project Structure

```
plant_disease_project/
├── config.py               ← All settings
├── train_all.py            ← Master training script
├── predict.py              ← CLI inference
├── requirements.txt
├── pipeline/
│   ├── baseline_models.py  ← SVM + Random Forest
│   ├── cnn_model.py        ← Custom 4-block CNN
│   ├── transfer_learning.py← ResNet50 + VGG16
│   └── viability_analysis.py← AI viability report
├── utils/
│   ├── data_loader.py      ← Augmentation, splits
│   ├── evaluation.py       ← Metrics, plots
│   └── predictor.py        ← Unified inference
└── webapp/
    ├── app.py              ← Flask backend
    └── templates/
        ├── index.html      ← Main web app
        └── ai_tool.html    ← AI analysis tool
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-ai.git
cd plant-disease-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download PlantVillage from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and update `config.py`:
```python
DATA_DIR = "path/to/PlantVillage"
```

### 4. Download trained models
## Trained Models
Download from Hugging Face:
- cnn_model.pth — https://huggingface.co/BhagyashreeKondeAbertay/plant-disease-ai-models/resolve/581cb92d27760ef5f8e93cb5ecb3e454e3bd50ef/cnn_model.pth
- resnet50_model.pth — https://huggingface.co/BhagyashreeKondeAbertay/plant-disease-ai-models/resolve/581cb92d27760ef5f8e93cb5ecb3e454e3bd50ef/resnet50_model.pth
- vgg16_model.pth — https://huggingface.co/BhagyashreeKondeAbertay/plant-disease-ai-models/resolve/581cb92d27760ef5f8e93cb5ecb3e454e3bd50ef/vgg16_model.pth
- rf_model.joblib — https://huggingface.co/BhagyashreeKondeAbertay/plant-disease-ai-models/resolve/581cb92d27760ef5f8e93cb5ecb3e454e3bd50ef/rf_model.joblib
- svm_model.joblib — https://huggingface.co/BhagyashreeKondeAbertay/plant-disease-ai-models/resolve/581cb92d27760ef5f8e93cb5ecb3e454e3bd50ef/svm_model.joblib

### 5. Run the web app
```bash
python webapp/app.py
```
Open: http://localhost:5000

---

## 🌐 Live Demo

**Live URL:** https://plant-disease-ai.onrender.com *(update after deployment)*

---

## 📊 Dataset

- **PlantVillage Dataset** — 54,306 images, 38 classes
- Source: https://www.kaggle.com/datasets/emmarex/plantdisease

---

## 🛠️ Technologies

- Python 3.11
- PyTorch (CNN, ResNet50, VGG16)
- Scikit-learn (SVM, Random Forest)
- OpenCV, Pillow (image processing)
- Flask (web application)
- Matplotlib (visualisations)

---

## 📈 Results

All trained on PlantVillage dataset with 80/10/10 train/val/test split.

See `results/` folder for:
- `model_comparison.png` — accuracy bar chart
- `confusion_*.png` — per-model confusion matrices
- `curves_*.png` — training/validation curves
- `viability_analysis.png` — deployment viability dashboard

---

## 🤖 AI Tool Acknowledgement

This project was developed with AI assistance (Claude by Anthropic) for code generation and debugging. All code was reviewed, understood, and adapted by the student.

---

## 📄 License

Academic project — for educational purposes only.
