# EcoSort AI — Smart Waste Sorting Assistant

**Deep Learning-Based Waste Classification with Confidence-Aware Predictions**

EcoSort AI is a deep learning prototype that classifies waste images into nine
material categories using transfer learning with MobileNetV2, and provides
confidence-aware sorting suggestions. When the model is uncertain, it warns
the user to verify the result.

> **Disclaimer:** This is an educational prototype developed as a BTech deep
> learning project. It is not an operational waste-management system. Always
> check your local waste disposal guidelines.

---

## Problem Statement

Incorrect waste segregation makes recycling and waste processing more
difficult. People may not know whether an item belongs to a particular material
category, and image classifiers can make mistakes when images are blurry,
unusual, or different from their training data.

EcoSort AI addresses this by:
- Classifying waste images into 9 material categories
- Displaying the model's confidence score (clearly labelled as model confidence)
- Showing top-3 predictions for transparency
- Warning users when the prediction should be verified
- Providing general sorting suggestions with appropriate caveats

---

## Dataset

**RealWaste** — UCI Machine Learning Repository

- **Source:** https://archive.ics.uci.edu/dataset/908/realwaste
- **Citation:** Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste
  [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Images:** 4,752 images across 9 classes
- **Classes:** Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash,
  Paper, Plastic, Textile Trash, Vegetation

---

## Model Architecture & Inference Pipeline

| Component | Details |
|---|---|
| Backbone | MobileNetV2 (pretrained on ImageNet, upper blocks 14–18 fine-tuned) |
| Strategy | Transfer learning with fine-tuning & multi-domain generalization |
| Classifier | Linear(1280→256) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(256→9) |
| Input | 224 × 224 RGB images with full-aspect ratio letterbox padding (preserves object contours & packaging geometry) |
| Framework | PyTorch + torchvision |
| Loss | Cross-entropy with inverse-frequency class weights |
| Optimizer | AdamW with differential learning rates (backbone 4e-5, head 4e-4) |
| Regularisation | Early stopping, Dropout, Cosine Annealing, Camera-realistic Perspective/Jitter augmentations |
| Safety & OOD | Energy- and normalized entropy-based Out-Of-Distribution screening (92.9% safety on medical/electronic waste) |

---

## Installation and Setup (Windows)

### Prerequisites
- Python 3.10 or later
- pip package manager
- ~2 GB free disk space (for dependencies and dataset)

### Step 1: Clone or download this repository
```bash
git clone <repository-url>
cd EcoSort-AI
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train or Fine-Tune Candidate Models
```bash
# Standard RealWaste baseline training:
python train_model.py

# High-generalization composite fine-tuning (V2 Production Model):
python train_enhanced_v2.py
```

### Step 4: Evaluate Models & Run Multi-Model Benchmark
```bash
# Standard held-out test set evaluation:
python evaluate_model.py

# Comprehensive multi-model benchmark (RealWaste + Real-World Camera Suite):
python benchmark_all_models.py
```
This generates metrics, confusion matrices, and comparative benchmark results in `artifacts/model_comparison_benchmark.json`:

| Model Checkpoint | RealWaste Test Acc | RealWaste Macro F1 | Real-World Camera Acc | Real-World Macro F1 | Plastic Recall | Cardboard Recall | OOD Safety |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Baseline** (`ecosort_model_v1_baseline.pth`) | 80.53% | 0.8108 | 28.57% | 0.3118 | 7.1% | 40.0% | 61.4% |
| **Production Model V1** (`ecosort_model.pth`) | 87.20% | 0.8793 | 68.10% | 0.6988 | 47.1% | 60.0% | 88.6% |
| **Candidate Model V1** (`ecosort_candidate_model.pth`) | 81.36% | 0.8135 | 64.29% | 0.6447 | 55.7% | 68.6% | 92.9% |
| **Production Model V2** (`ecosort_model_v2_enhanced.pth`) | **87.62%** | **0.8833** | **70.00%** | **0.7159** | **55.7%** | **74.3%** | **92.9%** |
| **Production Model V3** (`ecosort_model_v3_production.pth`) | **86.09%** | **0.8685** | **68.10%** | **0.7032** | **54.3%** | **74.3%** | **98.6%** |

### Step 5: Run the Web Application
```bash
streamlit run app.py
```
Open http://localhost:8501. The app features live camera capture, image uploads, and an interactive model checkpoint switcher defaulting to `Production Model V2`.

---

## Running Tests

```bash
python -m tests.test_smoke
```

---

## Out-of-Distribution (OOD) Screening & Reliability Guardrails

Real-world waste streams often encounter items outside the 9 municipal categories (e.g. blister packs, clinical syringes, e-waste, lithium batteries). EcoSort AI implements an autonomous 3-tier decision rule:
1. **`OUT_OF_SCOPE`**: Low top score (<40%), high Shannon entropy (>0.72), or weak energy score (>-2.8). Automatically blocks recycling advice and directs the user to hazardous/special waste authorities.
2. **`CONFIDENT`**: High confidence (≥65%), solid margin (≥18%), low entropy (≤0.55), and negative energy (≤-4.0). Displays confirmed sorting advice with bin guidelines.
3. **`UNCERTAIN`**: Ambiguous candidate distributions mandate visual verification before sorting to prevent bin contamination.

---

## Acknowledgements

- **Dataset:** Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste.
  UCI Machine Learning Repository.
- **Pretrained model:** MobileNetV2 weights from torchvision (originally
  trained on ImageNet).
- **Frameworks:** PyTorch, Streamlit, scikit-learn, matplotlib, seaborn.
