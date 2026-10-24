"""
EcoSort AI - Configuration Constants
=====================================
Central configuration for the entire project.
All hyperparameters and paths are defined here for reproducibility.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", "realwaste-main", "RealWaste"
)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ecosort_model_v2_enhanced.pth")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.json")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
LEARNING_CURVES_PATH = os.path.join(ARTIFACTS_DIR, "learning_curves.png")
CLASS_DISTRIBUTION_PATH = os.path.join(ARTIFACTS_DIR, "class_distribution.png")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# The 9 classes in the RealWaste dataset (alphabetical order as read by
# torchvision.datasets.ImageFolder).  We fix this list so that label
# indices are deterministic even if folder ordering changes.
CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224          # MobileNetV2 expected input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 25           # maximum epochs (early stopping may stop sooner)
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Prediction thresholds
# ---------------------------------------------------------------------------
# Provisional confidence threshold.  Below this value the UI shows a
# reliability warning asking the user to verify the prediction.
# This is NOT a scientifically validated threshold — it is a documented
# provisional choice whose behaviour is examined on validation data.
CONFIDENCE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Sorting guidance mapping
# ---------------------------------------------------------------------------
# Maps each predicted class to a *general* sorting suggestion.
# IMPORTANT: Local waste rules vary.  These suggestions are educational
# only and do not constitute official disposal guidance.
SORTING_GUIDANCE = {
    "Cardboard": {
        "bin": "Recycling (Paper/Cardboard)",
        "suggestion": "Flatten the cardboard and place it in the paper/cardboard recycling bin. Remove any tape or staples if possible.",
        "icon": "♻️",
    },
    "Food Organics": {
        "bin": "Organic / Compost",
        "suggestion": "Place in the organic waste or compost bin. Avoid mixing with non-organic materials.",
        "icon": "🍂",
    },
    "Glass": {
        "bin": "Recycling (Glass)",
        "suggestion": "Rinse the glass container and place it in the glass recycling bin. Remove lids if they are a different material.",
        "icon": "♻️",
    },
    "Metal": {
        "bin": "Recycling (Metal)",
        "suggestion": "Rinse metal cans and place in the metal recycling bin. Aluminium and steel cans are commonly recyclable.",
        "icon": "♻️",
    },
    "Miscellaneous Trash": {
        "bin": "General Waste / Landfill",
        "suggestion": "This item may not fit standard recycling categories. Place in general waste unless your local facility accepts it.",
        "icon": "🗑️",
    },
    "Paper": {
        "bin": "Recycling (Paper)",
        "suggestion": "Place clean, dry paper in the paper recycling bin. Heavily soiled paper (e.g., greasy pizza boxes) may need to go in general waste.",
        "icon": "♻️",
    },
    "Plastic": {
        "bin": "Recycling (Plastic) — check local rules",
        "suggestion": "Rinse and place in the plastics recycling bin. Note: not all plastic types are recyclable everywhere — check the resin code and your local guidelines.",
        "icon": "♻️",
    },
    "Textile Trash": {
        "bin": "Textile Recycling / Donation",
        "suggestion": "If in good condition, consider donating. Otherwise, check if your area has a textile recycling programme.",
        "icon": "👕",
    },
    "Vegetation": {
        "bin": "Green Waste / Compost",
        "suggestion": "Place in the green waste or compost bin. Garden trimmings and plant matter are typically compostable.",
        "icon": "🌿",
    },
}

# ---------------------------------------------------------------------------
# Dataset citation (for display in the app and README)
# ---------------------------------------------------------------------------
DATASET_CITATION = (
    "Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G"
)

DATASET_URL = "https://archive.ics.uci.edu/dataset/908/realwaste"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
