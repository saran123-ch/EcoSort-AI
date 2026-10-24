"""
EcoSort AI — Smart Waste Sorting Assistant
============================================
Streamlit web application for waste image classification
with confidence-aware predictions and sorting guidance.

Usage:
    streamlit run app.py

DISCLAIMER: This is an educational prototype, NOT an operational
waste-management authority. Predictions are model suggestions and
should be verified. Local waste rules vary.
"""

import os
import sys
import json

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MODEL_PATH, METRICS_PATH, HISTORY_PATH,
    CONFUSION_MATRIX_PATH, LEARNING_CURVES_PATH,
    CLASS_DISTRIBUTION_PATH, CLASS_NAMES, CONFIDENCE_THRESHOLD,
    SORTING_GUIDANCE, DATASET_CITATION, DATASET_URL, DATASET_LICENSE,
    NUM_CLASSES, IMAGE_SIZE,
)
from src.model import load_model
from src.predict import predict

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI — Smart Waste Sorting",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        color: #2E7D32;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #555;
        font-size: 1.1rem;
    }

    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2E7D32;
    }
    .prediction-card h3 {
        color: #1B5E20;
        margin-bottom: 0.5rem;
    }

    /* Warning card */
    .warning-card {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #E65100;
    }
    .warning-card h3 {
        color: #BF360C;
    }

    /* Guidance card */
    .guidance-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1565C0;
    }
    .guidance-card h3 {
        color: #0D47A1;
    }

    /* Disclaimer */
    .disclaimer {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #666;
        border: 1px solid #DDD;
    }

    /* Confidence bar */
    .confidence-bar {
        background: #E0E0E0;
        border-radius: 10px;
        height: 24px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }

    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stat-card h2 {
        color: #2E7D32;
        font-size: 2rem;
        margin: 0;
    }
    .stat-card p {
        color: #666;
        margin: 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8F5E9 0%, #FFFFFF 100%);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load the trained model (cached across reruns)."""
    if not os.path.exists(MODEL_PATH):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)
    return model


@st.cache_data
def get_metrics():
    """Load saved evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def get_history():
    """Load training history."""
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ♻️ EcoSort AI")
    st.markdown("**Smart Waste Sorting Assistant**")
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "Go to:",
        ["🔍 Classify Waste", "📊 Model Performance", "📖 About & Dataset"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Quick Info")
    st.markdown(f"- **Model:** MobileNetV2")
    st.markdown(f"- **Classes:** {NUM_CLASSES}")
    st.markdown(f"- **Input size:** {IMAGE_SIZE}×{IMAGE_SIZE}")
    st.markdown(f"- **Confidence threshold:** {CONFIDENCE_THRESHOLD:.0%}")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <strong>Educational Prototype</strong><br>'
        'This app is a student project for learning purposes. '
        'It is not an operational waste-management system. '
        'Always check your local waste disposal guidelines.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1: Classify Waste
# ---------------------------------------------------------------------------
if page == "🔍 Classify Waste":
    st.markdown(
        '<div class="main-header">'
        '<h1>♻️ EcoSort AI</h1>'
        '<p>Upload a waste image to get a sorting suggestion with confidence assessment</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Problem statement
    with st.expander("📋 Problem Statement", expanded=False):
        st.markdown("""
        **Incorrect waste segregation** makes recycling and waste processing more
        difficult. People may not know whether an item belongs to a particular
        material category, and image classifiers can make mistakes when images
        are blurry, unusual, or different from their training data.

        **EcoSort AI** classifies waste images into 9 material categories, presents
        a cautious sorting suggestion, and communicates when its prediction should
        be verified — helping users make more informed disposal decisions.
        """)

    # Load model
    model = get_model()

    if model is None:
        st.error(
            "🚫 **Model not found!**\n\n"
            f"Expected model file at: `{MODEL_PATH}`\n\n"
            "Please run the training pipeline first:\n"
            "```\npython train_model.py\npython evaluate_model.py\n```"
        )
        st.stop()
    
    # File uploader
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a waste image (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a photo of a waste item to classify it.",
        )

    medicine_packaging = st.checkbox(
        "This image contains medicine packaging or medical waste"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            if image.mode not in ("RGB", "L", "RGBA"):
                st.warning("Unusual image format. Converting to RGB...")
                image = image.convert("RGB")
        except Exception as e:
            st.error(f"❌ Could not read the uploaded file: {e}")
            st.stop()

        with col_preview:
            st.markdown("### 🖼️ Uploaded Image")
            st.image(image, use_container_width=True)

        # Predict button
        if st.button("🔍 Classify Waste", type="primary", use_container_width=True):

            if medicine_packaging:
                st.warning(
                    "Unsupported item: EcoSort AI is not trained to identify "
                    "medicine packaging or medical waste. No waste category "
                    "or recycling-bin recommendation will be provided. "
                    "Check official local disposal guidance."
                )
                st.stop()

            with st.spinner("Analysing image..."):
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                result = predict(model, image, device)

            # Results
            st.markdown("---")
            st.markdown("## 📋 Prediction Results")

            # Confidence color
            conf = result["confidence"]
            if conf >= 0.85:
                conf_color = "#4CAF50"  # green
            elif conf >= CONFIDENCE_THRESHOLD:
                conf_color = "#FF9800"  # orange
            else:
                conf_color = "#F44336"  # red

            # Main prediction card
            col1, col2 = st.columns([2, 1])
            with col1:
                icon = result["sorting_guidance"]["icon"]
                st.markdown(
                    f'<div class="prediction-card">'
                    f'<h3>{icon} Predicted Category</h3>'
                    f'<h2 style="color: #1B5E20; font-size: 1.8rem;">'
                    f'{result["predicted_class"]}</h2>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f'<div class="prediction-card">'
                    f'<h3>Model Confidence</h3>'
                    f'<h2 style="color: {conf_color}; font-size: 1.8rem;">'
                    f'{conf:.1%}</h2>'
                    f'<p style="font-size: 0.8rem; color: #666;">'
                    f'(This is the model\'s internal confidence, not a guaranteed '
                    f'probability of correctness.)</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Confidence bar
            st.markdown(
                f'<div class="confidence-bar">'
                f'<div class="confidence-fill" style="width: {conf*100}%; '
                f'background: {conf_color};">{conf:.1%}</div></div>',
                unsafe_allow_html=True,
            )

            # Reliability warning
            
            # Reliability warning — always shown
            st.warning(
                "**Important: Verify every prediction.** "
                "EcoSort AI predicts one of nine waste categories: "
                "Cardboard, Food Organics, Glass, Metal, "
                "Miscellaneous Trash, Paper, Plastic, Textile Trash, "
                "or Vegetation. Items outside these categories, including "
                "medicine packaging and medical waste, are not specifically "
                "supported. A high confidence score does not guarantee "
                "correct identification. Mixed-material packaging may need "
                "separate material assessment. Do not use this prediction "
                "alone to decide how to dispose of medical or hazardous waste; "
                "follow official local disposal guidance."
            )


            # Top 3 predictions
            st.markdown("### 🏆 Top 3 Predictions")
            top3_cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for i, (cls_name, score) in enumerate(result["top3"]):
                with top3_cols[i]:
                    st.metric(
                        label=f"{medals[i]} {cls_name}",
                        value=f"{score:.1%}",
                    )

            # All class scores
            with st.expander("📊 All Class Scores", expanded=False):
                scores = result["all_scores"]
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                fig, ax = plt.subplots(figsize=(10, 4))
                names = [s[0] for s in sorted_scores]
                values = [s[1] for s in sorted_scores]
                colors = [conf_color if n == result["predicted_class"] else "#90CAF9"
                         for n in names]
                bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                              edgecolor="white", linewidth=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Score")
                ax.set_title("All Class Prediction Scores")
                for bar, val in zip(bars, values[::-1]):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                           f"{val:.1%}", va="center", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Sorting guidance
            st.markdown("---")
            guidance = result["sorting_guidance"]
            st.markdown(
                f'<div class="guidance-card">'
                f'<h3>{guidance["icon"]} Tentative Sorting Suggestion — Verify First</h3>'
                f'<p><strong>Suggested bin:</strong> {guidance["bin"]}</p>'
                f'<p>{guidance["suggestion"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Caveat
            st.markdown(
                '<div class="disclaimer">'
                '📌 <strong>Important:</strong> This sorting suggestion is based on '
                'the predicted material category and general guidelines. '
                'Local waste disposal rules vary significantly. '
                'Always check with your local waste management authority for '
                'specific disposal instructions. This application does not provide '
                'hazardous waste handling advice — consult official local guidance '
                'for hazardous materials.'
                '</div>',
                unsafe_allow_html=True,
            )

    else:
        st.info("👆 Upload an image to get started!")


# ---------------------------------------------------------------------------
# Page 2: Model Performance
# ---------------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.markdown(
        '<div class="main-header">'
        '<h1>📊 Model Performance</h1>'
        '<p>Evaluation metrics from the held-out test set</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics = get_metrics()
    history = get_history()

    if metrics is None:
        st.warning(
            "⚠️ Evaluation metrics not found. "
            "Please run `python evaluate_model.py` first."
        )
        st.stop()

    # Summary stats
    st.markdown("### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="stat-card">'
            f'<h2>{metrics["overall_accuracy"]}%</h2>'
            f'<p>Test Accuracy</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        macro_f1 = metrics["classification_report"].get("macro avg", {}).get("f1-score", 0)
        st.markdown(
            f'<div class="stat-card">'
            f'<h2>{macro_f1:.3f}</h2>'
            f'<p>Macro F1-Score</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card">'
            f'<h2>{metrics["test_set_size"]}</h2>'
            f'<p>Test Samples</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        threshold_info = metrics.get("confidence_threshold_analysis", {})
        conf_acc = threshold_info.get("confident_accuracy", "N/A")
        st.markdown(
            f'<div class="stat-card">'
            f'<h2>{conf_acc}%</h2>'
            f'<p>Accuracy (confident)</p></div>',
            unsafe_allow_html=True,
        )

    # Per-class metrics
    st.markdown("### 📋 Per-Class Metrics")
    report = metrics["classification_report"]
    class_data = []
    for cls in CLASS_NAMES:
        if cls in report:
            r = report[cls]
            class_data.append({
                "Class": cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1-Score": f"{r['f1-score']:.3f}",
                "Support": int(r["support"]),
            })
    if class_data:
        st.table(class_data)

    # Confusion matrix
    st.markdown("### 🔢 Confusion Matrix")
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
    else:
        st.info("Confusion matrix plot not found. Run evaluate_model.py to generate it.")

    # Learning curves
    st.markdown("### 📉 Learning Curves")
    if os.path.exists(LEARNING_CURVES_PATH):
        st.image(LEARNING_CURVES_PATH, use_container_width=True)
    elif history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
        ax1.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Accuracy")
        ax2.plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Training history not available.")

    # Confidence threshold analysis
    st.markdown("### 🎯 Confidence Threshold Analysis")
    if "confidence_threshold_analysis" in metrics:
        ta = metrics["confidence_threshold_analysis"]
        st.markdown(f"""
        The provisional confidence threshold is **{ta['threshold']:.0%}**.
        This threshold was chosen as a documented starting point and its behaviour
        was examined on the test data below.

        | Metric | Confident Predictions | Uncertain Predictions |
        |---|---|---|
        | **Count** | {ta['confident_predictions']} ({ta['confident_percentage']}%) | {ta['uncertain_predictions']} ({ta['uncertain_percentage']}%) |
        | **Accuracy** | {ta['confident_accuracy']}% | {ta['uncertain_accuracy']}% |

        *Note: This threshold is NOT scientifically validated. It is a provisional
        design choice. A different threshold may be more appropriate depending on
        the deployment context.*
        """)

    # Class distribution
    st.markdown("### 📊 Dataset Class Distribution")
    if os.path.exists(CLASS_DISTRIBUTION_PATH):
        st.image(CLASS_DISTRIBUTION_PATH, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3: About & Dataset
# ---------------------------------------------------------------------------
elif page == "📖 About & Dataset":
    st.markdown(
        '<div class="main-header">'
        '<h1>📖 About EcoSort AI</h1>'
        '<p>Project information, dataset details, and limitations</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # About
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    **EcoSort AI** is a deep learning-based prototype that classifies waste
    images into 9 material categories and provides confidence-aware sorting
    suggestions.

    **Key features:**
    - Transfer learning with MobileNetV2 pretrained on ImageNet
    - Confidence-aware predictions with reliability warnings
    - Top-3 class scores for transparency
    - General sorting guidance with clear caveats
    - Complete evaluation with per-class metrics
    """)

    # Dataset
    st.markdown("### 📦 Dataset Information")
    st.markdown(f"""
    **Dataset:** RealWaste

    **Source:** [UCI Machine Learning Repository]({DATASET_URL})

    **Citation:**
    > {DATASET_CITATION}

    **License:** {DATASET_LICENSE}

    **Description:** An image classification dataset of waste items across
    9 major material types, collected within an authentic landfill environment.
    Images are 524×524 pixels and represent real waste items as they appear
    at the point of reception.

    **Classes:** {", ".join(CLASS_NAMES)}

    **Total images:** 4,752
    """)

    # Model architecture
    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    | Component | Details |
    |---|---|
    | **Backbone** | MobileNetV2 (pretrained on ImageNet) |
    | **Strategy** | Transfer learning — frozen feature extractor |
    | **Classifier** | Custom head: Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→9) |
    | **Input size** | 224 × 224 pixels (RGB) |
    | **Framework** | PyTorch + torchvision |
    | **Loss** | Cross-entropy with inverse-frequency class weights |
    | **Optimizer** | Adam (lr=0.001) |
    | **Regularisation** | Early stopping (patience=5), Dropout, Data augmentation |
    """)

    # Limitations
    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    1. **Dataset scope:** The model was trained on the RealWaste dataset
       (4,752 images from one landfill facility). It may not generalise well
       to waste items from different regions, lighting conditions, or
       backgrounds.

    2. **Confidence ≠ correctness:** The model's confidence score reflects
       its internal certainty, NOT the actual probability of being correct.
       High confidence does not guarantee a correct prediction.

    3. **No hazardous waste detection:** This model does not identify
       hazardous materials (batteries, chemicals, medical waste, etc.).
       Always consult official local guidance for hazardous waste disposal.

    4. **Local rules vary:** The sorting suggestions are general and may
       not match your local waste management rules. Always check with your
       local authority.

    5. **Single-item focus:** The model expects images of individual waste
       items. Mixed or cluttered scenes may produce unreliable results.

    6. **Not an authority:** This is an educational prototype, not an
       operational waste-management system. Do not rely on it as the sole
       basis for waste disposal decisions.
    """)

    # Future improvements
    st.markdown("### 🔮 Future Improvements")
    st.markdown("""
    - Train on larger, more diverse waste datasets
    - Add out-of-distribution detection for truly unfamiliar images
    - Implement Grad-CAM visualisations to show what the model focuses on
    - Support multi-item detection in a single image
    - Add localisation-aware sorting rules
    - Fine-tune deeper layers of MobileNetV2 for better accuracy
    - Deploy as a mobile application for field use
    """)

    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    | Component | Technology |
    |---|---|
    | Deep Learning | PyTorch + torchvision |
    | Pretrained Model | MobileNetV2 (ImageNet weights) |
    | Web Framework | Streamlit |
    | Evaluation | scikit-learn |
    | Visualisation | matplotlib, seaborn |
    | Language | Python 3.14 |
    """)
