# EcoSort AI — Smart Waste Sorting Assistant with Confidence-Aware Predictions

## Abstract

Incorrect waste segregation remains a practical barrier to effective recycling
and waste processing. While image classification models can help automate
material identification, standard classifiers typically present a single
predicted label without communicating the reliability of that prediction.
This can mislead users when the model encounters ambiguous, blurry, or
unfamiliar images.

**EcoSort AI** is a deep learning prototype that classifies waste images into
nine material categories — Cardboard, Food Organics, Glass, Metal,
Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation — using
transfer learning with MobileNetV2. Beyond basic classification, the system
implements **confidence-aware predictions**: it displays the model's internal
confidence score (clearly labelled as model confidence, not guaranteed
correctness), presents the top three class scores for transparency, and shows
a prominent verification warning when confidence falls below a documented
provisional threshold.

The model is trained and evaluated on the **RealWaste** dataset from the
UCI Machine Learning Repository (Single et al., 2023), which contains 4,752
images of authentic waste items captured in a landfill environment. The
training pipeline uses stratified splitting, data augmentation (applied only
to training data), class-weighted cross-entropy loss to handle class
imbalance, and early stopping to prevent overfitting.

A Streamlit web application allows users to upload waste images, view
predictions with confidence assessments, and receive general sorting
suggestions with clear caveats about local rule variation. The application
also displays genuine evaluation metrics including per-class precision,
recall, F1-score, confusion matrix, and learning curves.

This project is an **educational prototype** developed as a BTech deep
learning assignment. It does not claim to solve waste management, guarantee
correct recycling decisions, or constitute a novel research contribution.
Its primary contribution is demonstrating how a standard image classifier
can be augmented with reliability-aware features that communicate uncertainty
to end users.

---

**Dataset Citation:**
Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G

**Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
