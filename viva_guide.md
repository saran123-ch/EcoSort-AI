# EcoSort AI — Viva Preparation Guide

## 1. 60-Second Project Explanation

> "My project is called EcoSort AI. It's a deep learning prototype that
> classifies waste images into nine material categories — like cardboard,
> glass, plastic, and so on — using transfer learning with MobileNetV2.
>
> What makes it different from a basic classifier is that it doesn't just
> show a predicted label. It also shows the model's confidence score, the
> top three predictions, and a warning message when the model isn't sure.
> This helps users understand when they should double-check the result.
>
> I trained it on the RealWaste dataset from the UCI Machine Learning
> Repository, which has about 4,750 real landfill images. The app is
> built with Streamlit and shows the evaluation metrics alongside the
> predictions."

---

## 2. The Real-World Problem

**Problem:** When people throw waste into the wrong bin, it contaminates
recyclable materials and makes waste processing harder. An automated system
that can identify waste types from photos could help — but only if it also
tells you when it's unsure.

**Proposed solution:** A web app where you upload a photo of waste, the
model predicts the material type, and it clearly warns you if the prediction
might be wrong.

---

## 3. Why Deep Learning Is Appropriate

- Waste images are **visually complex** — different textures, shapes, colours,
  and conditions make simple rule-based approaches impractical.
- **Convolutional Neural Networks (CNNs)** can automatically learn visual
  features (edges, textures, shapes) from images without manual feature
  engineering.
- **Transfer learning** lets us use a model (MobileNetV2) that already
  learned general visual features from millions of images (ImageNet), and
  we just retrain the final classification layer for our 9 waste classes.
  This works well even with a relatively small dataset.

---

## 4. Key Concepts Explained Simply

### What is a CNN?
A Convolutional Neural Network processes images through multiple layers.
Early layers detect simple features like edges and corners. Deeper layers
combine these into complex patterns like textures and shapes. The final
layers use these patterns to make a classification decision.

### What is Transfer Learning?
Instead of training a model from scratch (which needs millions of images),
we take a model that was already trained on a large dataset (ImageNet with
1.2 million images) and reuse its learned features. We only replace and
retrain the final classification layer for our specific task. This saves
time and works well with smaller datasets.

### What is MobileNetV2?
A lightweight CNN architecture designed by Google for mobile devices. It's
efficient (small model size, fast inference) while still being accurate.
It uses "inverted residual blocks" with depthwise separable convolutions
to reduce computation.

---

## 5. Why the RealWaste Dataset

- It's from the **UCI Machine Learning Repository**, a well-known academic
  source.
- It contains **real landfill images**, not stock photos — so the model
  learns from realistic conditions.
- It has **9 clearly defined material classes** with reasonable image counts.
- It's **licensed under CC BY 4.0**, so we can use it freely with attribution.
- Total size (4,752 images) is manageable for training on standard hardware.

---

## 6. Train / Validation / Test — What's the Difference?

| Split | Purpose | Used for tuning? |
|---|---|---|
| **Training set (70%)** | The model learns from these images | N/A — it's what the model trains on |
| **Validation set (15%)** | Used to monitor performance *during* training and trigger early stopping | Yes — influences when we stop training |
| **Test set (15%)** | Evaluated *once* after training is complete to report final performance | No — never used for any training decisions |

The split is **stratified** — each split has roughly the same proportion
of each class, so no class is accidentally missing from the test set.

---

## 7. Metrics Explained

### Accuracy
The percentage of all predictions that are correct. Simple but can be
misleading with imbalanced classes.

### Precision
"Of everything the model predicted as class X, how many actually were
class X?" High precision means few false positives.

### Recall
"Of everything that actually is class X, how many did the model
correctly identify?" High recall means few false negatives.

### F1-Score
The harmonic mean of precision and recall. Useful when you want a
single number that balances both. Ranges from 0 to 1.

**Example:** If the model predicts "Glass" 100 times, and 80 of those
are actually glass (precision = 80%), but there are 120 actual glass
images total and it only found 80 of them (recall = 67%), then
F1 = 2 * (0.80 * 0.67) / (0.80 + 0.67) ≈ 0.73.

---

## 8. What Confidence Means and Does NOT Mean

**What it is:** The highest value in the model's softmax output. It
represents how strongly the model's internal computation favours one
class over others.

**What it is NOT:** It is NOT the probability that the prediction is
correct. A model can be 95% confident and still be wrong. This happens
especially with images that are very different from the training data.

**Why we show it:** It gives users a rough signal — if confidence is
very low, the model is "hesitant" and the user should be extra careful.

---

## 9. Why the Model Can Fail

1. **Unfamiliar images:** If the uploaded image looks nothing like the
   training data (different lighting, background, angle), the model may
   predict confidently but incorrectly.

2. **Ambiguous items:** Some waste items genuinely belong to multiple
   categories or are hard to classify even for humans.

3. **Dataset limitations:** The training data comes from one landfill
   in one geographic location. Waste items look different in different
   regions.

4. **Image quality:** Blurry, dark, or partially obscured images give
   the model less information to work with.

5. **Class imbalance:** Some classes (Plastic: 921 images) have much
   more training data than others (Textile Trash: 318 images), which
   can bias predictions.

---

## 10. Three Genuine Limitations

1. **Limited training data:** 4,752 images from a single source is
   small compared to industrial-scale datasets. The model is a prototype,
   not production-ready.

2. **No out-of-distribution detection:** If you upload an image of a
   cat, the model will still predict one of the 9 waste categories. It
   has no way to say "this is not waste."

3. **No hazardous waste awareness:** The model cannot identify batteries,
   chemicals, or medical waste that require special handling.

---

## 11. Five Likely Faculty Questions with Answers

### Q1: "Why did you use transfer learning instead of training from scratch?"
Training a CNN from scratch requires a very large dataset (typically
hundreds of thousands of images) and significant computation time.
The RealWaste dataset has only 4,752 images. Transfer learning lets us
leverage features already learned from ImageNet's 1.2 million images.
We only need to train the classifier head, which is faster and gives
better results with limited data.

### Q2: "How did you handle class imbalance?"
The dataset has unequal class sizes (318 to 921 images per class). I
used **inverse-frequency class weights** in the cross-entropy loss
function. This makes the loss function penalise mistakes on
under-represented classes more heavily, so the model doesn't just
learn to predict the majority class.

### Q3: "What does the confidence threshold of 70% mean?"
It's a provisional, documented cutoff. When the model's highest
softmax score is below 70%, we show a warning to the user. I chose
this value as a reasonable starting point and then examined its
behaviour on the test data — checking what percentage of predictions
fall below it and whether those uncertain predictions are indeed less
accurate. It's not a scientifically optimised threshold.

### Q4: "How do you prevent overfitting?"
Four mechanisms: (1) **Transfer learning** — using pretrained weights
provides good feature representations without overfitting to our small
dataset. (2) **Data augmentation** — random flips, rotations, and
colour jitter create variation in training data. (3) **Dropout** — randomly
disabling neurons during training prevents co-adaptation.
(4) **Early stopping** — I monitor validation loss and stop training if
it hasn't improved for 5 consecutive epochs.

### Q5: "Can this be deployed in a real waste facility?"
Not as-is. This is an educational prototype. For real deployment, you
would need: a much larger and more diverse dataset, extensive testing
across different conditions, proper out-of-distribution detection,
integration with facility-specific sorting rules, hardware optimisation,
and regulatory compliance. The model should be treated as a decision
support tool, not an autonomous system.

---

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Framework | PyTorch |
| Architecture | MobileNetV2 (pretrained on ImageNet) |
| Dataset | RealWaste (UCI ML Repository) |
| Total images | 4,752 |
| Classes | 9 |
| Split | 70% train / 15% val / 15% test (stratified) |
| Loss | Cross-entropy with class weights |
| Optimizer | Adam (lr=0.001) |
| Early stopping | Patience = 5 epochs |
| Confidence threshold | 0.70 (provisional) |
| Input size | 224 × 224 pixels |
| App framework | Streamlit |
