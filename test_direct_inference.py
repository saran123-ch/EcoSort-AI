import os
import sys
from PIL import Image
from src.model import load_model
from src.predict import predict

image_path = os.path.join("artifacts", "test_sample_metal.jpg")
print(f"Image exists: {os.path.exists(image_path)}")
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found!")
    sys.exit(1)

img = Image.open(image_path)
print(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")

checkpoints = [
    ("FINE-TUNED V2 (NEW PRODUCTION)", os.path.join("artifacts", "ecosort_model_v2_enhanced.pth")),
    ("PRODUCTION MODEL V1", os.path.join("artifacts", "ecosort_model.pth")),
    ("ENHANCED CANDIDATE MODEL V1", os.path.join("artifacts", "ecosort_candidate_model.pth")),
    ("ORIGINAL BASELINE MODEL", os.path.join("artifacts", "ecosort_model_v1_baseline.pth")),
]

for label, ckpt_path in checkpoints:
    print(f"\n{'='*20} {label} {'='*20}")
    print(f"Checkpoint path: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} missing!")
        continue
    model = load_model(ckpt_path)
    res = predict(model, img)
    print(f"Expected Label: Metal")
    print(f"Predicted Class: {res['predicted_class']}")
    print(f"Raw Predicted Class: {res['raw_predicted_class']}")
    print(f"Confidence: {res['confidence']*100:.2f}%")
    print(f"Margin: {res['margin']*100:.2f}%")
    print(f"Entropy: {res['entropy']:.4f}")
    print(f"Energy: {res['energy']:.4f}")
    print(f"Decision Status: {res['decision_status']}")
    print(f"Is Confident: {res['is_confident']}")
    print(f"Is Out of Scope: {res['is_out_of_scope']}")
    print(f"Sorting Guidance Bin: {res['sorting_guidance']['bin']}")
    print("Top 3 Predictions:")
    for cls, score in res['top3']:
        print(f"  - {cls}: {score*100:.2f}%")
