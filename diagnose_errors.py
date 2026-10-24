import os
import torch
from src.model import load_model
from src.predict import predict
from PIL import Image

device = torch.device('cpu')
ext_test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"

model_prod = load_model('artifacts/ecosort_model.pth', device)
model_v2 = load_model('artifacts/ecosort_model_v2_enhanced.pth', device)

print("="*80)
print("DIAGNOSING PLASTIC ERRORS (First 70 unseen Plastic test images)")
print("="*80)
p_plastic = os.path.join(ext_test_dir, 'plastic')
pl_files = [f for f in os.listdir(p_plastic) if f.lower().endswith(('.jpg','.jpeg','.png'))][:70]

for f in pl_files:
    fp = os.path.join(p_plastic, f)
    img = Image.open(fp)
    res_p = predict(model_prod, img, device=device)
    res_v2 = predict(model_v2, img, device=device)
    pred_p = res_p['raw_predicted_class']
    pred_v2 = res_v2['raw_predicted_class']
    if pred_p != 'Plastic' or pred_v2 != 'Plastic':
        print(f"{f:22s} | Prod: {pred_p:15s} ({res_p['confidence']:.1%}) | V2: {pred_v2:15s} ({res_v2['confidence']:.1%})")

print("\n" + "="*80)
print("DIAGNOSING PAPER ERRORS (First 35 unseen Paper test images)")
print("="*80)
p_paper = os.path.join(ext_test_dir, 'paper')
pa_files = [f for f in os.listdir(p_paper) if f.lower().endswith(('.jpg','.jpeg','.png'))][:35]

for f in pa_files:
    fp = os.path.join(p_paper, f)
    img = Image.open(fp)
    res_p = predict(model_prod, img, device=device)
    res_v2 = predict(model_v2, img, device=device)
    pred_p = res_p['raw_predicted_class']
    pred_v2 = res_v2['raw_predicted_class']
    if pred_p != 'Paper' or pred_v2 != 'Paper':
        print(f"{f:22s} | Prod: {pred_p:15s} ({res_p['confidence']:.1%}) | V2: {pred_v2:15s} ({res_v2['confidence']:.1%})")

print("\n" + "="*80)
print("DIAGNOSING CARDBOARD ERRORS (First 35 unseen Cardboard test images)")
print("="*80)
p_cb = os.path.join(ext_test_dir, 'cardboard')
cb_files = [f for f in os.listdir(p_cb) if f.lower().endswith(('.jpg','.jpeg','.png'))][:35]

for f in cb_files:
    fp = os.path.join(p_cb, f)
    img = Image.open(fp)
    res_p = predict(model_prod, img, device=device)
    res_v2 = predict(model_v2, img, device=device)
    pred_p = res_p['raw_predicted_class']
    pred_v2 = res_v2['raw_predicted_class']
    if pred_p != 'Cardboard' or pred_v2 != 'Cardboard':
        print(f"{f:22s} | Prod: {pred_p:15s} ({res_p['confidence']:.1%}) | V2: {pred_v2:15s} ({res_v2['confidence']:.1%})")
