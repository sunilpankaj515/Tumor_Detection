# src/hard_mining/soft_mining_by_threshold.py

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transform import get_transforms

# -------------------- CONFIG -----------------------
PATCH_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/images"
MODEL_PATH = "/home/ubuntu/project/Tumor_Detection/models/run_20250803-082615/best_model_epoch27.pth"
SOFT_PATCH_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/soft_mined_img"
CONFIDENCE_MIN = 0.3     # lower bound for (tumor + benign)
CONFIDENCE_MAX = 0.8     # upper bound for (tumor + benign)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------

# A. Load model
def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.eval()
    return model

# B. Inference + filtering based on confidence
def infer_and_filter(model, patch_dir):
    transform = get_transforms("val")
    selected = []

    os.makedirs(SOFT_PATCH_DIR, exist_ok=True)

    for fname in tqdm(sorted(os.listdir(patch_dir)), desc="🔍 Inference"):
        fpath = os.path.join(patch_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=img)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(img_tensor)  # [1, C, H, W]
            softmax = torch.softmax(out, dim=1)[0]
            tumor_prob = softmax[3].mean().item()
            benign_prob = softmax[2].mean().item()

        total_conf = tumor_prob + benign_prob

        # C. Select if confidence falls between threshold bounds
        if CONFIDENCE_MIN <= total_conf <= CONFIDENCE_MAX:
            selected.append((fname, total_conf))

    return selected

# D. Save selected patches
def save_selected_patches(patch_list, src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname, _ in tqdm(patch_list, desc="💾 Saving soft patches"):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(out_dir, fname)
        img = cv2.imread(src)
        if img is not None:
            cv2.imwrite(dst, img)

# 🔁 Runner
def run_pipeline():
    print(f"\n📦 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"\n🧠 Running inference on patches in: {PATCH_DIR}")
    soft_samples = infer_and_filter(model, PATCH_DIR)

    print(f"\n🔍 Found {len(soft_samples)} patches in confidence range [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}]")
    save_selected_patches(soft_samples, PATCH_DIR, SOFT_PATCH_DIR)

    print("\n✅ Soft mining complete. You may now generate pseudo-labels.")

if __name__ == "__main__":
    run_pipeline()
