# src/hard_mining/infer_and_select_hard.py

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from transform import get_transforms

# -------------------- CONFIG -----------------------
PATCH_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/images"
MODEL_PATH = "/home/ubuntu/project/Tumor_Detection/models/run_20250804-193621/best_model.pth"
HARD_PATCH_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/mined_img"
TOP_K = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------

# A. Load model
def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.eval()
    return model

# B. Inference on all patches with transform
def infer_and_collect(model, patch_dir):
    transform = get_transforms("val")
    scores = []

    os.makedirs(HARD_PATCH_DIR, exist_ok=True)

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
            softmax = torch.softmax(out, dim=1)[0]  # [C, H, W]
            tumor_prob = softmax[3].mean().item()
            benign_prob = softmax[2].mean().item()

        scores.append((fname, tumor_prob, benign_prob))

    return scores

# C. Select lowest tumor+benign confidence
def select_k_least_confident(scores, k=TOP_K):
    sorted_scores = sorted(scores, key=lambda x: x[1] + x[2])
    return sorted_scores[:k]

# D. Copy selected hard patches to target folder
def save_selected_patches(patch_list, src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname, _, _ in tqdm(patch_list, desc="💾 Saving hard patches"):
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
    all_scores = infer_and_collect(model, PATCH_DIR)

    print(f"\n🔍 Selecting {TOP_K} hardest patches (lowest tumor + benign confidence)")
    hard_samples = select_k_least_confident(all_scores, k=TOP_K)

    print(f"\n💾 Saving selected hard patches to: {HARD_PATCH_DIR}")
    save_selected_patches(hard_samples, PATCH_DIR, HARD_PATCH_DIR)

    print("\n✅ Done. You can now generate pseudo-labels or mix these into your training.")

if __name__ == "__main__":
    run_pipeline()
