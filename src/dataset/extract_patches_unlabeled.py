# step1_extract_patches.py

import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------- CONFIG -----------------------
PATCH_SIZE = 256
STRIDE = 256
THRESHOLD = 210
MIN_FG_RATIO = 0.4


WSI_DIR = "/home/ubuntu/project/Tumor_Detection/data/Extra"
PATCH_OUTPUT_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/images"
# -----------------------------------------------------


def is_foreground_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    binary_mask = gray < THRESHOLD
    return np.sum(binary_mask) / (PATCH_SIZE * PATCH_SIZE) >= MIN_FG_RATIO


def extract_patches_from_wsi(image_path, out_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not read {image_path}")
        return
    base = os.path.splitext(os.path.basename(image_path))[0]
    h, w = img.shape[:2]

    for i in range(0, h - PATCH_SIZE + 1, STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            if is_foreground_patch(patch):
                fname = f"{base}_{i}_{j}.png"
                cv2.imwrite(os.path.join(out_dir, fname), patch)


def process_wsi_folder():
    os.makedirs(PATCH_OUTPUT_DIR, exist_ok=True)
    for fname in tqdm(os.listdir(WSI_DIR)):
        if fname.lower().endswith((".png", ".tif", ".jpg")):
            extract_patches_from_wsi(os.path.join(WSI_DIR, fname), PATCH_OUTPUT_DIR)


if __name__ == "__main__":
    process_wsi_folder()
