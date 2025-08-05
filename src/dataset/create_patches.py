import os
import cv2
import numpy as np
from tqdm import tqdm

PATCH_SIZE = 256
STRIDE = 256
THRESHOLD = 210        # Global threshold value for tissue
MIN_FG_RATIO = 0.1     # Minimum % of tissue required in a patch

def is_foreground_patch(patch, threshold=THRESHOLD, min_fg_ratio=MIN_FG_RATIO):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    binary_mask = gray < threshold
    fg_ratio = np.sum(binary_mask) / (PATCH_SIZE * PATCH_SIZE)
    return fg_ratio >= min_fg_ratio

def extract_patches(image_path, mask_path, out_img_dir, out_mask_dir):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    if img is None or mask is None:
        print(f"[!] Skipping: Could not load {image_path} or {mask_path}")
        return

    h, w, _ = img.shape
    assert mask.shape[:2] == (h, w), f"Mask and image shapes don't match for {image_path}"

    for i in range(0, h - PATCH_SIZE + 1, STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, STRIDE):
            img_patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

            if not is_foreground_patch(img_patch):
                continue

            fname = f"{os.path.basename(image_path).split('.')[0]}_{i}_{j}.png"
            cv2.imwrite(os.path.join(out_img_dir, fname), img_patch)
            cv2.imwrite(os.path.join(out_mask_dir, fname), mask_patch)

def process_all(data_dir, out_dir):
    img_dir = os.path.join(data_dir)
    mask_dir = os.path.join(data_dir)
    out_img_dir = os.path.join(out_dir, "images")
    out_mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for fname in tqdm(os.listdir(img_dir)):
        if "_mask" in fname or not fname.endswith(".png"):
            continue
        image_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".png", "_mask.png"))
        extract_patches(image_path, mask_path, out_img_dir, out_mask_dir)

if __name__ == "__main__":
    # Adjust paths if needed
    process_all("/home/ubuntu/project/Tumor_Detection/data/Training", "/home/ubuntu/project/Tumor_Detection/patch_with_1024/train")
    process_all("/home/ubuntu/project/Tumor_Detection/data/Validation", "/home/ubuntu/project/Tumor_Detection/patch_with_1024/val")
