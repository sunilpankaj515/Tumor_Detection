import os
import cv2
import numpy as np
import random
from tqdm import tqdm

PATCH_SIZE = 256
STRIDE = 256
THRESHOLD = 210
MIN_FG_RATIO = 0.1
OVERSAMPLE_FACTOR = 2  # how many extra patches to extract if tumor/benign is found
PERTURBATION = 128      # max pixel shift for jittering

PALETTE = {
    (0, 0, 0): 0,          # Background - Black
    (0, 0, 255): 1,        # Stroma - Blue
    (0, 255, 0): 2,        # Benign - Green
    (255, 255, 0): 3       # Tumor - Yellow
}

def rgb_to_label(mask_rgb):
    """
    Convert a 3-channel RGB mask into a 2D mask of class indices.
    """
    h, w, _ = mask_rgb.shape
    mask_label = np.zeros((h, w), dtype=np.uint8)

    for color, cls_id in PALETTE.items():
        matches = np.all(mask_rgb == np.array(color), axis=-1)
        mask_label[matches] = cls_id

    return mask_label


def is_foreground_patch(patch, threshold=THRESHOLD, min_fg_ratio=MIN_FG_RATIO):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    binary_mask = gray < threshold
    fg_ratio = np.sum(binary_mask) / (PATCH_SIZE * PATCH_SIZE)
    return fg_ratio >= min_fg_ratio

def contains_foreground_class(mask_patch_label, target_classes=[2, 3]):
    unique = np.unique(mask_patch_label)
    return any(cls in unique for cls in target_classes)

def extract_patches(image_path, mask_path, out_img_dir, out_mask_dir):
    img = cv2.imread(image_path)
    mask_rgb = cv2.imread(mask_path)
    if img is None or mask_rgb is None:
        print(f"[!] Skipping: Cannot read {image_path} or {mask_path}")
        return

    h, w, _ = img.shape
    mask_label = rgb_to_label(mask_rgb)

    base_name = os.path.basename(image_path).replace(".png", "")
    patch_count = 0

    # Grid-based patching (with stride)
    for i in range(0, h - PATCH_SIZE + 1, STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, STRIDE):
            img_patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch_rgb = mask_rgb[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch_label = mask_label[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

            if not is_foreground_patch(img_patch):
                continue

            save_name = f"{base_name}_{i}_{j}.png"
            cv2.imwrite(os.path.join(out_img_dir, save_name), img_patch)
            cv2.imwrite(os.path.join(out_mask_dir, save_name), mask_patch_rgb)
            patch_count += 1

            # If tumor or benign present, oversample with jitter
            if contains_foreground_class(mask_patch_label):
                for _ in range(OVERSAMPLE_FACTOR):
                    ii = np.clip(i + random.randint(-PERTURBATION, PERTURBATION), 0, h - PATCH_SIZE)
                    jj = np.clip(j + random.randint(-PERTURBATION, PERTURBATION), 0, w - PATCH_SIZE)

                    jittered_img_patch = img[ii:ii+PATCH_SIZE, jj:jj+PATCH_SIZE]
                    jittered_mask_patch = mask_rgb[ii:ii+PATCH_SIZE, jj:jj+PATCH_SIZE]

                    if is_foreground_patch(jittered_img_patch):
                        jitter_name = f"{base_name}_{ii}_{jj}_j.png"
                        cv2.imwrite(os.path.join(out_img_dir, jitter_name), jittered_img_patch)
                        cv2.imwrite(os.path.join(out_mask_dir, jitter_name), jittered_mask_patch)

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
    process_all(
        "/home/ubuntu/project/Tumor_Detection/data/Training",
        "/home/ubuntu/project/Tumor_Detection/patch_with_sampler/train"
    )
    # process_all(
    #     "/home/ubuntu/project/Tumor_Detection/data/Validation",
    #     "/home/ubuntu/project/Tumor_Detection/patch_with_sampler/val"
    # )
