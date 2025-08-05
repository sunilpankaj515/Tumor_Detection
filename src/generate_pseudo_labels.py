import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dataset.transform import get_transforms
from config import CONFIG

# -------------------- CONFIG -----------------------
IMG_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/mined_img"
OUT_MASK_DIR = "/home/ubuntu/project/Tumor_Detection/patches/pseudo/mined_masks"
MODEL_PATH = CONFIG['best_checkpoint_path']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------

# ✅ Correct class-to-RGB color map
color_map = np.array([
    [0, 0, 0],        # Background (Black)
    [0, 0, 255],      # Stroma (Blue)
    [0, 255, 0],      # Benign (Green)
    [255, 255, 0],    # Tumor (Yellow)
], dtype=np.uint8)

# Load model
def load_model(path):
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.eval()
    return model

# Predict and save masks
def generate_pseudo_labels(model, img_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    transform = get_transforms(split="val")

    for fname in tqdm(sorted(os.listdir(img_dir)), desc="Generating pseudo-labels"):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=img)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            preds = torch.argmax(output, dim=1)[0].cpu().numpy().astype(np.uint8)

        # Convert label to RGB mask using correct color map
        color_mask = color_map[preds]
        save_path = os.path.join(mask_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

# Main runner
if __name__ == "__main__":
    print(f"📦 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"🧠 Generating pseudo-labels for images in: {IMG_DIR}")
    generate_pseudo_labels(model, IMG_DIR, OUT_MASK_DIR)

    print(f"✅ Saved pseudo-labels to: {OUT_MASK_DIR}")
