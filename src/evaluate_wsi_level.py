import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

from utils.utils import rgb_to_label, label_to_rgb
from dataset.transform import get_transforms
from config import CONFIG
# -------------------------
# Config
# -------------------------
PATCH_SIZE = 256
STRIDE = 256
THRESHOLD = 210
MIN_FG_RATIO = 0.1
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = CONFIG['Validation_path']
MODEL_PATH = CONFIG['best_checkpoint_path']
SAVE_DIR = "/home/ubuntu/project/Tumor_Detection/data/pred_mask"
os.makedirs(SAVE_DIR, exist_ok=True)

class_names = ['Background', 'Stroma', 'Benign', 'Tumor']

# -------------------------
# Shared Compute IoU
# -------------------------
# def compute_iou(pred, target, num_classes=4):
#     ious = []
#     pred = torch.argmax(pred, dim=1)
#     for cls in range(num_classes):
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = (pred_inds & target_inds).sum().item()
#         union = (pred_inds | target_inds).sum().item()
#         if union == 0:
#             ious.append(float('nan'))
#         else:
#             ious.append(intersection / union)
#     return ious

def compute_iou(pred, target, num_classes=4):
    """
    Computes mean IoU per class across each sample in the batch, then averages across the batch.
    Args:
        pred: Raw logits from model (B, C, H, W)
        target: Ground truth masks (B, H, W)
        num_classes: Total number of classes
    Returns:
        List of class-wise mean IoUs (length = num_classes)
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    B = pred.shape[0]
    class_ious = [[] for _ in range(num_classes)]

    for i in range(B):
        for cls in range(num_classes):
            pred_inds = (pred[i] == cls)
            target_inds = (target[i] == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                class_ious[cls].append(float('nan'))
            else:
                class_ious[cls].append(intersection / union)

    # Compute mean IoU per class across batch
    mean_ious = [np.nanmean(class_ious[cls]) for cls in range(num_classes)]
    return mean_ious

def batch_evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_ious = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            for i in range(x.size(0)):
                sample_iou = compute_iou(out[i].unsqueeze(0), y[i].unsqueeze(0))
                all_ious.append(sample_iou)

    avg_loss = total_loss / len(loader)
    avg_ious = np.nanmean(np.array(all_ious), axis=0)
    return avg_loss, avg_ious


# -------------------------
# Foreground Patch Filter
# -------------------------
def is_foreground_patch(patch, threshold=THRESHOLD, min_fg_ratio=MIN_FG_RATIO):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    binary_mask = gray < threshold
    fg_ratio = np.sum(binary_mask) / (PATCH_SIZE * PATCH_SIZE)
    return fg_ratio >= min_fg_ratio

# -------------------------
# Load Model
# -------------------------
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)

# -------------------------
# Evaluation Loop
# -------------------------
transform = get_transforms("val")
all_patch_ious = [[] for _ in range(NUM_CLASSES)]
all_wsi_ious = [[] for _ in range(NUM_CLASSES)]

image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".png") and "_mask" not in f])

for fname in tqdm(image_files):
    img_path = os.path.join(DATA_DIR, fname)
    mask_path = os.path.join(DATA_DIR, fname.replace(".png", "_mask.png"))

    image = cv2.imread(img_path)
    mask_rgb = cv2.imread(mask_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    gt_mask = rgb_to_label(mask_rgb)

    H, W, _ = image.shape
    pred_full = np.zeros((H, W), dtype=np.uint8)

    # ---- Patch-wise prediction ----
    for i in range(0, H - PATCH_SIZE + 1, STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, STRIDE):
            img_patch = image[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch = gt_mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

            if not is_foreground_patch(img_patch):
                continue

            transformed = transform(image=img_patch, mask=mask_patch)
            img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
            mask_tensor = transformed['mask'].to(DEVICE)

            with torch.no_grad():
                logits = model(img_tensor)

            # ✅ Patch-level class-wise IoU
            patch_ious = compute_iou(logits, mask_tensor.unsqueeze(0), num_classes=NUM_CLASSES)
            for cls in range(NUM_CLASSES):
                all_patch_ious[cls].append(patch_ious[cls])

            # Stitch prediction
            preds = torch.argmax(logits, dim=1)[0].cpu().numpy()
            pred_full[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = preds

    # ---- WSI-level IoU ----
    pred_tensor = torch.from_numpy(pred_full).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
    with torch.no_grad():
        wsi_logits = torch.nn.functional.one_hot(pred_tensor.long(), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        wsi_ious = compute_iou(wsi_logits, gt_tensor, num_classes=NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        all_wsi_ious[cls].append(wsi_ious[cls])

    # ---- Save predicted mask ----
    pred_rgb = label_to_rgb(pred_full)
    pred_path = os.path.join(SAVE_DIR, fname.replace(".png", "_pred.png"))
    cv2.imwrite(pred_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

    # ---- Save overlap visualization ----
    overlap = np.zeros((H, W, 3), dtype=np.uint8)
    for cls in range(1, NUM_CLASSES):  # Skip background for visual clarity
        pred_mask = (pred_full == cls)
        gt_mask_c = (gt_mask == cls)

        # TP: green
        overlap[np.logical_and(pred_mask, gt_mask_c)] = [0, 255, 0]
        # FN: red
        overlap[np.logical_and(~pred_mask, gt_mask_c)] = [255, 0, 0]
        # FP: blue
        overlap[np.logical_and(pred_mask, ~gt_mask_c)] = [0, 0, 255]

    overlap_path = os.path.join(SAVE_DIR, fname.replace(".png", "_overlap.png"))
    cv2.imwrite(overlap_path, cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR))

# -------------------------
# Final Reporting
# -------------------------
print("\n✅ Average Patch-Level IoU (across all images):")
for cls in range(NUM_CLASSES):
    ious = np.array(all_patch_ious[cls])
    if len(ious) == 0 or np.all(np.isnan(ious)):
        print(f"  {class_names[cls]}: IoU = N/A")
    else:
        print(f"  {class_names[cls]}: IoU = {np.nanmean(ious):.4f}")

print("\n✅ Average WSI-Level IoU (across all images):")
for cls in range(NUM_CLASSES):
    ious = np.array(all_wsi_ious[cls])
    if len(ious) == 0 or np.all(np.isnan(ious)):
        print(f"  {class_names[cls]}: IoU = N/A")
    else:
        print(f"  {class_names[cls]}: IoU = {np.nanmean(ious):.4f}")
