import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import CONFIG
from dataset.dataset import PatchDataset
from utils.metrics import compute_iou
from utils.utils import label_to_rgb

# def evaluate_batch(model_path="best_model.pth"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.load(model_path,weights_only=False, map_location=device)
#     model.eval()

#     val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')
#     val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

#     all_ious = []

#     with torch.no_grad():
#         for x, y in val_loader:
#             x, y = x.to(device), y.to(device)
#             out = model(x)
#             ious = compute_iou(out, y)
#             all_ious.append(ious)

#     all_ious = np.array(all_ious)
#     mean_ious = np.nanmean(all_ious, axis=0)
#     print("Per-class IoU:", mean_ious)


def evaluate(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

    num_classes = 4
    all_ious = [[] for _ in range(num_classes)]  # list of lists: class-wise IoUs

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            for i in range(x.size(0)):  # loop over each patch
                pred_patch = out[i].unsqueeze(0)
                gt_patch = y[i].unsqueeze(0)

                patch_ious = compute_iou(pred_patch, gt_patch, num_classes=num_classes)
                for cls in range(num_classes):
                    all_ious[cls].append(patch_ious[cls])

    # Report mean per-class IoU
    print("\n✅ Patch-Level Per-Class IoU (from PatchDataset):")
    class_names = ['Background', 'Stroma', 'Benign', 'Tumor']
    for cls in range(num_classes):
        ious = np.array(all_ious[cls])
        if len(ious) == 0 or np.all(np.isnan(ious)):
            print(f"  {class_names[cls]}: IoU = N/A")
        else:
            print(f"  {class_names[cls]}: IoU = {np.nanmean(ious):.4f}")



def visualize_predictions(model_path="best_model.pth", num_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    os.makedirs("misc", exist_ok=True)
    for i in range(len(images)):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(images[i])
        axs[0].set_title("Original Patch")
        axs[0].axis('off')

        axs[1].imshow(label_to_rgb(masks[i]))
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')

        axs[2].imshow(label_to_rgb(preds[i]))
        axs[2].set_title("Predicted Mask")
        axs[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join("misc", f"prediction_{i+1}.png")
        plt.savefig(save_path)
        plt.close(fig)

# === Optional usage ===
if __name__ == "__main__":
    model_path =  CONFIG['best_checkpoint_path']
    evaluate(model_path)
    #visualize_predictions(model_path)
