import os
import torch
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.mixed_dataset import MixedPatchDataset
from dataset.dataset import PatchDataset
from model.model import get_model
from config import CONFIG
from utils.metrics import compute_iou
from loss.hybrid import HybridLoss

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = True

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_ious = []

    with torch.no_grad():
        for x, y in loader:  # no third output here
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            ious = compute_iou(out, y)
            all_ious.append(ious)

    avg_loss = total_loss / len(loader)
    avg_ious = np.nanmean(np.array(all_ious), axis=0)
    return avg_loss, avg_ious


def finetune_with_pseudo():
    device = torch.device(CONFIG["device"])
    model = torch.load(CONFIG["best_checkpoint_path"], map_location=device, weights_only=False)
    model.to(device)

    print("🔁 Fine-tuning starting from:", CONFIG["best_checkpoint_path"])

    train_ds = MixedPatchDataset(
        real_img_dir=CONFIG["train_img_dir"],
        real_mask_dir=CONFIG["train_mask_dir"],
        pseudo_img_dir=CONFIG["pseudo_img_dir"],
        pseudo_mask_dir=CONFIG["pseudo_mask_dir"],
        split='train',
        return_sample_type=True
    )
    val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

    class_counts = np.array([23933484, 34821418, 1982345, 4340001], dtype=np.float32)
    raw_weights = 1.0 / (class_counts + 1e-6)
    normalized_weights = raw_weights / raw_weights.sum()
    class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)

    criterion = HybridLoss(
        num_classes=CONFIG["num_classes"],
        alpha=0.5,
        beta=0.5,
        weight=class_weights_tensor,
        use_focal=True,
        focal_gamma=2.0
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(CONFIG["log_dir"], f"finetune_{timestamp}")
    run_ckpt_dir = os.path.join(os.path.dirname(CONFIG["checkpoint_path"]), f"finetune_{timestamp}")
    os.makedirs(run_ckpt_dir, exist_ok=True)

    writer = SummaryWriter(run_log_dir)
    best_val_loss = float("inf")
    patience = CONFIG["early_stopping_patience"]
    trials = 0

    for epoch in range(CONFIG["num_epochs"]):
        # if epoch == 0:
        #     print("🧊 Freezing backbone...")
        #     freeze_backbone(model)
        # elif epoch == 5:
        #     print(" Unfreezing backbone...")
        #     unfreeze_backbone(model)

        model.train()
        train_loss = 0
        train_ious = []

        for x, y, sample_types in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CONFIG['num_epochs']}]"):
            x, y, sample_types = x.to(device), y.to(device), sample_types.to(device)
            optimizer.zero_grad()
            out = model(x)

            real_idx = (sample_types == 0)
            pseudo_idx = (sample_types == 1)
            loss = torch.tensor(0.0, device=device)

            if real_idx.any():
                real_loss = criterion(out[real_idx], y[real_idx])
                loss += 1.0 * real_loss

            if pseudo_idx.any():
                pseudo_loss = criterion(out[pseudo_idx], y[pseudo_idx])
                loss += 0.3 * pseudo_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            ious = compute_iou(out, y)
            train_ious.append(ious)

        scheduler.step()
        val_loss, val_ious = evaluate_epoch(model, val_loader, criterion, device)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ious = np.nanmean(np.array(train_ious), axis=0)

        writer.add_scalars("Loss", {"Train": avg_train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("IoU/Background", {"Train": avg_train_ious[0], "Val": val_ious[0]}, epoch)
        writer.add_scalars("IoU/Stroma", {"Train": avg_train_ious[1], "Val": val_ious[1]}, epoch)
        writer.add_scalars("IoU/Benign", {"Train": avg_train_ious[2], "Val": val_ious[2]}, epoch)
        writer.add_scalars("IoU/Tumor", {"Train": avg_train_ious[3], "Val": val_ious[3]}, epoch)

        print(f"\n📊 Epoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train IoUs: {[round(i, 4) for i in avg_train_ious]}")
        print(f"Val   IoUs: {[round(i, 4) for i in val_ious]}")

        model_save_path = os.path.join(run_ckpt_dir, f"best_model_epoch{epoch+1}.pth")
        torch.save(model, model_save_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(run_ckpt_dir, "best_model.pth"))
            trials = 0
            print(f"✅ New best model saved to: {os.path.join(run_ckpt_dir, 'best_model.pth')}")
        else:
            trials += 1
            if trials >= patience:
                print("⏹️ Early stopping triggered.")
                break

    writer.close()

if __name__ == "__main__":
    finetune_with_pseudo()
