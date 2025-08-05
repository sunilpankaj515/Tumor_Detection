import os
import torch
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter

from dataset.dataset import PatchDataset
from model.model import get_model
from config import CONFIG
from utils.metrics import compute_iou
from loss.focal import FocalLoss
from loss.hybrid import HybridLoss, HybridLoss_FG_BG
from loss.HybridLossLovasz import HybridLossLovasz

from sklearn.utils import resample
from torch.utils.data import Subset
from collections import Counter

def create_balanced_subset(dataset, labels, total_per_class=None):
    """
    Creates a balanced subset of the dataset using sklearn's resample.
    Args:
        dataset: PyTorch dataset.
        labels: List of integer class labels (0 or 1).
        total_per_class: Number of samples to use from each class (default=min class size).
    Returns:
        A torch.utils.data.Subset object with balanced samples.
    """
    # Convert labels to numpy
    labels = np.array(labels)

    # Indices for each class
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]

    # Determine size
    if total_per_class is None:
        total_per_class = max(len(idx_0), len(idx_1))

    # Resample both classes
    sampled_0 = resample(idx_0, replace=True, n_samples=total_per_class, random_state=42)
    sampled_1 = resample(idx_1, replace=True, n_samples=total_per_class, random_state=42)

    # Combine and shuffle
    all_indices = np.concatenate([sampled_0, sampled_1])
    np.random.shuffle(all_indices)

    return Subset(dataset, all_indices)

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_ious = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            ious = compute_iou(out, y)
            all_ious.append(ious)

    avg_loss = total_loss / len(loader)
    avg_ious = np.nanmean(np.array(all_ious), axis=0)
    return avg_loss, avg_ious

def train_model():
    device = torch.device(CONFIG["device"])
    model = get_model().to(device)

    # === Load dataset with presence-aware labels + caching ===
    
    train_ds = PatchDataset(CONFIG["train_img_dir"], CONFIG["train_mask_dir"], split='train', label_cache_path=CONFIG['label_cache'])
    val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')
    # === Compute sample weights for balancing ===
    patch_labels = train_ds.get_patch_labels()  # contains 0 or 1 now
    # class_counts = np.bincount(patch_labels, minlength=2)
    # class_weights = 1.0 / (class_counts + 1e-6)
    # sample_weights = [class_weights[label] for label in patch_labels]

    # print(f"Binary Patch Label Counts: {class_counts.tolist()}")
    # print(f"Binary Sample Weights: {class_weights.tolist()}")
    # print (len(sample_weights), Counter(sample_weights))
    
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )

    balanced_train_ds = create_balanced_subset(train_ds, patch_labels)

    train_loader = DataLoader(
        balanced_train_ds,
        batch_size=CONFIG["batch_size"],
        num_workers=4,
        pin_memory=True,
        shuffle = True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # === Pixel-level class weights (optional for loss) ===
    #class_counts_pix = np.array([23933484, 34821418, 1982345, 4340001], dtype=np.float32)  #256x256 , no stride
    class_counts_pix = np.array([113364666, 42819817, 2462751, 4930622], dtype=np.float32) # 768x768 , 512 stride
   
    raw_weights = 1.0 / (class_counts_pix + 1e-6)
    normalized_weights = raw_weights / raw_weights.sum()
    class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)
    print("Pixel Class Weights:", class_weights_tensor)

    # === Loss function ===
    criterion = HybridLoss(
        num_classes=CONFIG["num_classes"],
        alpha=0.5,
        beta=0.5,
        weight=class_weights_tensor,
        use_focal=True,
        focal_gamma=2.0
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(CONFIG["log_dir"], f"run_{timestamp}")
    run_ckpt_dir = os.path.join(os.path.dirname(CONFIG["checkpoint_path"]), f"run_{timestamp}")
    os.makedirs(run_ckpt_dir, exist_ok=True)

    writer = SummaryWriter(run_log_dir)

    best_val_loss = float("inf")
    patience = CONFIG["early_stopping_patience"]
    trials = 0

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0
        train_ious = []

        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CONFIG['num_epochs']}]"):
            #print (y.shape)
            #check whether samples are balanced
            # fg_patch_count = 0
            # for i in range(y.shape[0]):
            #     mask = y[i]
            #     counts = np.bincount(mask.flatten(), minlength=4)
            #     total = counts.sum()
            #     ratios = counts / (total + 1e-6)
            #     if ratios[2] > 0.1 or ratios[3] > 0.1:
            #         fg_patch_count += 1
            # print ("fg_patch_count ", fg_patch_count)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
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

        # === Save model checkpoint ===
        model_save_path = os.path.join(run_ckpt_dir, f"best_model_epoch{epoch+1}.pth")
        torch.save(model, model_save_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(run_ckpt_dir, "best_model.pth")
            torch.save(model, model_save_path)
            trials = 0
            print(f"✅ New best model saved to: {model_save_path}")
        else:
            trials += 1
            if trials >= patience:
                print("⏹️ Early stopping triggered.")
                break

    writer.close()


if __name__ == "__main__":
    train_model()



# import os
# import torch
# import numpy as np
# from datetime import datetime
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from dataset.dataset import PatchDataset
# from model.model import get_model
# from config import CONFIG
# from utils.metrics import compute_iou
# from loss.focal import FocalLoss
# from loss.hybrid import HybridLoss, HybridLoss_FG_BG
# from loss.HybridLossLovasz import HybridLossLovasz

# def evaluate_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_ious = []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             out = model(x)
#             loss = criterion(out, y)
#             total_loss += loss.item()
#             ious = compute_iou(out, y)
#             all_ious.append(ious)

#     avg_loss = total_loss / len(loader)
#     avg_ious = np.nanmean(np.array(all_ious), axis=0)
#     return avg_loss, avg_ious

# def train_model():
#     device = torch.device(CONFIG["device"])
#     model = get_model().to(device)

#     train_ds = PatchDataset(CONFIG["train_img_dir"], CONFIG["train_mask_dir"], split='train')
#     val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')

#     train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

#     # === Class weights based on pixel frequency ===
#     class_counts = np.array([23933484, 34821418, 1982345, 4340001], dtype=np.float32) #for 256x256
#     raw_weights = 1.0 / (class_counts + 1e-6)
#     normalized_weights = raw_weights / raw_weights.sum()
#     class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)
#     print("Class Weights:", class_weights_tensor)

    
#     #criterion = HybridLoss(num_classes=CONFIG["num_classes"], weight=class_weights_tensor)
#     criterion = HybridLoss(
#                 num_classes=CONFIG["num_classes"],
#                 alpha=0.5,
#                 beta=0.5,
#                 weight=class_weights_tensor,
#                 use_focal=True,           # enables focal loss
#                 focal_gamma=2.0           # tuning parameter
#             )

#     # criterion = HybridLoss_FG_BG(
#     #             alpha=0.6,          # CE/Focal contribution
#     #             beta=0.1,          # Dice for background, stroma classes [0, 1]
#     #             gamma=0.4,         # Dice for benign, tumor classes [2, 3]
#     #             weight=class_weights_tensor, 
#     #             use_focal=True,     # Focal loss instead of vanilla CE
#     #             focal_gamma=2.0     # Default gamma for Focal loss
#     #         )
    
#     #criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
#     #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

#     optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

#     # === Timestamped run folder for logs and checkpoints ===
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     run_log_dir = os.path.join(CONFIG["log_dir"], f"run_{timestamp}")
#     run_ckpt_dir = os.path.join(os.path.dirname(CONFIG["checkpoint_path"]), f"run_{timestamp}")
#     os.makedirs(run_ckpt_dir, exist_ok=True)

#     writer = SummaryWriter(run_log_dir)

#     best_val_loss = float("inf")
#     patience = CONFIG["early_stopping_patience"]
#     trials = 0

#     for epoch in range(CONFIG["num_epochs"]):
#         model.train()
#         train_loss = 0
#         train_ious = []

#         for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CONFIG['num_epochs']}]"):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             out = model(x)
#             loss = criterion(out, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             ious = compute_iou(out, y)
#             train_ious.append(ious)

#         scheduler.step()
#         val_loss, val_ious = evaluate_epoch(model, val_loader, criterion, device)
#         avg_train_loss = train_loss / len(train_loader)
#         avg_train_ious = np.nanmean(np.array(train_ious), axis=0)

#         # === TensorBoard Logging ===
#         writer.add_scalars("Loss", {"Train": avg_train_loss, "Val": val_loss}, epoch)
#         writer.add_scalars("IoU/Background", {"Train": avg_train_ious[0], "Val": val_ious[0]}, epoch)
#         writer.add_scalars("IoU/Stroma", {"Train": avg_train_ious[1], "Val": val_ious[1]}, epoch)
#         writer.add_scalars("IoU/Benign", {"Train": avg_train_ious[2], "Val": val_ious[2]}, epoch)
#         writer.add_scalars("IoU/Tumor", {"Train": avg_train_ious[3], "Val": val_ious[3]}, epoch)

#         print(f"\n📊 Epoch {epoch+1}:")
#         print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
#         print(f"Train IoUs: {[round(i, 4) for i in avg_train_ious]}")
#         print(f"Val   IoUs: {[round(i, 4) for i in val_ious]}")
        
#         # === Save best model (entire model) ===
#         model_save_path = os.path.join(run_ckpt_dir, f"best_model_epoch{epoch+1}.pth")
#         torch.save(model, model_save_path)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model_save_path = os.path.join(run_ckpt_dir, "best_model.pth")
#             torch.save(model, model_save_path)
#             trials = 0
#             print(f"✅ New best model saved to: {model_save_path}")
#         else:
#             trials += 1
#             if trials >= patience:
#                 print("⏹️ Early stopping triggered.")
#                 break

#     writer.close()


# if __name__ == "__main__":
#     train_model()
