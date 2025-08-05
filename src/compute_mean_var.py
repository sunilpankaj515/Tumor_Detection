import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import PatchDataset
from tqdm import tqdm
import numpy as np
from config import CONFIG

def compute_mean_std(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in tqdm(loader, desc="Computing mean and std"):
        # Images are in [B, C, H, W], convert to [B * H * W, C]
        batch_samples = images.size(0)
        pixels = batch_samples * images.size(2) * images.size(3)

        mean += images.sum(dim=[0, 2, 3])
        std += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += pixels

    mean /= total_pixels
    std = (std / total_pixels - mean ** 2).sqrt()

    return mean.numpy(), std.numpy()

if __name__ == "__main__":
    # Note: Ensure ToTensor is applied but NOT Normalize
    from torchvision import transforms as T
    from dataset.transform import get_transforms

    # Custom transform with only ToTensor (no normalization)
    def basic_transform():
        return T.Compose([T.ToPILImage(), T.ToTensor()])

    train_ds = PatchDataset(CONFIG["train_img_dir"], CONFIG["train_mask_dir"], split='train')
    val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')
    
    # Temporarily override transforms
    train_ds.transforms = basic_transform()
    val_ds.transforms = basic_transform()

    print("📊 Computing stats on Train Set")
    train_mean, train_std = compute_mean_std(train_ds)
    print(f"Train Mean: {train_mean}, Std: {train_std}")

    print("📊 Computing stats on Val Set")
    val_mean, val_std = compute_mean_std(val_ds)
    print(f"Val Mean: {val_mean}, Std: {val_std}")
