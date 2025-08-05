import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.transform import get_transforms
from utils.utils import rgb_to_label

class MixedPatchDataset(Dataset):
    def __init__(self, real_img_dir, real_mask_dir, pseudo_img_dir, pseudo_mask_dir, split="train", return_sample_type=False):
        self.real_img_dir = real_img_dir
        self.real_mask_dir = real_mask_dir
        self.pseudo_img_dir = pseudo_img_dir
        self.pseudo_mask_dir = pseudo_mask_dir
        self.return_sample_type = return_sample_type

        self.real_fnames = sorted([
            f for f in os.listdir(real_img_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])
        self.pseudo_fnames = sorted([
            f for f in os.listdir(pseudo_img_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.real_fnames) + len(self.pseudo_fnames)

    def __getitem__(self, idx):
        if idx < len(self.real_fnames):
            fname = self.real_fnames[idx]
            img_path = os.path.join(self.real_img_dir, fname)
            mask_path = os.path.join(self.real_mask_dir, fname)
            sample_type = 0  # 0 = real
        else:
            fname = self.pseudo_fnames[idx - len(self.real_fnames)]
            img_path = os.path.join(self.pseudo_img_dir, fname)
            mask_path = os.path.join(self.pseudo_mask_dir, fname)
            sample_type = 1  # 1 = pseudo

        image = cv2.imread(img_path)
        mask_rgb = cv2.imread(mask_path)

        if image is None or mask_rgb is None:
            raise RuntimeError(f"[!] Could not read image or mask for: {fname}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_to_label(mask_rgb).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.return_sample_type:
            return image, mask.long(), torch.tensor(sample_type)
        else:
            return image, mask.long()
