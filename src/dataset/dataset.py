import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.transform import get_transforms
from utils.utils import rgb_to_label
from tqdm import tqdm
import pickle, csv

class PatchDataset(Dataset):
    def __init__(self, img_dir, mask_dir, split="train", label_cache_path=None):
        """
        Args:
            img_dir (str): Path to image patches.
            mask_dir (str): Path to label masks (RGB format).
            split (str): 'train' or 'val'.
            label_cache_path (str): Optional path to store/load patch labels.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.transform = get_transforms(split)
        self.label_cache_path = label_cache_path

        if label_cache_path and os.path.exists(label_cache_path):
            print(f"Loading cached patch labels from: {label_cache_path}")
            with open(label_cache_path, "rb") as f:
                self.labels = pickle.load(f)
        else:
            print("Computing patch-level labels...")
            self.labels = self._compute_patch_labels()
            if label_cache_path:
                print(f"Saving patch labels to: {label_cache_path}")
                with open(label_cache_path, "wb") as f:
                    pickle.dump(self.labels, f)

    def _compute_patch_labels(self):
        """Assign a class to each patch using presence-aware logic."""
        labels = []

        # with open("ratio_out.csv", mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["patch_name", "background_ratio", "stroma_ratio", "benign_ratio", "tumor_ratio"])

        for fname in tqdm(self.imgs, desc="Labeling patches"):
            mask_path = os.path.join(self.mask_dir, fname)
            mask_rgb = cv2.imread(mask_path)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
            mask = rgb_to_label(mask_rgb).astype(np.uint8)

            counts = np.bincount(mask.flatten(), minlength=4)
            total = counts.sum()
            ratios = counts / (total + 1e-6)
            #writer.writerow([fname] + [round(r, 6) for r in ratios])
            #print ("data _count ", ratios)

            if ratios[2] > 0.10 or ratios[3] > 0.10:
                label = 1  # Foreground
            else:
                label = 0  # Background-Stroma

            labels.append(label)
        return labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.imgs[idx])

        image = cv2.imread(img_path)
        mask_rgb = cv2.imread(mask_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_to_label(mask_rgb).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.long()

    def get_patch_labels(self):
        return self.labels



# import os
# import cv2
# import torch
# import numpy as np
# from torch.utils.data import Dataset

# from dataset.transform import get_transforms
# from utils.utils import rgb_to_label

# class PatchDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, split="train"):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.imgs = sorted(os.listdir(img_dir))
#         self.transform = get_transforms(split)

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.imgs[idx])
#         mask_path = os.path.join(self.mask_dir, self.imgs[idx])

#         image = cv2.imread(img_path)
#         mask_rgb = cv2.imread(mask_path)

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

#         # Convert mask to label before applying joint transforms
#         mask = rgb_to_label(mask_rgb).astype(np.uint8)

#         if self.transform:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed['image']
#             mask = transformed['mask']

#         # Convert mask to tensor
#         mask = mask.long()

#         return image, mask


