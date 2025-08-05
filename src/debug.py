import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset import PatchDataset
from config import CONFIG

import torch 

train_ds = PatchDataset(CONFIG["train_img_dir"], CONFIG["train_mask_dir"], split='train')
val_ds = PatchDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], split='val')

def check_class_distribution(dataset):
    class_counts = np.zeros(4, dtype=int)
    for _, mask in dataset:
        unique, counts = np.unique(mask.numpy(), return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_counts[cls] += cnt
    return class_counts


classes = ['Background', 'Stroma', 'Benign', 'Tumor']

train_counts = check_class_distribution(train_ds)
val_counts = check_class_distribution(val_ds)


print("Train class distribution:", train_counts)
print("Val   class distribution:", val_counts)

plt.figure(figsize=(8, 4))
plt.bar(classes, train_counts, label='Train', alpha=0.7)
plt.bar(classes, val_counts, label='Val', alpha=0.7)
plt.title("Pixel Class Distribution")
plt.ylabel("Pixel Count")
plt.legend()
plt.tight_layout()
plt.savefig("class_distribution_sampler.png")


# class_counts = np.array([23933484, 34821418, 1982345, 4340001], dtype=np.float32)
# raw_weights = 1.0 / (class_counts + 1e-6)
# normalized_weights = raw_weights / raw_weights.sum()
# class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to('cuda')
# print (class_weights_tensor, class_weights_tensor.sum())