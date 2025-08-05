# src/dataset/transform.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),

            # Color and illumination augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            #A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

            # Noise and blur
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ]) 

    elif split == "val":
        return A.Compose([
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])
    else:
        raise ValueError("split must be 'train' or 'val'")
