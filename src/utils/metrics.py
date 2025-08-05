import torch
import numpy as np

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
