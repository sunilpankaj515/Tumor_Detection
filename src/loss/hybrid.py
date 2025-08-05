# src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, num_classes):
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_loss = 1.0 - ((2. * intersection + self.smooth) / (cardinality + self.smooth))
        return dice_loss.mean()

class DiceLoss_fg_bg(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, class_indices):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        class_indices: list of class IDs to compute Dice over (e.g., [2, 3] for FG)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dice = 0.0
        for cls in class_indices:
            intersection = torch.sum(probs[:, cls] * targets_one_hot[:, cls])
            cardinality = torch.sum(probs[:, cls] + targets_one_hot[:, cls])
            dice_cls = 1.0 - ((2. * intersection + self.smooth) / (cardinality + self.smooth))
            dice += dice_cls

        return dice / len(class_indices)
    

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=None):
        """
        Multi-class Focal Loss based on CrossEntropy
        Args:
            weight: class weights (Tensor of shape [C])
            gamma: focusing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logpt = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * logpt

        if self.weight is not None:
            weight = self.weight[targets]  # class-wise weight
            loss = loss * weight

        return loss.mean()


class HybridLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, weight=None, use_focal=True, focal_gamma=2.0):
        """
        Args:
            alpha: weight for focal/CE loss
            beta: weight for dice loss
            use_focal: True to use Focal + Dice, False to use CE + Dice
        """
        super().__init__()
        self.dice = DiceLoss()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

        if use_focal:
            self.ce_or_focal = FocalLoss(weight=weight, gamma=focal_gamma)
        else:
            self.ce_or_focal = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_or_focal_loss = self.ce_or_focal(logits, targets)
        dice_loss = self.dice(logits, targets, self.num_classes)
        return self.alpha * ce_or_focal_loss + self.beta * dice_loss



class HybridLoss_FG_BG(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25, weight=None, use_focal=True, focal_gamma=2.0):
        """
        alpha: weight for CE/Focal loss
        beta: weight for background Dice loss (classes 0,1)
        gamma: weight for foreground Dice loss (classes 2,3)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dice = DiceLoss_fg_bg()
        self.fg_classes = [2, 3]
        self.bg_classes = [0, 1]

        if use_focal:
            self.ce_or_focal = FocalLoss(weight=weight, gamma=focal_gamma)
        else:
            self.ce_or_focal = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_focal_loss = self.ce_or_focal(logits, targets)
        dice_bg = self.dice(logits, targets, self.bg_classes)
        dice_fg = self.dice(logits, targets, self.fg_classes)
        return self.alpha * ce_focal_loss + self.beta * dice_bg + self.gamma * dice_fg
    