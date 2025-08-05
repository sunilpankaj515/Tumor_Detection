import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovasz-Softmax loss for multi-class segmentation.
    Reference: https://arxiv.org/abs/1705.08790
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] — raw outputs from model
        targets: [B, H, W] — class indices
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        loss = 0.0
        for c in range(probs.shape[1]):
            fg = (targets == c).float()  # foreground mask for class c
            if fg.sum() == 0:
                continue  # skip absent classes
            pred = probs[:, c, :, :]
            loss += self.lovasz_hinge_flat(pred, fg)
        return loss / probs.shape[1]

    def lovasz_hinge_flat(self, logits, labels):
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
        gt_sorted = labels.view(-1)[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gtsum = torch.sum(gt_sorted)
        intersect = gtsum - gt_sorted.cumsum(0)
        union = gtsum + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersect / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * logpt
        if self.weight is not None:
            weight = self.weight[targets]
            loss *= weight
        return loss.mean()


class HybridLossLovasz(nn.Module):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3, weight=None, use_focal=True, focal_gamma=2.0):
        """
        alpha: CE or Focal loss
        beta: Lovasz-Softmax loss
        gamma: optional Dice/other loss (set to 0.0 if unused)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.lovasz = LovaszSoftmaxLoss()
        self.use_focal = use_focal

        if use_focal:
            self.ce_or_focal = FocalLoss(weight=weight, gamma=focal_gamma)
        else:
            self.ce_or_focal = nn.CrossEntropyLoss(weight=weight)

        # Optional: you can plug in DiceLoss here if gamma > 0
        self.dice = None

    def forward(self, logits, targets):
        loss_ce = self.ce_or_focal(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)

        if self.gamma > 0 and self.dice is not None:
            loss_dice = self.dice(logits, targets)
        else:
            loss_dice = 0.0

        return self.alpha * loss_ce + self.beta * loss_lovasz + self.gamma * loss_dice
