import torch
from torch import nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape (C,) for class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W)
        targets: (N, H, W) — integer class labels
        """
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        # Get log_probs for true class only: shape (N, H, W)
        log_probs_true = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probs_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal loss calculation
        loss = -(1 - probs_true) ** self.gamma * log_probs_true

        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # shape (N, H, W)
            loss *= alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
