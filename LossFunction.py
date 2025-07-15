import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss for Data-Imbalanced NLP Tasks 2020.
        Article from: https://arxiv.org/abs/1911.02855

        :param smooth:
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for Dense Object Detection.
        Article from: https://doi.org/10.1109/TPAMI.2018.2858826

        Focal Loss for binary classification.

        Args:
            alpha (float): Weighting factor for positive class (0-1). Default: 0.25
            gamma (float): Focusing parameter for hard examples. Default: 2.0
            reduction (str): Reduction method ('mean', 'sum', or 'none'). Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute Focal Loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities (after sigmoid), shape (batch_size, 256, 256)
            targets (torch.Tensor): Ground truth binary mask, shape (batch_size, 256, 256)

        Returns:
            torch.Tensor: Focal Loss value
        """
        # Ensure inputs are probabilities (0-1)
        inputs = torch.clamp(inputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)
        # Compute binary cross-entropy loss (without reduction)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Compute probability for focal term
        pt = torch.where(targets == 1.0, inputs, 1.0 - inputs)
        focal_term = (1.0 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)

        # Compute focal loss
        loss = alpha_weight * focal_term * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
    def forward(self, inputs, targets):
        return (1 - self.dice_weight) * self.focal(inputs, targets) + self.dice_weight * self.dice(inputs, targets)
    
# Example usage
if __name__ == "__main__":
    # Simulate model outputs and labels
    batch_size = 2
    outputs = torch.sigmoid(torch.randn(batch_size, 256, 256))  # Simulated predictions (after sigmoid)
    labels = torch.randint(0, 2, (batch_size, 256, 256)).float()  # Simulated binary fire mask

    # Initialize FocalLoss
    # loss_function = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    loss_function = CombinedLoss()

    input = torch.tensor([[0., 0.6, 0.2],
                          [0.2, 0., 0.8],
                          [0., 0.4, 0.3]])
    target = torch.tensor([[0., 0., 0.],
                           [1., 0., 1.],
                           [0., 1., 1.]])
    # Compute loss
    # loss = loss_function(outputs, labels)
    loss = loss_function(input, target)
    print(f"Focal Loss: {loss.item()}")