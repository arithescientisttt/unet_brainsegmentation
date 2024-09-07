import torch.nn as nn

class DiceCoefficientLoss(nn.Module):

    def __init__(self):
        super(DiceCoefficientLoss, self).__init__()
        # Add a smoothing term to avoid division by zero
        self.epsilon = 1.0

    def forward(self, predictions, targets):
        # Ensure the predicted and target tensors have the same shape
        assert predictions.size() == targets.size()

        # Flatten the tensors to 1D vectors for the calculation
        predictions_flat = predictions[:, 0].contiguous().view(-1)
        targets_flat = targets[:, 0].contiguous().view(-1)

        # Compute the intersection between predictions and true labels
        intersection = (predictions_flat * targets_flat).sum()

        # Compute the Dice coefficient
        dice_score = (2. * intersection + self.epsilon) / (
            predictions_flat.sum() + targets_flat.sum() + self.epsilon
        )

        # The Dice loss is 1 minus the Dice coefficient
        return 1. - dice_score
