import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, smooth= 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice = (2*intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice

class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        total_loss = bce_loss + dice_loss

        return total_loss

if __name__ == "__main__":

    predictions = torch.randn(2, 1, 256, 256)

    targets = torch.randint(
        0, 2, (2, 1, 256, 256)
    ).float()

    loss_fn = BCEDiceLoss()

    loss = loss_fn(predictions, targets)

    print("Loss:", loss.item())