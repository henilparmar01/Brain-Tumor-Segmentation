import torch
import torch.nn as nn

class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        true_positive = (predictions * targets).sum()

        false_positive = ((1 - targets) * predictions).sum()

        false_negative = (targets * (1 - predictions)).sum()

        tversky = (
            true_positive + self.smooth
            ) / (
                true_positive
                + self.alpha * false_positive
                + self.beta * false_negative
                + self.smooth
            )

        return 1- tversky


class FocalLoss(nn.Module):

    def  __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none"
        )

        probabilities = torch.sigmoid(predictions)

        pt = torch.where(
            targets == 1,
            probabilities,
            1 - probabilities
        )

        focal_loss = (
            self.alpha
            * (1 - pt) ** self.gamma
            * bce_loss
        )

        return focal_loss.mean()


class TverskyFocalLoss(nn.Module):

    def __init__(self, tversky_weight=0.5, focal_weight=0.5):
        super().__init__()

        self.tversky = TverskyLoss(
            alpha=0.2,
            beta=0.8
        )

        self.focal = FocalLoss(
            alpha=0.25,
            gamma=2.0
        )

        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, predictions, targets):

        tversky_loss = self.tversky(
            predictions,
            targets
        )

        focal_loss = self.focal(
            predictions,
            targets
        )

        total_loss = (
            self.tversky_weight * tversky_loss
            + self.focal_weight * focal_loss
        )

        return total_loss

if __name__ == "__main__":

    predictions = torch.randn(2, 1, 256, 256)

    targets = torch.randint(
        0, 2, (2, 1, 256, 256)
    ).float()

    loss_fn = TverskyFocalLoss()

    loss = loss_fn(predictions, targets)

    print("Loss:", loss.item())
    