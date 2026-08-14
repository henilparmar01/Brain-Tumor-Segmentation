import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
from dataloader import create_dataloader


def dice_score(pred_mask, true_mask, smooth=1.0):
    """
    Dice score = how much the predicted tumor area overlaps with the real tumor area.
    1.0 = perfect match, 0.0 = no overlap at all.
    """
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    intersection = (pred_mask * true_mask).sum()

    dice = (2.0 * intersection + smooth) / (
        pred_mask.sum() + true_mask.sum() + smooth
    )

    return dice.item()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    model = UNet().to(device)
    model.load_state_dict(torch.load("src/best_model.pth", map_location=device))
    model.eval()

    # Get validation data
    print("Loading validation data...")
    _, val_loader = create_dataloader()

    # Collect samples that ACTUALLY contain a tumor (skip empty slices)
    # so we get a meaningful test of the model, not trivial "no tumor" freebies.
    print("Searching for validation slices that contain a tumor...")
    tumor_images = []
    tumor_masks = []
    needed = 5

    with torch.no_grad():
        for images_batch, masks_batch in val_loader:
            for i in range(images_batch.shape[0]):
                if masks_batch[i].sum() > 0:  # this slice has a real tumor in it
                    tumor_images.append(images_batch[i])
                    tumor_masks.append(masks_batch[i])
                    if len(tumor_images) >= needed:
                        break
            if len(tumor_images) >= needed:
                break

    if len(tumor_images) == 0:
        print("No tumor-containing slices found in validation set! Something may be wrong.")
        return

    images = torch.stack(tumor_images).to(device)
    masks = torch.stack(tumor_masks).to(device)

    # Run prediction (no gradient needed, just looking at results)
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)          # convert raw output to 0-1 probability
        predictions = (probabilities > 0.3).float()      # convert to binary mask (0 or 1)

    # How many examples to show
    num_examples = min(needed, images.shape[0])
    print(f"Found {num_examples} tumor-containing slices to visualize.")

    # Calculate average Dice score across shown examples
    dice_scores = []

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

    for i in range(num_examples):

        # Use one MRI channel (t1c, channel index 0) just for display
        image_display = images[i, 0].cpu().numpy()

        true_mask = masks[i].cpu().numpy()
        pred_mask = predictions[i, 0].cpu().numpy()

        score = dice_score(predictions[i, 0], masks[i])
        dice_scores.append(score)

        # Original scan
        axes[i, 0].imshow(image_display, cmap="gray")
        axes[i, 0].set_title(f"Sample {i+1} - MRI Scan")
        axes[i, 0].axis("off")

        # Ground truth mask (real tumor)
        axes[i, 1].imshow(image_display, cmap="gray")
        axes[i, 1].imshow(true_mask, cmap="Reds", alpha=0.5)
        axes[i, 1].set_title("Ground Truth (Real Tumor)")
        axes[i, 1].axis("off")

        # Predicted mask (model's guess)
        axes[i, 2].imshow(image_display, cmap="gray")
        axes[i, 2].imshow(pred_mask, cmap="Blues", alpha=0.5)
        axes[i, 2].set_title(f"Prediction (Dice: {score:.3f})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("predictions_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to predictions_comparison.png")

    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"\nAverage Dice Score (these {num_examples} tumor examples): {avg_dice:.4f}")
    print("(1.0 = perfect overlap, 0.0 = no overlap)")

    # Now compute a Dice score across a SAMPLE of the full validation set
    # (not literally every single slice, to keep this fast) for a more reliable
    # average than just the 5 examples shown above.
    MAX_SLICES_TO_CHECK = 300  # adjust up/down depending on how much time you want to spend

    print(f"\nCalculating average Dice score across up to {MAX_SLICES_TO_CHECK} "
          f"tumor-containing validation slices (faster than checking everything)...")
    all_dice_scores = []

    with torch.no_grad():
        for images_batch, masks_batch in val_loader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            outputs_batch = model(images_batch)
            preds_batch = (torch.sigmoid(outputs_batch) > 0.5).float()

            for i in range(images_batch.shape[0]):
                if masks_batch[i].sum() > 0:  # only score slices that actually contain a tumor
                    all_dice_scores.append(
                        dice_score(preds_batch[i, 0], masks_batch[i])
                    )
                    if len(all_dice_scores) >= MAX_SLICES_TO_CHECK:
                        break
            if len(all_dice_scores) >= MAX_SLICES_TO_CHECK:
                break

    overall_avg = sum(all_dice_scores) / len(all_dice_scores)
    print(f"Average Dice Score (sample of {len(all_dice_scores)} tumor slices): {overall_avg:.4f}")

    plt.show()


if __name__ == "__main__":
    main()