import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Paths

GT_PATH = r"../Brain Tumor Data/BraTS-GLI-01000-000/BraTS-GLI-01000-000-seg.nii.gz"
PRED_PATH = r"api_outputs/predicted_mask.npy"

# Load masks

ground_truth = nib.load(GT_PATH).get_fdata()
prediction = np.load(PRED_PATH)


# Convert ground truth into binary mask
ground_truth = (ground_truth > 0).astype(np.uint8)
prediction = (prediction > 0).astype(np.uint8)


print("Ground truth shape:", ground_truth.shape)
print("Prediction shape:", prediction.shape)


# Check shapes

if ground_truth.shape != prediction.shape:
    raise ValueError(
        f"Shape mismatch! Ground truth: {ground_truth.shape}, "
        f"Prediction: {prediction.shape}"
    )

# Calculate Dice Score


intersection = np.logical_and(ground_truth, prediction).sum()

dice = (2 * intersection) / (
    ground_truth.sum() + prediction.sum() + 1e-8
)

print(f"\nDice Score: {dice:.4f}")


# Find slice with most actual tumor

gt_tumor_per_slice = ground_truth.sum(axis=(0, 1))

best_slice = int(np.argmax(gt_tumor_per_slice))

print("Slice with most actual tumor:", best_slice)


# Visualization


plt.figure(figsize=(12, 4))


# Original MRI
plt.subplot(1, 3, 1)

# We use prediction only to identify the slice.
# For visualization, load the T1c MRI.
t1c_path = r"../Brain Tumor Data/BraTS-GLI-01000-000/BraTS-GLI-01000-000-t1c.nii.gz"

t1c = nib.load(t1c_path).get_fdata()

plt.imshow(t1c[:, :, best_slice], cmap="gray")
plt.title("MRI")
plt.axis("off")


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(t1c[:, :, best_slice], cmap="gray")
plt.imshow(
    ground_truth[:, :, best_slice],
    cmap="Reds",
    alpha=0.5
)

plt.title("Actual Tumor")
plt.axis("off")


# Prediction
plt.subplot(1, 3, 3)

plt.imshow(t1c[:, :, best_slice], cmap="gray")
plt.imshow(
    prediction[:, :, best_slice],
    cmap="Blues",
    alpha=0.5
)

plt.title("Predicted Tumor")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "../actual_vs_predicted.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()