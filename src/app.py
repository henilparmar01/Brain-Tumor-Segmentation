import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import UNet


# ============================================================
# Setup
# ============================================================

app = FastAPI(title="Brain Tumor Segmentation API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model ONCE when the server starts (not on every request)
print("Loading model...")
model = UNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("Model loaded and ready.")

# Folder where we save results so the user can download them
OUTPUT_DIR = Path("api_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Makes files inside OUTPUT_DIR downloadable via a URL like /outputs/filename.png
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ============================================================
# Helper functions
# ============================================================

def normalize_image(image):
    """Same normalization used during training (0-1 scaling)."""
    image = image.astype(np.float32)
    min_value = np.min(image)
    max_value = np.max(image)
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    return image


def run_inference(volume_4ch, threshold=0.5, batch_size=16):
    """
    volume_4ch: numpy array of shape (H, W, D, 4) - already normalized
    Returns: predicted mask volume of shape (H, W, D), values 0 or 1
    """
    H, W, D, C = volume_4ch.shape
    predicted_mask = np.zeros((H, W, D), dtype=np.uint8)

    with torch.no_grad():
        for start in range(0, D, batch_size):
            end = min(start + batch_size, D)
            batch_slices = []

            for slice_index in range(start, end):
                img_slice = volume_4ch[:, :, slice_index, :]           # (H, W, 4)
                img_slice = img_slice.transpose(2, 0, 1)                # (4, H, W)
                batch_slices.append(img_slice)

            batch_tensor = torch.from_numpy(np.stack(batch_slices)).float().to(device)  # (B, 4, H, W)

            # Resize down to 128x128, same as training
            batch_resized = F.interpolate(
                batch_tensor, size=(128, 128), mode="bilinear", align_corners=False
            )

            outputs = model(batch_resized)                              # (B, 1, 128, 128)
            probabilities = torch.sigmoid(outputs)

            # Resize prediction back up to original H, W
            probabilities_full = F.interpolate(
                probabilities, size=(H, W), mode="bilinear", align_corners=False
            )

            batch_pred = (probabilities_full > threshold).float().cpu().numpy()  # (B, 1, H, W)

            for i, slice_index in enumerate(range(start, end)):
                predicted_mask[:, :, slice_index] = batch_pred[i, 0].astype(np.uint8)

    return predicted_mask


def save_visualization(mri_slice, pred_mask_slice, out_path):
    """Save a side-by-side image: original scan vs predicted tumor overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(mri_slice, cmap="gray")
    axes[0].set_title("MRI Scan")
    axes[0].axis("off")

    axes[1].imshow(mri_slice, cmap="gray")
    axes[1].imshow(pred_mask_slice, cmap="Blues", alpha=0.5)
    axes[1].set_title("Predicted Tumor")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# API endpoint
# ============================================================

@app.post("/predict")
async def predict(
    t1c: UploadFile = File(...),
    t1n: UploadFile = File(...),
    t2f: UploadFile = File(...),
    t2w: UploadFile = File(...),
    threshold: float = 0.5,
):
    """
    Upload the 4 MRI modality files (.nii.gz) for one patient.
    Returns a predicted tumor mask (downloadable .npy) and a preview image.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Save uploaded files temporarily so nibabel can read them
        file_paths = {}
        for name, upload in [("t1c", t1c), ("t1n", t1n), ("t2f", t2f), ("t2w", t2w)]:
            save_path = tmp_dir / upload.filename
            with open(save_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            file_paths[name] = save_path

        # Load each modality
        images = {}
        for name, path in file_paths.items():
            img = nib.load(str(path)).get_fdata()
            images[name] = normalize_image(img)

        # Stack into (H, W, D, 4) - same order as training: t1c, t1n, t2f, t2w
        volume_4ch = np.stack(
            [images["t1c"], images["t1n"], images["t2f"], images["t2w"]],
            axis=-1
        )

        # Run the model on every slice
        predicted_mask = run_inference(volume_4ch, threshold=threshold)

        # ---- Save raw prediction (.npy) ----
        npy_filename = "predicted_mask.npy"
        npy_path = OUTPUT_DIR / npy_filename
        np.save(npy_path, predicted_mask)

        # ---- Save a preview image (the slice with the most predicted tumor) ----
        tumor_per_slice = predicted_mask.sum(axis=(0, 1))
        best_slice = int(np.argmax(tumor_per_slice))

        image_filename = "prediction_preview.png"
        image_path = OUTPUT_DIR / image_filename
        save_visualization(
            images["t1c"][:, :, best_slice],
            predicted_mask[:, :, best_slice],
            image_path
        )

        # ---- Basic stats ----
        total_tumor_voxels = int(predicted_mask.sum())
        slices_with_tumor = int((tumor_per_slice > 0).sum())

        return JSONResponse({
            "message": "Prediction complete",
            "threshold_used": threshold,
            "total_predicted_tumor_voxels": total_tumor_voxels,
            "slices_with_predicted_tumor": slices_with_tumor,
            "slice_with_most_tumor": best_slice,
            "raw_mask_download": f"/outputs/{npy_filename}",
            "preview_image": f"/outputs/{image_filename}",
        })


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Segmentation API is running.",
        "how_to_use": "Go to /docs in your browser to upload files and test the model.",
    }