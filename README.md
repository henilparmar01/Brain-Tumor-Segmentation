# Brain Tumor Segmentation

A deep learning pipeline for segmenting brain tumors from multi-modal MRI scans, built with PyTorch and following BraTS-style data conventions. Includes a full training pipeline and a FastAPI inference server for testing on new patient scans.

## Overview

This project takes four MRI modalities per patient (T1c, T1n, T2f, T2w) and predicts a pixel-level tumor segmentation mask using a 2D U-Net trained on axial slices. The pipeline covers preprocessing, training, evaluation, and deployment via a REST API.

## Results

| Metric | Score |
|---|---|
| Best Validation Loss | **0.0375** |
| Best Validation Dice | **0.9123** |
| Overall Dice (tumor-containing slices, validation set) | **0.8569** |

Trained on 299 patients (240 train / 60 validation) from the BraTS dataset, using a combined Tversky + Focal loss tuned to prioritize recall (catching tumors, including small ones) over precision.

### Training Progress

The model was iteratively improved across several rounds of tuning:

| Version | Val Loss | Notes |
|---|---|---|
| Baseline | ~0.048 | Initial working pipeline, struggled with small tumors |
| Tuned loss + threshold | ~0.0375 | Adjusted Tversky loss (`alpha=0.2, beta=0.8`), reduced empty-slice ratio, lowered prediction threshold |

## Architecture

- **Model:** Standard 2D U-Net (4 down/up blocks, skip connections)
- **Input:** 4-channel MRI slices (T1c, T1n, T2f, T2w), resized to 128x128
- **Output:** Single-channel tumor probability mask
- **Loss:** Combined Tversky Loss (`alpha=0.2, beta=0.8`) + Focal Loss
- **Optimizer:** Adam with `ReduceLROnPlateau` scheduling
- **Training:** Automatic mixed precision (AMP), early stopping (patience=5)

## Project Structure

```
src/
├── config.py              # Paths and hyperparameters
├── load_data.py            # Locate and load raw patient files
├── preprocess.py           # Normalize modalities, binarize masks
├── precompute_slices.py    # One-time preprocessing: raw volumes -> 2D slice files
├── dataset.py               # PyTorch Dataset (reads precomputed slices)
├── dataloader.py            # Train/val split and DataLoader creation
├── model.py                 # U-Net architecture
├── loss.py                  # Tversky + Focal loss
├── train.py                 # Training loop with early stopping
├── visualize_predictions.py # Visual comparison + Dice scoring on validation data
└── app.py                    # FastAPI inference server
```

## Setup

```bash
pip install torch torchvision nibabel numpy scipy tqdm matplotlib fastapi uvicorn python-multipart
```

## Usage

### 1. Preprocess the dataset (run once)

```bash
python src/precompute_slices.py
```

Converts raw `.nii.gz` patient volumes into fast-loading 2D slice files, keeping all tumor-containing slices plus a sample of empty slices.

### 2. Train the model

```bash
python src/train.py
```

Trains with early stopping and saves the best-performing model as `best_model.pth`.

### 3. Evaluate / visualize predictions

```bash
python src/visualize_predictions.py
```

Runs the trained model on validation slices, saves a side-by-side comparison image (MRI scan / ground truth / prediction), and prints the average Dice score.

### 4. Run the inference API

```bash
cd src
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to upload a patient's 4 MRI files and get back a predicted tumor mask (`.npy`) plus a preview image.

## Dataset

Trained on BraTS-style data: four co-registered MRI modalities per patient (T1c, T1n, T2f, T2w) with expert-annotated segmentation masks.

## Notes

- Small/tiny tumors remain the hardest case for the model (a well-known challenge in medical image segmentation), though loss-function tuning and data balancing meaningfully improved performance on them.
- Currently trained on a 299-patient subset; scaling to the full dataset (1251 patients) is expected to further improve performance, particularly on small tumors.

## License

For educational and research purposes.
