import torch
from torch.utils.data import Dataset
import random
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate
import torch.nn.functional as F

# Must match OUTPUT_DIR in precompute_slices.py
PREPROCESSED_DIR = Path("preprocessed_slices")


class BrainTumorDataset(Dataset):

    def __init__(self, patient_folders):
        # Which patients belong to this split (train or val)
        patient_names = {p.name for p in patient_folders}

        # Only keep precomputed slice files belonging to these patients
        self.samples = [
            f for f in PREPROCESSED_DIR.glob("*.npz")
            if f.name.rsplit("_slice", 1)[0] in patient_names
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with np.load(self.samples[index]) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
            

        # data augmentation
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() < 0.3:  # slightly lower probability, rotation is the slowest augmentation
            angle = random.uniform(-10, 10)

            image = rotate(
                image, angle, axes=(0, 1), reshape=False,
                order=1, mode='constant', cval=0
            )
            mask = rotate(
                mask, angle, axes=(0, 1), reshape=False,
                order=0, mode='constant', cval=0
            )

        # Conv2D expects (channel, height, width), we have (height, width, channel)
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        # Resize image 256x256 -> 128x128
        image = F.interpolate(
            image.unsqueeze(0), size=(128, 128),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # Resize mask 256x256 -> 128x128
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=(128, 128),
            mode="nearest"
        ).squeeze(0).squeeze(0)

        return image, mask
