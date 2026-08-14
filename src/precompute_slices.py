import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

from load_data import get_patient_folders, get_patient_files, load_patient_data
from preprocess import preprocess_patient


OUTPUT_DIR = Path("preprocessed_slices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMPTY_SLICE_RATIO = 0.2


def main():
    patient_folders = get_patient_folders()
    print(f"Preprocessing {len(patient_folders)} patients into 2D slices...")

    for patient_folder in tqdm(patient_folders):
        patient_files = get_patient_files(patient_folder)
        patient_data = load_patient_data(patient_files)
        image, mask = preprocess_patient(patient_data)

        tumor_slices = []
        empty_slices = []

        for slice_index in range(mask.shape[2]):
            if np.any(mask[:, :, slice_index] > 0):
                tumor_slices.append(slice_index)
            else:
                empty_slices.append(slice_index)

        n_keep = int(len(empty_slices) * EMPTY_SLICE_RATIO)
        selected_empty = random.sample(empty_slices, n_keep)

        keep_slices = tumor_slices + selected_empty

        for slice_index in keep_slices:
            img_slice = image[:, :, slice_index, :].astype(np.float16)  # float16 = half the size, fine for 0-1 data
            mask_slice = mask[:, :, slice_index].astype(np.uint8)

            out_path = OUTPUT_DIR / f"{patient_folder.name}_slice{slice_index}.npz"
            np.savez(out_path, image=img_slice, mask=mask_slice)

    print("Done! Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
