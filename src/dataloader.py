from torch.utils.data import DataLoader
from dataset import BrainTumorDataset

from sklearn.model_selection import train_test_split
from config import BATCH_SIZE, NUM_WORKERS

from pathlib import Path


PREPROCESSED_DIR = Path("preprocessed_slices")


def create_dataloader():

    # Get all preprocessed .npz files
    slice_files = list(PREPROCESSED_DIR.glob("*.npz"))

    print("Total .npz files:", len(slice_files))

    # Extract unique patient names from filenames
    patient_names = sorted({
        f.name.rsplit("_slice", 1)[0]
        for f in slice_files
    })

    print("Total Patients:", len(patient_names))

    # Convert patient names into Path objects
    # BrainTumorDataset expects objects with .name
    patient_paths = [Path(name) for name in patient_names]

    # Split patients: 80% train, 20% validation
    train_patients, val_patients = train_test_split(
        patient_paths,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print("Train Patients:", len(train_patients))
    print("Validation Patients:", len(val_patients))

    # Create datasets
    train_dataset = BrainTumorDataset(train_patients)
    val_dataset = BrainTumorDataset(val_patients)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    return train_loader, val_loader


def main():

    train_loader, val_loader = create_dataloader()

    print("Train Samples:", len(train_loader.dataset))
    print("Validation Samples:", len(val_loader.dataset))

    images, masks = next(iter(train_loader))

    print("\nTrain Batch")
    print("Images:", images.shape)
    print("Masks:", masks.shape)

    images, masks = next(iter(val_loader))

    print("\nValidation Batch")
    print("Images:", images.shape)
    print("Masks:", masks.shape)


if __name__ == "__main__":
    main()