from torch.utils.data import DataLoader
from dataset import BrainTumorDataset


from sklearn.model_selection import train_test_split
from load_data import get_patient_folders
from config import BATCH_SIZE, NUM_WORKERS



def create_dataloader():

    patient_folders = get_patient_folders()
    print("Total Patients:", len(patient_folders))

    print("Patients used:", len(patient_folders))


    train_patients, val_patient = train_test_split(
        patient_folders,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_dataset = BrainTumorDataset(train_patients)

    val_dataset = BrainTumorDataset(val_patient)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True

    )

    return train_loader, val_loader

    

def main():
    train_loader, val_loader = create_dataloader()

    print("Train Samples:", len(train_loader.dataset))
    print("Validation Samples:", len(val_loader.dataset))

    images, masks = next(iter(train_loader))

    print("\nTrain Batch")
    print(images.shape)
    print(masks.shape)

    images, masks = next(iter(val_loader))

    print("\nValidation Batch")
    print(images.shape)
    print(masks.shape)
   

if __name__ == "__main__":
    main()