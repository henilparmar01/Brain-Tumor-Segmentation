import torch
from torch.utils.data import Dataset
import random
import numpy as np
from scipy.ndimage import rotate
import torch.nn.functional as F

from load_data import(
    get_patient_folders,
    get_patient_files,
    load_patient_data
)

from preprocess import preprocess_patient



class BrainTumorDataset(Dataset):

    def __init__(self, patient_folders, empty_slice_ratio = 0.5):
        self.patient_folder = patient_folders
        
        self.samples=[]

        self.cached_patient = None
        self.cached_data = None

        for patient_folder in self.patient_folder:

            patient_files = get_patient_files(patient_folder)

            patient_data = load_patient_data(patient_files)

            image, mask = preprocess_patient(patient_data)

            mask_slices_with_tumor = []
            mask_slices_without_tumor = []

            for slice_index in range(mask.shape[2]):

                slice_mask = mask[:, :, slice_index]

                if np.any(slice_mask > 0):
                    mask_slices_with_tumor.append(slice_index)

                else:
                    mask_slices_without_tumor.append(slice_index)

            #Keep All slices containing tumor

            for slice_index in mask_slices_with_tumor:
                self.samples.append(
                    (patient_folder, slice_index)
                )

            #Keep only 50% of empty slices
            number_to_keep = int(
                len(mask_slices_without_tumor)
                * empty_slice_ratio
            )

            selected_empty_slices = random.sample(
                mask_slices_without_tumor,
                number_to_keep
            )

            for slie_index in selected_empty_slices:
                self.samples.append(
                    (patient_folder, slice_index)
                )

        random.shuffle(self.samples)


    def __len__(self):
       #return total number of patient in dataset
       return len(self.samples)

    def __getitem__(self, index):
        #return one preprocessed patient(image , mask)
        
        patient_folder, slice_index = self.samples[index]

        #check if this patient is already cached

        if self.cached_patient != patient_folder:

            print("Loading:", patient_folder)

            patient_files = get_patient_files(patient_folder)
            
            patient_data = load_patient_data(patient_files)
            
            image , mask = preprocess_patient(patient_data)

            #Store this patient in cache
            self.cached_patient = patient_folder
            self.cached_data = (image, mask)

        else:
            print("Using cache:", patient_folder)

            #Use already loaded patient
            image, mask = self.cached_data

        #Get Required slice
        
        image = image[:, :, slice_index, :]
        mask = mask[:, :, slice_index]

        # data augmantation

        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() < 0.5:
            angle = random.uniform(-10, 10)

            image = rotate(
                image,
                angle,
                axes=(0, 1),
                reshape=False,
                order=1,
                mode='constant',
                cval=0
            )

            mask = rotate(
                mask,
                angle,
                axes=(0, 1),
                reshape=False,
                order=0,
                mode='constant',
                cval=0
            )

            
        #Conv2D accept (channel, height, width) but we have (height, width, channel)
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Resize image 256 x 256 -> 128 x 128
        image = F.interpolate(
            image.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Resize mask 256 x 256 -> 128 x 128
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(128,128),
            mode="nearest"
        
        ).squeeze(0).squeeze(0)

        return image, mask


def main():

    # Get patient folders
    patient_folders = get_patient_folders()

    print("Total patients:", len(patient_folders))

    # Use only 100 patient
    patient_folders = sorted(patient_folders)[:100]

    print("Patient used:",len(patient_folders))

    # Create dataset
    dataset = BrainTumorDataset(patient_folders)

    print("Total samples:", len(dataset))

    # Get one sample
    image, mask = dataset[75]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)

    print("Image min:", image.min().item())
    print("Image max:", image.max().item())

    print("Mask min:", mask.min().item())
    print("Mask max:", mask.max().item())


if __name__ == "__main__":
    main()