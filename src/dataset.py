import torch
from torch.utils.data import Dataset
import random
import numpy as np
from scipy.ndimage import rotate

from load_data import(
    get_patient_folders,
    get_patient_files,
    load_patient_data
)

from preprocess import preprocess_patient



class BrainTumorDataset(Dataset):

    def __init__(self, patient_folders):
        self.patient_folder = patient_folders
        
        self.samples=[]

        for patient_folder in self.patient_folder:

            for slice_index in range(155):

                self.samples.append((patient_folder, slice_index))


    def __len__(self):
       #return total number of patient in dataset
       return len(self.samples)

    def __getitem__(self, index):
        #return one preprocessed patient(image , mask)
        
        patient_folder, slice_index = self.samples[index]

        patient_files = get_patient_files(patient_folder)

        patient_data = load_patient_data(patient_files)

        image , mask = preprocess_patient(patient_data)

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

        return image, mask


def main():

    # Get patient folders
    patient_folders = get_patient_folders()

    print("Total patients:", len(patient_folders))

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