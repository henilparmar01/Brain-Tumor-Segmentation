import torch
from torch.utils.data import Dataset

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

        #Conv2D accept (channel, height, width) but we have (height, width, channel)
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask


def main():
    pass
   

    
if __name__ == "__main__":
    main()

