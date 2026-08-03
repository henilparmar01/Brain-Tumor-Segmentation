from torch.utils.data import Dataset

from load_data import(
    get_patient_folders,
    get_patient_files,
    load_patient_data
)

from preprocess import preprocess_patient



class BrainTumorDataset(Dataset):

    def __init__(self):
        self.patient_folder = get_patient_folders()

    def __len__(self):
       #return total number of patient in dataset
       return len(self.patient_folder)

    def __getitem__(self, index):
        #return one preprocessed patient(image , mask)
        
        patient_folder = self.patient_folder[index]

        patient_files = get_patient_files(patient_folder)

        patient_data = load_patient_data(patient_files)

        image , mask = preprocess_patient(patient_data)

        return image, mask


def main():
    pass

    
if __name__ == "__main__":
    main()

