from config import DATASET_DIR
import nibabel as nib


def get_patient_folders():

    #retun list of all patient folders

    patient_folders = []

    for item in DATASET_DIR.iterdir():

        if item.is_dir():
            patient_folders.append(item)

    return patient_folders


def get_patient_files(patient_folder):

    #reurn all MRI file path for one patient

    patient_files = {}

    for file in patient_folder.iterdir():

        if file.is_file():

            if file.name.endswith("-t1c.nii.gz"):
                patient_files["t1c"] = file

            elif file.name.endswith("-t1n.nii.gz"):
                patient_files["t1n"] = file

            elif file.name.endswith("-t2f.nii.gz"):
                patient_files["t2f"] = file

            elif file.name.endswith("-t2w.nii.gz"):
                patient_files["t2w"] = file

            elif file.name.endswith("-seg.nii.gz"):
                patient_files["mask"] = file

    return patient_files


def load_patient_data(patient_files):

    #load all MRI file into memory

    patient_data = {}

    for image_type , file_path in patient_files.items():

     image = nib.load(file_path)

     image = image.get_fdata()

     patient_data[image_type] = image

    return patient_data 


def main():

    patient_folders = get_patient_folders()

    print(f"Total Patients: {len(patient_folders)}\n")

    for patient in patient_folders:
        print(patient.name)

if __name__ == '__main__':
    main()