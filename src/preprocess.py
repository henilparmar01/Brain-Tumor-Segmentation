import numpy as np

# normalize image

def normalize_image(image):

    image = image.astype(np.float32)
    min_value = np.min(image)
    max_value = np.max(image)

    if max_value > min_value:

        #normalization formula
        image = (image - min_value) / (max_value - min_value)

    return image

#combine four MRI modalities into one multi-channel image

def create_multi_channel_image(patient_data):

    t1c = patient_data["t1c"]
    t1n = patient_data["t1n"]
    t2f = patient_data["t2f"]
    t2w = patient_data["t2w"]

    multi_channel_image = np.stack(
        [t1c,t1n,t2f,t2w],
        axis=-1
    )

    return multi_channel_image


#normalize all MRI image and create multichannel image

def preprocess_patient(patient_data):

    patient_data["t1c"] = normalize_image(patient_data["t1c"])
    patient_data["t1n"] = normalize_image(patient_data["t1n"])
    patient_data["t2f"] = normalize_image(patient_data["t2f"])
    patient_data["t2w"] = normalize_image(patient_data["t2w"])

    multi_channel_image = create_multi_channel_image(patient_data)

    mask = patient_data["mask"]

    mask = (mask > 0).astype(np.float32)

    return multi_channel_image , mask    


def main():
    pass
    

if __name__ == "__main__":
    main()