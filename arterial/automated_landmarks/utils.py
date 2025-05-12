####utils for CarotiCAT

###starting with preprocessing functions 

import os
import torchio as tio

####individiual version which is most likely the one more used since there's a patient at a time
def resample_image(image, new_voxel_size):
    return tio.Resample(new_voxel_size, scalars_only=True)(image)

###just in case we need batch processing
def process_files_resample(folder_path, new_voxel_size=(0.8, 0.8, 0.8)):
    for root, _, files in os.walk(folder_path):
        if "cta.nii.gz" in files:
            file_path = os.path.join(root, "cta.nii.gz")
            resampled = resample_image(tio.ScalarImage(file_path), new_voxel_size)
            resampled.save(file_path)
            print(f"Resampled: {file_path}")
            
def process_files_crop(folder_path, target_shape=(320, 320, 480)):
    for root, _, files in os.walk(folder_path):
        if "cta.nii.gz" in files:
            file_path = os.path.join(root, "cta.nii.gz")
            cropped = tio.CropOrPad(target_shape)(tio.ScalarImage(file_path))
            cropped.save(file_path)
            print(f"Cropped/Padded: {file_path}")