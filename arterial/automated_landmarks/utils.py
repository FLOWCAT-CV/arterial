####utils for CarotiCAT

###starting with preprocessing functions: resampling and cropping

import os
import torchio as tio
import os
import nibabel as nib
import numpy as np
import json
import shutil

####individiual version which is most likely the one more used since there's a patient at a time
def resample_image(image, new_voxel_size):
    """
    Resample the image to a new voxel size.
    Args:
        image (torchio.ScalarImage): The input image to be resampled.
        new_voxel_size (tuple): The new voxel size in mm (x, y, z).
    Returns:
        torchio.ScalarImage: The resampled image.
    """
    return tio.Resample(new_voxel_size, scalars_only=True)(image)

###just in case we need batch processing
def process_files_resample(folder_path, new_voxel_size=(0.8, 0.8, 0.8)):
    """
    Resample all NIfTI files in the given folder to a new voxel size.
    Args:
        folder_path (str): The path to the folder containing NIfTI files.
        new_voxel_size (tuple): The new voxel size in mm (x, y, z).
    """
    for root, _, files in os.walk(folder_path):
        if "cta.nii.gz" in files:
            file_path = os.path.join(root, "cta.nii.gz")
            resampled = resample_image(tio.ScalarImage(file_path), new_voxel_size)
            resampled.save(file_path)
            print(f"Resampled: {file_path}")
            
def process_files_crop(folder_path, target_shape=(320, 320, 480)):
    """
    Crop or pad all NIfTI files in the given folder to a target shape.
    Args:
        folder_path (str): The path to the folder containing NIfTI files.
        target_shape (tuple): The target shape for cropping or padding.
    """
    for root, _, files in os.walk(folder_path):
        if "cta.nii.gz" in files:
            file_path = os.path.join(root, "cta.nii.gz")
            cropped = tio.CropOrPad(target_shape)(tio.ScalarImage(file_path))
            cropped.save(file_path)
            print(f"Cropped/Padded: {file_path}")

### adding origin changer functions required in the original setup of CarotiCAT
### this function is used to change the origin of the nifti file to a new origin
def change_origin_preprocess(nifti_path, new_origin):
    """
    Change the origin of a NIfTI file to a new origin.
    Args:
        nifti_path (str): The path to the NIfTI file.
        new_origin (tuple): The new origin in mm (x, y, z).
    Returns:
        str: The path to the modified NIfTI file.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    new_affine = np.copy(affine)
    #save the previous affine in a txt in the same folder before changing it
    np.savetxt(os.path.join(os.path.dirname(nifti_path), "affine_before_origin_change.txt"), affine)
    new_affine[:3, 3] = new_origin
    new_img = nib.Nifti1Image(data, new_affine)
    nib.save(new_img, nifti_path)
    return nifti_path

def process_files_change_origin(folder_path, new_origin=(0, 0, 0)):
    """
    Change the origin of all NIfTI files in the given folder to a new origin.
    Args:
        folder_path (str): The path to the folder containing NIfTI files.
        new_origin (tuple): The new origin in mm (x, y, z).
    """
    for root, _, files in os.walk(folder_path):
        if "cta.nii.gz" in files:
            change_origin_preprocess(os.path.join(root, "cta.nii.gz"), new_origin)
            print(f"Origin changed: {root}/cta.nii.gz")

def restore_centroids_to_original_origin(centroids_mm, affine_path, image_orientation):
    """
    Restore centroids to the original image space using the affine transformation.
    Args:
        centroids_mm (np.ndarray): Centroids in mm to be restored.
        affine_path (str): Path to the affine transformation file.
        image_orientation (tuple): Orientation of the image (e.g., ('L', 'P', 'S')).
    Returns:
        np.ndarray: Centroids restored to the original image space.
    """
    affine = np.loadtxt(affine_path)
    centroids_mm_restored = np.copy(centroids_mm)

    if image_orientation == ('L', 'P', 'S'):
        centroids_mm_restored[:, 0] -= affine[0, 3]
        centroids_mm_restored[:, 1] -= affine[1, 3]
        centroids_mm_restored[:, 2] += affine[2, 3]
    else:  # LAS
        centroids_mm_restored[:, 0] -= affine[0, 3]
        centroids_mm_restored[:, 1] = ((-centroids_mm[:, 1]) - affine[1, 3]) * -1
        centroids_mm_restored[:, 2] += affine[2, 3]
        
        # Optional safety tweak
        for i in range(centroids_mm_restored.shape[0]):
            if np.abs(centroids_mm_restored[i, 1]) > 180:
                centroids_mm_restored[i, 1] -= 2 * affine[1, 3]

    return centroids_mm_restored