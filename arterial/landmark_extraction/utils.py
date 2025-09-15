#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

####utils for CarotiCAT 

###starting with preprocessing functions: resampling and cropping

import os
import torchio as tio
import os
import nibabel as nib
import json
import numpy as np
import shutil
import cc3d
import cv2

####individual version which is most likely the one more used since there's a patient at a time
def resample_image(image, new_voxel_size):
    """
    Resample the image to a new voxel size.
    Parameters
    ----------
        image (torchio.ScalarImage): The input image to be resampled.
        new_voxel_size (tuple): The new voxel size in mm (x, y, z).
    Returns
    -------
        torchio.ScalarImage: The resampled image.
    """
    return tio.Resample(new_voxel_size, scalars_only=True)(image)

def crop_or_pad_image(image, target_shape):
    """
    Crop or pad the image to a target shape.
    Parameters
    ----------
        image (torchio.ScalarImage): The input image to be cropped or padded.
        target_shape (tuple): The target shape for cropping or padding.
    Returns
    -------
        torchio.ScalarImage: The cropped or padded image.
    """
    return tio.CropOrPad(target_shape)(image)

###just in case we need batch processing
def process_files_resample(folder_path, new_voxel_size=(0.8, 0.8, 0.8)):
    """
    Resample all NIfTI files in the given folder to a new voxel size.
    Parameters
    ----------
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
    Parameters
    ----------
        folder_path (str): The path to the folder containing NIfTI files.
        target_shape (tuple): The target shape for cropping or padding.
    """
    for root, _, files in os.walk(folder_path):
        if "cta.nii.gz" in files:
            file_path = os.path.join(root, "cta.nii.gz")
            cropped = crop_or_pad_image(tio.ScalarImage(file_path), target_shape)
            cropped.save(file_path)
            print(f"Cropped/Padded: {file_path}")

### adding origin changer functions required in the original setup of CarotiCAT
### this function is used to change the origin of the nifti file to a new origin
def change_origin_preprocess(nifti_path_or_tio_image, new_origin):
    """
    Change the origin of a NIfTI file or TorchIO image to a new origin.
    Parameters
    ----------
        nifti_path_or_tio_image (str or torchio.ScalarImage): The path to the NIfTI file or TorchIO image.
        new_origin (tuple): The new origin in mm (x, y, z).
    Returns
    -------
        str or torchio.ScalarImage: The path to the modified NIfTI file or modified TorchIO image.
    """
    if isinstance(nifti_path_or_tio_image, str):
        # Original file path version
        nifti_path = nifti_path_or_tio_image
        img = nib.load(nifti_path)
        data = img.get_fdata()
        affine = img.affine

        new_affine = np.copy(affine)
        np.savetxt(os.path.join(os.path.dirname(nifti_path), "affine_before_origin_change.txt"), affine)
        new_affine[:3, 3] = new_origin

        new_img = nib.Nifti1Image(data, new_affine, header=img.header)
        new_img.set_qform(new_affine, code=1)
        new_img.set_sform(new_affine, code=1)

        nib.save(new_img, nifti_path)
        return nifti_path
    else:
        # TorchIO image version - return modified copy
        tio_image = nifti_path_or_tio_image
        print("utils iimage read")
        data = tio_image.data.numpy()
        print("read the tio object")
        affine = tio_image.affine
        print("was the affine the one that failed?")
        
        new_affine = np.copy(affine)
        new_affine[:3, 3] = new_origin
        
        # Create new TorchIO image with modified affine
        import torch
        new_tio_image = tio.ScalarImage(tensor=torch.from_numpy(data), affine=new_affine)
        return new_tio_image

def process_files_change_origin(folder_path, new_origin=(0, 0, 0)):
    """
    Change the origin of all NIfTI files in the given folder to a new origin.
    Parameters
    ----------
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
    Parameters
    ----------
        centroids_mm (np.ndarray): Centroids in mm to be restored.
        affine_path (str): Path to the affine transformation file.
        image_orientation (tuple): Orientation of the image (e.g., ('L', 'P', 'S')).
    Returns
    -------
        np.ndarray: Centroids restored to the original image space.
    """
    if not os.path.exists(affine_path):
        print(f"Warning: Affine file {affine_path} not found. Returning original centroids.")
        return centroids_mm
        
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

#this one here reads the json template and updates the control points with the predicted values
def update_json_with_predictions(predichas, output_json_path, original_json_path=None, template_json_path="configs/template_landmark.json"):
    """
    Update the JSON file with predicted control points.
    Parameters
    ----------
        predichas (np.ndarray): Predicted control points.
        output_json_path (str): Path to save the updated JSON file.
        original_json_path (str): Path to the original JSON file (if any).
        template_json_path (str): Path to the template JSON file.
    """
    # Check if the original JSON file exists; if not, use the template
    json_to_use = original_json_path if original_json_path and os.path.exists(original_json_path) else template_json_path
    
    # Try to load from file first, then fallback to embedded template
    try:
        with open(json_to_use, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IOError):
        # Fallback to embedded template
        try:
            from template_embedded import load_template_from_embedded
            data = load_template_from_embedded()
            print(f"Warning: Could not load template from {json_to_use}, using embedded template")
        except ImportError:
            raise FileNotFoundError(f"Template file {json_to_use} not found and embedded template not available")

    label_to_index = {"l-tica": 0, "r-tica": 1, "l-eica": 2, "r-eica": 3, "r-mca": 4, "l-mca": 5}
    control_points = data["markups"][0]["controlPoints"]
    for cp in control_points:
        label = cp.get("label")
        if label in label_to_index:
            cp["position"] = predichas[label_to_index[label]].tolist()

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)


def actualizar_json_con_predicciones(original_json_path, predichas, output_json_path):
    """
    Original function from training script.
    Reads the original JSON and updates the position of each landmark with the predicted
    coordinates (in mm, in LPS).
    """
    with open(original_json_path, 'r') as f:
        data = json.load(f)
    
    labels = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
    control_points = data["markups"][0]["controlPoints"]
    
    for landmark_idx, label in enumerate(labels):
        for cp in control_points:
            if cp["label"] == label:
                cp["position"] = predichas[landmark_idx].tolist()
                break
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)