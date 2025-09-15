"""
Simple usage examples for CarotiCAT landmark extraction.

This file shows the most common use cases for the LandmarkAutomator class.
"""

import numpy as np
import nibabel as nib
from landmark_extractor import LandmarkAutomator


def example_folder_inference():
    """Example: Basic folder-based inference
    
    This example assumes a folder structure where each patient's folder contains 'cta.nii.gz'.
    
    The results (landmarks.txt, heatmaps.pt, template.json) will be saved in the specified output folder.
    
    Returns:
        landmarks_mm (np.ndarray): Array of shape (6, 3) with [x, y, z] coordinates in mm.
    """
    
    # Paths
    model_path = "/path/to/your/model.pth"
    input_folder = "/path/to/folder/containing/cta.nii.gz"
    output_folder = "/path/to/save/results"
    
    # Initialize automator
    automator = LandmarkAutomator(model_path, device="cuda")  # or "cpu"
    
    # Perform inference
    landmarks_mm = automator.infer_folder(
        folder_path=input_folder,
        output_folder=output_folder,
        save_mask=True,    # Save prediction mask
        save_json=True     # Save landmarks JSON
    )
    
    # landmarks_mm is a numpy array of shape (6, 3) with [x, y, z] coordinates
    print("Landmarks extracted:", landmarks_mm.shape)
    
    return landmarks_mm


def example_array_inference_from_file():
    """Example: Load NIfTI file and perform inference on array
    
    This example shows two ways to pass image data:
    1) Load NIfTI with nibabel and pass numpy array + affine
    2) Load NIfTI with nibabel and pass the nibabel object directly"""
    
    # Paths
    model_path = "/path/to/your/model.pth"
    nifti_path = "/path/to/your/image.nii.gz"
    
    # Load NIfTI file
    nib_img = nib.load(nifti_path)
    data = nib_img.get_fdata()  # numpy array
    affine = nib_img.affine     # affine matrix
    
    # Initialize automator
    automator = LandmarkAutomator(model_path, device="cuda")
    
    # Method 1: Pass numpy array + affine
    results = automator.infer_from_array(
        data=data,
        affine=affine,
        output_folder=None,  # Don't save files
        save_mask=False,
        save_json=False
    )
    
    # Method 2: Pass nibabel image directly
    results = automator.infer_from_array(
        data=nib_img,        # Pass the nibabel image directly
        output_folder=None,
        save_mask=False,
        save_json=False
    )
    
    # Results contain multiple outputs
    landmarks_mm = results['landmarks_mm']        # (6, 3) coordinates in mm
    landmarks_voxel = results['landmarks_voxel']  # (6, 3) coordinates in voxels
    heatmaps = results['heatmaps']               # (7, D, H, W) prediction heatmaps
    
    return results


def example_batch_processing():
    """Example: Process multiple images"""
    
    model_path = "/path/to/your/model.pth"
    input_folders = [
        "/path/to/patient1",
        "/path/to/patient2", 
        "/path/to/patient3"
    ]
    output_base = "/path/to/results"
    
    # Initialize once
    automator = LandmarkAutomator(model_path, device="cuda")
    
    all_landmarks = []
    
    for i, folder in enumerate(input_folders):
        print(f"Processing patient {i+1}/{len(input_folders)}: {folder}")
        
        try:
            landmarks = automator.infer_folder(
                folder_path=folder,
                output_folder=f"{output_base}/patient_{i+1}",
                save_mask=True,
                save_json=True
            )
            all_landmarks.append(landmarks)
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue
    
    # Convert to numpy array for analysis
    all_landmarks = np.array(all_landmarks)  # Shape: (n_patients, 6, 3)
    
    return all_landmarks


def example_just_get_coordinates():
    """Example: Quick coordinate extraction without saving files"""
    
    model_path = "/path/to/your/model.pth"
    nifti_path = "/path/to/your/image.nii.gz"
    
    # Initialize
    automator = LandmarkAutomator(model_path)
    
    # Quick inference - just get coordinates
    nib_img = nib.load(nifti_path)
    results = automator.infer_from_array(nib_img)
    
    # Extract just the mm coordinates
    landmarks = results['landmarks_mm']
    
    # Print landmark names and positions
    landmark_names = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
    for name, coords in zip(landmark_names, landmarks):
        print(f"{name}: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}] mm")
    
    return landmarks


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # example_folder_inference()
    # example_array_inference_from_file()
    # example_batch_processing()
    # example_just_get_coordinates()
    
    print("Update the paths in the examples and uncomment the function you want to test!")