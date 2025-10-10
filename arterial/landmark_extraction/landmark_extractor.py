#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import numpy as np
import torch
import nibabel as nib
import cc3d
import cv2
import torchio as tio
import tempfile

from arterial.landmark_extraction.utils import (
    actualizar_json_con_predicciones,
    restore_centroids_to_original_origin,
    resample_image,
    crop_or_pad_image,
    change_origin_preprocess,
)
from arterial.landmark_extraction.model import load_trained_model_seg

class SingleCTADataset:
    """
    Dataset class for loading a single CTA image.
    Parameters
    ----------
        folder_path (str): Path to the folder containing 'cta.nii.gz'.
    Returns
    -------
        dict: A dictionary with keys 'volume', 'affine', and 'folder'.
    """
    def __init__(self, folder_path: str):
        """Initialize the dataset with the folder path.
        Parameters
        ----------
            folder_path (str): Path to the folder containing 'cta.nii.gz'.
        """
        self.folder_path = folder_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cta_path = os.path.join(self.folder_path, "cta.nii.gz")
        # Load original image once
        tio_object = tio.ScalarImage(cta_path)
        
        # Store original affine before preprocessing
        original_affine = tio_object.affine

        # Preprocess in memory - EXACTLY like training script
        resampled = resample_image(tio_object, (0.8, 0.8, 0.8))
        cropped_or_padded = crop_or_pad_image(resampled, (320, 320, 480))
        
        # Change origin and save affine info
        temp_affine_path = os.path.join(self.folder_path, "affine_before_origin_change.txt")
        np.savetxt(temp_affine_path, cropped_or_padded.affine)
        final_image = change_origin_preprocess(cropped_or_padded, (0, 0, 0))

        # Get as nibabel image directly (no disk write)
        img = final_image.numpy()  # Should be (1, 320, 320, 480)
        #convert it to 320 x320x480
        img = np.squeeze(img)  # Remove channel dim if exists
        # Normalize and shape - match original training script EXACTLY
        volume = np.clip(img, 0, 700) / 700
        # Transpose to match training: (H, W, D) -> (D, H, W) -> add channel dim
        volume = np.transpose(volume, (2, 0, 1))
        volume = np.expand_dims(volume, axis=0)    # (1, D, H, W) - add channel dim
        volume = torch.tensor(volume, dtype=torch.float32)

        print(f"Preprocessed volume shape: {volume.shape}")  # Should be (1, 480, 320, 320)

        return {
            "volume": volume,
            "affine": final_image.affine,  # original affine for centroids
            "folder": self.folder_path
        }

class SingleCTAFromArray:
    """
    Dataset class for loading a single CTA from numpy array or nibabel object.
    Parameters
    ----------
        data (np.ndarray or nibabel image): CTA data.
        affine (np.ndarray, optional): Affine transformation matrix. If None, identity matrix is used.
        temp_folder (str, optional): Temporary folder path for processing. If None, creates one.
    """
    def __init__(self, data, affine=None, temp_folder=None):
        if isinstance(data, np.ndarray):
            if affine is None:
                affine = np.eye(4)
            self.nib_img = nib.Nifti1Image(data, affine)
        elif hasattr(data, 'get_fdata'):  # nibabel image
            self.nib_img = data
        else:
            raise ValueError("Data must be numpy array or nibabel image")
        
        self.original_affine = self.nib_img.affine
        self.temp_folder = temp_folder or tempfile.mkdtemp()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Create temporary file to work with TorchIO
        temp_nii_path = os.path.join(self.temp_folder, "temp_cta.nii.gz")
        nib.save(self.nib_img, temp_nii_path)
        
        # Load with TorchIO
        tio_object = tio.ScalarImage(temp_nii_path)
        
        # Store original affine before preprocessing
        original_affine = self.original_affine

        # Preprocess in memory - EXACTLY like training script
        resampled = resample_image(tio_object, (0.8, 0.8, 0.8))
        cropped_or_padded = crop_or_pad_image(resampled, (320, 320, 480))
        
        # Change origin and save affine info
        temp_affine_path = os.path.join(self.temp_folder, "affine_before_origin_change.txt")
        np.savetxt(temp_affine_path, cropped_or_padded.affine)
        
        final_image = change_origin_preprocess(cropped_or_padded, (0, 0, 0))

        # Get as nibabel image directly (no disk write)
        img = final_image.numpy()
        img = np.squeeze(img)  # Remove channel dim if exists

        # Normalize and shape - match original training script EXACTLY
        volume = np.clip(img, 0, 700) / 700
        # Transpose to match training: (H, W, D) -> (D, H, W) -> add channel dim
        volume = np.transpose(volume, (2, 0, 1))  # (D, H, W)
        volume = np.expand_dims(volume, axis=0)    # (1, D, H, W) - add channel dim
        volume = torch.tensor(volume, dtype=torch.float32)

        print(f"Preprocessed volume shape: {volume.shape}")  # Should be (1, 480, 320, 320)

        return {
            "volume": volume,
            "affine": final_image.affine,  # original affine for centroids
            "folder": self.temp_folder
        }

class LandmarkAutomator:
    """
    Class for automating landmark extraction from CTA images.
    Parameters
    ----------
        model_path (str): Path to the trained model file.
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to None, which auto-selects.
    Methods
    -------
        infer_folder(folder_path, output_folder, save_mask=True, save_json=True):
            Perform inference on a folder containing 'cta.nii.gz' and save results.
        infer_from_array(data, affine=None, output_folder=None, save_mask=False, save_json=False):
            Perform inference on numpy array or nibabel image and return landmarks.
    """
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_trained_model_seg(model_path, self.device)
        self.model.eval()

    def _prepare_input_folder(self, folder_path: str):
        dataset = SingleCTADataset(folder_path)
        sample = dataset[0]
        volume = sample["volume"].unsqueeze(0).to(self.device)  # Add batch dimension
        return volume, sample

    def _prepare_input_array(self, data, affine=None, temp_folder=None):
        dataset = SingleCTAFromArray(data, affine, temp_folder)
        sample = dataset[0]
        volume = sample["volume"].unsqueeze(0).to(self.device)  # Add batch dimension
        return volume, sample

    def _postprocess_and_save(self, preds, sample, output_folder, save_mask=True, save_json=True):
        """
        Post-process the model predictions and save the results.
        Based on the original evaluate_on_test_seg_multiclass function.
        
        Parameters
        ----------
            preds (torch.Tensor): Model predictions with shape (1, 7, D, H, W).
            sample (dict): Sample dictionary containing 'affine' and 'folder'.
            output_folder (str): Folder to save the output results.
            save_mask (bool, optional): Whether to save the predicted mask. Defaults to True.
            save_json (bool, optional): Whether to save the predicted landmarks in JSON. Defaults to True.
        
        Returns
        -------
            np.ndarray: Landmark coordinates in mm space (6, 3).
        """
        preds = preds.cpu().numpy()[0]  # Remove batch dim: (7, D, H, W)
        
        # Create combined mask by taking the class with highest confidence
        combined = np.zeros(preds.shape[1:], dtype=np.uint8)
        confidence_map = np.zeros(preds.shape[1:], dtype=np.float32)
        
        for c in range(1, preds.shape[0]):  # Skip background (class 0)
            # Only update voxels where current prediction is higher than previous ones
            mask = (preds[c] > -1) & ((preds[c] > confidence_map) | (combined == 0))
            confidence_map[mask] = preds[c][mask]
            combined[mask] = c

        # Extract centroids for each landmark class
        centroids_voxel = np.zeros((6, 3), dtype=np.float32)
        all_largest_components = []
        
        for label in range(1, 7):  # Classes 1-6
            binary_mask = (combined == label).astype(np.uint8)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            # Find largest connected component
            labels_cc = cc3d.largest_k(binary_mask, k=1, connectivity=26)
            all_largest_components.append(labels_cc)
            
            # Calculate centroid
            stats = cc3d.statistics(labels_cc)
            if len(stats["centroids"]) > 1:
                centroid = stats["centroids"][1]  # Index 1 is the largest component
                # Convert from (z, y, x) to (x, y, z) order to match original
                centroids_voxel[label - 1] = [centroid[1], centroid[2], centroid[0]]
            else:
                # If no component found, set to zero coordinates
                centroids_voxel[label - 1] = [0, 0, 0]

        # Reconstruct combined mask with only largest components
        combined_largest_components = np.zeros(preds.shape[1:], dtype=np.uint8)
        for i, mask in enumerate(all_largest_components):
            combined_largest_components[mask] = i + 1

        # Convert to mm coordinates (match original: voxel spacing is 0.8mm)
        centroids_mm = centroids_voxel * 0.8
        
        # Handle orientation (match original training script)
        affine = sample["affine"]
        orientation = nib.aff2axcodes(affine)



        if orientation == ('L', 'A', 'S'):
            centroids_mm[:, 1] = -centroids_mm[:, 1]

        # Restore to original origin if affine file exists
        affine_path = os.path.join(sample["folder"], "affine_before_origin_change.txt")
        if os.path.exists(affine_path):
            centroids_mm = restore_centroids_to_original_origin(centroids_mm, affine_path, orientation)

        # Save results if output folder specified
        if output_folder:
            output_folder_img = os.path.join(output_folder, os.path.basename(sample["folder"]))
            os.makedirs(output_folder_img, exist_ok=True)

            if save_json:
                output_json_path = os.path.join(output_folder_img, "F_o.json")
                input_json_path = os.path.join(sample["folder"], "F.json")
                
                # Use the embedded template function that handles missing files
                from template_embedded import load_template_from_embedded
                
                if os.path.exists(input_json_path):
                    # Use original JSON if it exists
                    actualizar_json_con_predicciones(input_json_path, centroids_mm, output_json_path)
                else:
                    # Use embedded template if original doesn't exist
                    print(f"Warning: {input_json_path} not found, using embedded template")
                    data = load_template_from_embedded()
                    
                    # Update with predictions
                    labels = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
                    control_points = data["markups"][0]["controlPoints"]
                    
                    for landmark_idx, label in enumerate(labels):
                        for cp in control_points:
                            if cp["label"] == label:
                                cp["position"] = centroids_mm[landmark_idx].tolist()
                                break
                    
                    # Save the updated JSON
                    import json
                    with open(output_json_path, 'w') as f:
                        json.dump(data, f, indent=4)

            if save_mask:
                # Transpose back to (H, W, D) to match original saving format
                combined_img = np.transpose(combined_largest_components, (1, 2, 0))
                output_mask_path = os.path.join(output_folder_img, "pred_mask.nii.gz")
                nib.save(nib.Nifti1Image(combined_img, affine), output_mask_path)

        return centroids_mm

    def infer_folder(self, folder_path: str, output_folder: str, save_mask=True, save_json=True):
        """
        Perform inference on a folder containing 'cta.nii.gz' and save results.
        
        Parameters
        ----------
            folder_path (str): Path to folder containing 'cta.nii.gz'
            output_folder (str): Path to save results
            save_mask (bool): Whether to save prediction mask
            save_json (bool): Whether to save landmarks JSON
            
        Returns
        -------
            np.ndarray: Landmark coordinates in mm space (6, 3)
        """
        volume, sample = self._prepare_input_folder(folder_path)
        with torch.no_grad():
            preds = self.model(volume)
        
        centroids_mm = self._postprocess_and_save(preds, sample, output_folder, save_mask=save_mask, save_json=save_json)
        print(f"Inference done for: {folder_path}")
        return centroids_mm

    def infer_from_array(self, data, affine=None, output_folder=None, save_mask=False, save_json=False):
        """
        Perform inference on numpy array or nibabel image and return landmarks.
        
        Parameters
        ----------
            data (np.ndarray or nibabel image): CTA data
            affine (np.ndarray, optional): Affine transformation matrix
            output_folder (str, optional): Path to save results (if None, nothing is saved)
            save_mask (bool): Whether to save prediction mask
            save_json (bool): Whether to save landmarks JSON
            
        Returns
        -------
            dict: Dictionary containing:
                - 'landmarks_mm': np.ndarray of shape (6, 3) with landmark coordinates in mm
                - 'landmarks_voxel': np.ndarray of shape (6, 3) with landmark coordinates in voxel space
                - 'raw_predictions': np.ndarray of shape (7, D, H, W) with model output logits
        """
        volume, sample = self._prepare_input_array(data, affine)
        with torch.no_grad():
            preds = self.model(volume)
        
        # Get landmarks in mm space using the same processing as folder method
        centroids_mm = self._postprocess_and_save(preds, sample, output_folder, save_mask=save_mask, save_json=save_json)
        
        # Also extract voxel coordinates before mm conversion
        preds_np = preds.cpu().numpy()[0]  # (7, D, H, W)
        combined = np.zeros(preds_np.shape[1:], dtype=np.uint8)
        confidence_map = np.zeros(preds_np.shape[1:], dtype=np.float32)
        
        for c in range(1, preds_np.shape[0]):
            mask = (preds_np[c] > -1) & ((preds_np[c] > confidence_map) | (combined == 0))
            confidence_map[mask] = preds_np[c][mask]
            combined[mask] = c

        centroids_voxel = np.zeros((6, 3), dtype=np.float32)
        for label in range(1, 7):
            binary_mask = (combined == label).astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            labels_cc = cc3d.largest_k(binary_mask, k=1, connectivity=26)
            stats = cc3d.statistics(labels_cc)
            if len(stats["centroids"]) > 1:
                centroid = stats["centroids"][1]
                centroids_voxel[label - 1] = [centroid[1], centroid[2], centroid[0]]
            else:
                centroids_voxel[label - 1] = [0, 0, 0]
        
        return {
            'landmarks_mm': centroids_mm,
            'landmarks_voxel': centroids_voxel,
            'raw_predictions': preds_np  # Raw model logits
        }

# Add the original JSON update function for compatibility
def actualizar_json_con_predicciones(original_json_path, predichas, output_json_path):
    """
    Original function from training script.
    Reads the original JSON and updates the position of each landmark with the predicted
    coordinates (in mm, in LPS).
    """
    import json
    
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