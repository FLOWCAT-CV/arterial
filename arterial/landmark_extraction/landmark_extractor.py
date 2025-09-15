#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import numpy as np
import torch
import nibabel as nib
import cc3d
import cv2
import torchio as tio
import tempfile

from utils import (
    update_json_with_predictions,
    restore_centroids_to_original_origin,
    resample_image,
    crop_or_pad_image,
    change_origin_preprocess,
    postprocess_heatmaps,
)
from model import load_trained_model_seg
from arterial.io.load_and_save_operations import save_nifti

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

        # Preprocess in memory
        resampled = resample_image(tio_object, (0.8, 0.8, 0.8))
        cropped_or_padded = crop_or_pad_image(resampled, (320, 320, 480))
        final_image = change_origin_preprocess(cropped_or_padded, (0, 0, 0))

        # Get as nibabel image directly (no disk write)
        img = final_image.as_nibabel()

        # Normalize and shape
        volume = np.clip(img.get_fdata().astype(np.float32), 0, 700) / 700
        volume = torch.tensor(
            np.expand_dims(np.transpose(volume, (2, 0, 1)), axis=0),
            dtype=torch.float32
        )

        return {
            "volume": volume,
            "affine": tio_object.affine,  # original affine for centroids
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

        # Preprocess in memory
        resampled = resample_image(tio_object, (0.8, 0.8, 0.8))
        cropped_or_padded = crop_or_pad_image(resampled, (320, 320, 480))
        final_image = change_origin_preprocess(cropped_or_padded, (0, 0, 0))

        # Get as nibabel image directly (no disk write)
        img = final_image.as_nibabel()

        # Normalize and shape
        volume = np.clip(img.get_fdata().astype(np.float32), 0, 700) / 700
        volume = torch.tensor(
            np.expand_dims(np.transpose(volume, (2, 0, 1)), axis=0),
            dtype=torch.float32
        )

        return {
            "volume": volume,
            "affine": self.original_affine,  # original affine for centroids
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
        volume = sample["volume"].unsqueeze(0).to(self.device)
        return volume, sample

    def _prepare_input_array(self, data, affine=None, temp_folder=None):
        dataset = SingleCTAFromArray(data, affine, temp_folder)
        sample = dataset[0]
        volume = sample["volume"].unsqueeze(0).to(self.device)
        return volume, sample

    def _postprocess_and_save(self, preds, sample, output_folder, save_mask=True, save_json=True):
        """
        Post-process the model predictions and save the results.
        Parameters
        ----------
            preds (torch.Tensor): Model predictions.
            sample (dict): Sample dictionary containing 'affine' and 'folder'.
            output_folder (str): Folder to save the output results.
            save_mask (bool, optional): Whether to save the predicted mask. Defaults to True.
            save_json (bool, optional): Whether to save the predicted landmarks in JSON. Defaults to True.
        
        Returns
        -------
            np.ndarray: Landmark coordinates in mm space (6, 3).
        """
        preds = preds.cpu().numpy()[0]  # (7, D, H, W)
        combined = np.zeros(preds.shape[1:], dtype=np.uint8)
        confidence_map = np.zeros(preds.shape[1:], dtype=np.float32)

        for c in range(1, preds.shape[0]):
            mask = (preds[c] > -1) & ((preds[c] > confidence_map) | (combined == 0))
            confidence_map[mask] = preds[c][mask]
            combined[mask] = c

        centroids_voxel = np.zeros((6, 3), dtype=np.float32)
        all_largest_components = []
        for label in range(1, 7):
            binary_mask = (combined == label).astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            labels_cc = cc3d.largest_k(binary_mask, k=1, connectivity=26)
            all_largest_components.append(labels_cc)
            stats = cc3d.statistics(labels_cc)
            if len(stats["centroids"]) > 1:
                centroid = stats["centroids"][1]
                centroids_voxel[label - 1] = [centroid[1], centroid[2], centroid[0]]

        combined_largest_components = np.zeros(preds.shape[1:], dtype=np.uint8)
        for i, mask in enumerate(all_largest_components):
            combined_largest_components[mask] = i + 1

        centroids_mm = centroids_voxel * 0.6
        affine = sample["affine"]
        orientation = nib.aff2axcodes(affine)

        if orientation == ('L', 'A', 'S'):
            centroids_mm[:, 1] = -centroids_mm[:, 1]

        affine_path = os.path.join(sample["folder"], "affine_before_origin_change.txt")
        if os.path.exists(affine_path):
            centroids_mm = restore_centroids_to_original_origin(centroids_mm, affine_path, orientation)

        if output_folder:
            output_folder_img = os.path.join(output_folder, "results")
            os.makedirs(output_folder_img, exist_ok=True)

            if save_json:
                output_json_path = os.path.join(output_folder_img, "F_o.json")
                input_json_path = os.path.join(sample["folder"], "F.json")
                update_json_with_predictions(centroids_mm, output_json_path, input_json_path)

            if save_mask:
                combined_img = np.transpose(combined_largest_components, (1, 2, 0))
                save_nifti(nib.Nifti1Image(combined_img, affine), os.path.join(output_folder_img, "pred_mask.nii.gz"))

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
                - 'heatmaps': np.ndarray of shape (6, D, H, W) with prediction heatmaps
        """
        volume, sample = self._prepare_input_array(data, affine)
        with torch.no_grad():
            preds = self.model(volume)
        
        # Get landmarks in mm space
        centroids_mm = self._postprocess_and_save(preds, sample, output_folder, save_mask=save_mask, save_json=save_json)
        
        # Also get landmarks in voxel space using the utility function
        centroids_voxel = postprocess_heatmaps(preds.cpu())
        
        # Return both coordinate systems and the raw heatmaps
        return {
            'landmarks_mm': centroids_mm,
            'landmarks_voxel': centroids_voxel,
            'heatmaps': preds.cpu().numpy()[0]  # Remove batch dimension
        }