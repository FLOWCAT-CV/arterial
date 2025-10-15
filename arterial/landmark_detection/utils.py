#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import torch
import cc3d
import cv2

import numpy as np
import torchio as tio

class SingleCTADataset:
    """
    Dataset class for loading a single CTA for the landmark detection model.

    Parameters
    ----------
    cta_array: np.ndarray
        CTA data.
    cta_affine: np.ndarray
        Affine transformation matrix.

    """
    def __init__(self, 
                 cta_array, 
                 cta_affine
                 ):
        self.cta_array = cta_array
        self.cta_affine = cta_affine
        self.target_spacing = (0.8, 0.8, 0.8) # mm
        self.target_shape = (320, 320, 480) # (W, H, D)
        self.clipping_lower_bound = 0
        self.clipping_upper_bound = 700

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Uses TorchIO to resample, crop and pad the CTA to the target shape and spacing and normalize the data.
        """
        tio_img = tio.ScalarImage(tensor=torch.from_numpy(self.cta_array).unsqueeze(0), affine=self.cta_affine)
        # Resamples the CTA to the target spacing
        tio_img = tio.Resample(self.target_spacing, scalars_only=True)(tio_img)
        # Crops and pads the CTA to the target shape
        tio_img = tio.CropOrPad(self.target_shape)(tio_img)
        # Normalize: clip and divide the CTA intensity to the target range 
        # TODO: perhaps this should be adjusted to a more restrisctive window (usually, I use W=300, L=150, which would be equivalent of changing the upper bound to 300)
        volume = torch.clamp(tio_img.data, self.clipping_lower_bound, self.clipping_upper_bound) / self.clipping_upper_bound
        # Transpose to match training: (C, H, W, D) -> (C, D, H, W)
        volume = volume.permute(0, 3, 1, 2)
        volume = volume.float()

        return volume, tio_img.affine

def preprocess_for_landmark_detection(cta_array, cta_affine, device='cpu'):
    """
    Prepare the input array for the model.

    Parameters
    ----------
    cta_array: np.ndarray
        CTA data.
    cta_affine: np.ndarray
        Affine transformation matrix.
    device: str
        Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns
    -------
    preprocessed_volume: torch.Tensor
        Preprocessed CTA volume.
    preprocessed_affine: np.ndarray
        Affine transformation matrix.

    """
    dataset = SingleCTADataset(cta_array, cta_affine)
    preprocessed_volume, preprocessed_affine = dataset[0]
    return preprocessed_volume.unsqueeze(0).to(device), preprocessed_affine

def resample_mask_to_original_cta(predicted_mask_array, predicted_mask_affine, cta_array, cta_affine):
    """
    Resample the predicted mask to the original CTA space. Reverses operations performed in preprocess_for_landmark_detection.

    Parameters
    ----------
    predicted_mask_array: np.ndarray
        Predicted mask.
    predicted_mask_affine: np.ndarray
        Affine transformation matrix of the predicted mask.
    cta_array: np.ndarray
        CTA data.
    cta_affine: np.ndarray
        Affine transformation matrix of the CTA.

    Returns
    -------
    resampled_predicted_mask: np.ndarray
        Resampled predicted mask.

    """
    tio_img = tio.ScalarImage(tensor=torch.from_numpy(predicted_mask_array).unsqueeze(0), affine=predicted_mask_affine)
    target_spacing = np.abs(np.diag(cta_affine)[:3])
    tio_img = tio.Resample(target_spacing, scalars_only=True, image_interpolation = "nearest")(tio_img)
    tio_img = tio.CropOrPad(cta_array.shape)(tio_img)
    return tio_img.data.squeeze(0).numpy()

def postprocess_preds(preds, affine, return_mask=False):
    """

    Parameters
    ----------
    preds: torch.Tensor
        Model predictions with shape (1, 7, D, H, W).
    affine: np.ndarray
        Original affine transformation matrix of the CTA image. Used for orientation only.
    return_mask: bool
        Whether to return the predicted mask.

    Returns
    -------
    landmarks_ras_mm: np.ndarray
        Landmark coordinates in mm space (6, 3).
    predicted_mask: np.ndarray, None
        Predicted mask. None if return_mask is False.
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

    # detect centroids for each landmark class
    centroids_ijk = np.zeros((6, 3), dtype=np.float32)
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
            centroids_ijk[label - 1] = [centroid[1], centroid[2], centroid[0]]
        else:
            # If no component found, set to zero coordinates
            centroids_ijk[label - 1] = [0, 0, 0]
    
    # Convert to mm coordinates with affine from preprocessed volume
    centroids_ras = np.array([ijk_to_ras(centroids_ijk[i], affine) for i in range(6)])

    # # Reconstruct combined mask with only largest components
    if return_mask:
        combined_largest_components = np.zeros(preds.shape[1:], dtype=np.uint8)
        for i, mask in enumerate(all_largest_components):
            combined_largest_components += mask * (i + 1)
        combined_largest_components = combined_largest_components.transpose(1, 2, 0)
        return centroids_ras, combined_largest_components
    else:   
        return centroids_ras, None

def ras_to_ijk(coordinates_ras, affine):
    """
    Convert RAS coordinates to IJK coordinates.

    Parameters
    ----------
    coordinates_ras: np.ndarray
        RAS coordinates.    
    affine: np.ndarray
        Affine transformation matrix.

    Returns
    -------
    coordinates_ijk: np.ndarray
        IJK coordinates.
        
    """
    return np.dot(np.linalg.inv(affine), np.append(coordinates_ras, 1))[:3]

def ijk_to_ras(coordinates_ijk, affine):
    """
    Convert IJK coordinates to RAS coordinates.

    Parameters
    ----------
    coordinates_ijk: np.ndarray
        IJK coordinates.
    affine: np.ndarray
        Affine transformation matrix.

    Returns
    -------
    coordinates_ras: np.ndarray
        RAS coordinates.

    """
    return np.dot(affine, np.append(coordinates_ijk, 1))[:3]