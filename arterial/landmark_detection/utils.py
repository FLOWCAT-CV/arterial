#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import cc3d

import numpy as np
import torchio as tio
from scipy import ndimage
from skimage.morphology import skeletonize


class SingleCTADataset:
    """
    Dataset class for loading a single CTA for the landmark detection model.

    Parameters
    ----------
    cta_array : np.ndarray
        CTA data.
    cta_affine : np.ndarray
        Affine transformation matrix.
    segmentation_array : np.ndarray, optional
        Vessel segmentation mask. If provided, will be used as second channel.
    """
    def __init__(self, 
                 cta_array, 
                 cta_affine,
                 segmentation_array=None
                 ):
        self.cta_array = cta_array
        self.cta_affine = cta_affine
        self.segmentation_array = segmentation_array
        self.target_spacing = (0.8, 0.8, 0.8)  # mm
        self.target_shape = (320, 320, 480)  # (W, H, D)
        self.clipping_lower_bound = 0
        self.clipping_upper_bound = 700

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Uses TorchIO to resample, crop and pad the CTA to the target shape and spacing and normalize the data.
        Returns 1-channel or 2-channel volume depending on whether segmentation was provided.
        """
        # Process CTA
        tio_img = tio.ScalarImage(tensor=torch.from_numpy(self.cta_array).unsqueeze(0), affine=self.cta_affine)
        tio_img = tio.Resample(self.target_spacing, scalars_only=True)(tio_img)
        tio_img = tio.CropOrPad(self.target_shape)(tio_img)
        
        # Normalize CTA: clip and divide
        cta_volume = torch.clamp(tio_img.data, self.clipping_lower_bound, self.clipping_upper_bound) / self.clipping_upper_bound
        # Transpose to match training: (C, H, W, D) -> (C, D, H, W)
        cta_volume = cta_volume.permute(0, 3, 1, 2).float()
        
        preprocessed_affine = tio_img.affine
        
        # If segmentation provided, process and stack as second channel
        if self.segmentation_array is not None:
            tio_seg = tio.ScalarImage(
                tensor=torch.from_numpy(self.segmentation_array.astype(np.float32)).unsqueeze(0), 
                affine=self.cta_affine
            )
            tio_seg = tio.Resample(self.target_spacing, scalars_only=True, image_interpolation="nearest")(tio_seg)
            tio_seg = tio.CropOrPad(self.target_shape)(tio_seg)
            
            # Binarize segmentation
            seg_volume = (tio_seg.data > 0).float()
            # Transpose to match training: (C, H, W, D) -> (C, D, H, W)
            seg_volume = seg_volume.permute(0, 3, 1, 2)
            
            # Stack: (2, D, H, W)
            volume = torch.cat([cta_volume, seg_volume], dim=0)
        else:
            volume = cta_volume

        return volume, preprocessed_affine


def preprocess_for_landmark_detection(cta_array, cta_affine, segmentation_array=None, device='cpu'):
    """
    Prepare the input array for the model.

    Parameters
    ----------
    cta_array : np.ndarray
        CTA data.
    cta_affine : np.ndarray
        Affine transformation matrix.
    segmentation_array : np.ndarray, optional
        Vessel segmentation mask. If provided, creates 2-channel input.
    device : str
        Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns
    -------
    preprocessed_volume : torch.Tensor
        Preprocessed volume. Shape (1, C, D, H, W) where C=1 or 2.
    preprocessed_affine : np.ndarray
        Affine transformation matrix.
    """
    dataset = SingleCTADataset(cta_array, cta_affine, segmentation_array)
    preprocessed_volume, preprocessed_affine = dataset[0]
    return preprocessed_volume.unsqueeze(0).to(device), preprocessed_affine


def resample_mask_to_original_cta(predicted_mask_array, predicted_mask_affine, cta_array, cta_affine):
    """
    Resample the predicted mask to the original CTA space.

    Parameters
    ----------
    predicted_mask_array : np.ndarray
        Predicted mask.
    predicted_mask_affine : np.ndarray
        Affine transformation matrix of the predicted mask.
    cta_array : np.ndarray
        CTA data.
    cta_affine : np.ndarray
        Affine transformation matrix of the CTA.

    Returns
    -------
    resampled_predicted_mask : np.ndarray
        Resampled predicted mask.
    """
    tio_img = tio.ScalarImage(tensor=torch.from_numpy(predicted_mask_array).unsqueeze(0), affine=predicted_mask_affine)
    target_spacing = np.abs(np.diag(cta_affine)[:3])
    tio_img = tio.Resample(target_spacing, scalars_only=True, image_interpolation="nearest")(tio_img)
    tio_img = tio.CropOrPad(cta_array.shape)(tio_img)
    return tio_img.data.squeeze(0).numpy()


# Reference TICA→MCA distances from GT analysis. Used to flag anatomically
# implausible MCA placements at inference time (no GT required).
# Index 0 = r-mca (r-tica pair), index 1 = l-mca (l-tica pair).
_TICA_MCA_DIST_MEAN = np.array([17.1, 16.1])  # mm
_TICA_MCA_DIST_STD  = np.array([ 6.2,  7.0])  # mm
_MIRROR_MIN_DISP_MM = 5.0


def _apply_mca_mirror_correction(centroids_ras):
    """
    Conditional mirror-reflection correction for MCA landmarks.

    When one MCA has an anatomically implausible TICA→MCA distance and the
    contralateral MCA is healthy, the healthy MCA is reflected across the
    sagittal midplane (estimated from all four bilateral landmarks) to replace
    the bad prediction.

    Landmark index layout (training label order 1-6):
        0=l-tica  1=r-tica  2=l-eica  3=r-eica  4=r-mca  5=l-mca

    Parameters
    ----------
    centroids_ras : np.ndarray, shape (6, 3)
        Landmark coordinates in RAS mm. Modified in-place.

    Returns
    -------
    centroids_ras : np.ndarray
    correction_log : dict  {landmark_name: {"displacement_mm": float}}
    """
    L_TICA, R_TICA, L_EICA, R_EICA, R_MCA, L_MCA = 0, 1, 2, 3, 4, 5

    midplane_x = float(np.mean(centroids_ras[[L_TICA, R_TICA, L_EICA, R_EICA], 0]))

    r_dist = float(np.linalg.norm(centroids_ras[R_MCA] - centroids_ras[R_TICA]))
    l_dist = float(np.linalg.norm(centroids_ras[L_MCA] - centroids_ras[L_TICA]))

    def _bad(dist, i):
        return (dist < _TICA_MCA_DIST_MEAN[i] - 2.0 * _TICA_MCA_DIST_STD[i] or
                dist > _TICA_MCA_DIST_MEAN[i] + 2.0 * _TICA_MCA_DIST_STD[i])

    r_bad, l_bad = _bad(r_dist, 0), _bad(l_dist, 1)
    correction_log = {}

    for bad_idx, good_idx, name, is_bad, contra_bad in [
        (R_MCA, L_MCA, "r-mca", r_bad, l_bad),
        (L_MCA, R_MCA, "l-mca", l_bad, r_bad),
    ]:
        if is_bad and not contra_bad:
            mirrored = np.array([2.0 * midplane_x - centroids_ras[good_idx, 0],
                                 centroids_ras[good_idx, 1],
                                 centroids_ras[good_idx, 2]])
            disp = float(np.linalg.norm(mirrored - centroids_ras[bad_idx]))
            if disp > _MIRROR_MIN_DISP_MM:
                centroids_ras[bad_idx] = mirrored
                correction_log[name] = {"displacement_mm": round(disp, 1)}

    return centroids_ras, correction_log


def postprocess_preds(preds, affine, return_mask=False):
    """
    Postprocess model predictions to extract landmark coordinates.

    Parameters
    ----------
    preds : torch.Tensor
        Model predictions with shape (1, 7, D, H, W).
    affine : np.ndarray
        Affine transformation matrix of the preprocessed volume.
    return_mask : bool
        Whether to return the predicted mask.

    Returns
    -------
    landmarks_ras_mm : np.ndarray
        Landmark coordinates in mm space (6, 3).
    predicted_mask : np.ndarray or None
        Predicted mask. None if return_mask is False.
    """
    preds = preds.cpu().numpy()[0]  # Remove batch dim: (7, D, H, W)

    # Create combined mask by taking the class with highest confidence
    combined = np.zeros(preds.shape[1:], dtype=np.uint8)
    confidence_map = np.zeros(preds.shape[1:], dtype=np.float32)

    for c in range(1, preds.shape[0]):  # Skip background (class 0)
        mask = (preds[c] > -1) & ((preds[c] > confidence_map) | (combined == 0))
        confidence_map[mask] = preds[c][mask]
        combined[mask] = c

    # Detect centroids for each landmark class
    centroids_ijk = np.zeros((6, 3), dtype=np.float32)
    all_largest_components = []

    for label in range(1, 7):  # Classes 1-6
        binary_mask = (combined == label).astype(np.uint8)
        # Apply a 2x2 morphological closing on each (axis 0, axis 1) slice to clean up the mask.
        # This reproduces the previous OpenCV call, which treated axis 2 as channels; OpenCV >= 5
        # rejects 3D inputs, and scipy needs no extra dependency.
        binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((2, 2, 1), dtype=bool)).astype(np.uint8)
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
            centroids_ijk[label - 1] = [0, 0, 0]

    # Convert to mm coordinates with affine from preprocessed volume
    centroids_ras = np.array([ijk_to_ras(centroids_ijk[i], affine) for i in range(6)])

    # MCA mirror correction: fixes hemisphere confusion when one MCA is
    # placed on the wrong side (detected via anatomical distance plausibility).
    centroids_ras, _ = _apply_mca_mirror_correction(centroids_ras)

    if return_mask:
        combined_largest_components = np.zeros(preds.shape[1:], dtype=np.uint8)
        for i, mask in enumerate(all_largest_components):
            combined_largest_components += mask * (i + 1)
        combined_largest_components = combined_largest_components.transpose(1, 2, 0)
        return centroids_ras, combined_largest_components
    else:
        return centroids_ras, None


def ras_to_ijk(coordinates_ras, affine):
    """Convert RAS coordinates to IJK coordinates."""
    return np.dot(np.linalg.inv(affine), np.append(coordinates_ras, 1))[:3]


def ijk_to_ras(coordinates_ijk, affine):
    """Convert IJK coordinates to RAS coordinates."""
    return np.dot(affine, np.append(coordinates_ijk, 1))[:3]


# =============================================================================
# LANDMARK REFINEMENT FUNCTIONS
# =============================================================================

def refine_landmarks_with_segmentation(landmarks_dict, segmentation_array, affine, 
                                        method='adaptive', search_radius_mm=5.0):
    """
    Refine landmark positions using the vessel segmentation mask.
    
    Snaps landmarks to the vessel centerline or bifurcation points.

    Parameters
    ----------
    landmarks_dict : dict
        Dictionary of landmark names to RAS coordinates.
    segmentation_array : np.ndarray
        Binary vessel segmentation mask.
    affine : np.ndarray
        Affine transformation matrix.
    method : str
        Refinement method: 'centerline', 'bifurcation', 'adaptive'.
        'adaptive' uses bifurcation for eica landmarks, centerline for others.
    search_radius_mm : float
        Maximum search radius in mm.

    Returns
    -------
    refined_landmarks : dict
        Dictionary of refined landmark coordinates.
    refinement_stats : dict
        Statistics about the refinement (displacements).
    """
    refiner = _LandmarkRefiner(landmarks_dict, segmentation_array, affine)
    refined_landmarks = refiner.refine_all(method=method, search_radius_mm=search_radius_mm)
    return refined_landmarks, refiner.get_stats()


class _LandmarkRefiner:
    """Internal class for landmark refinement."""
    
    def __init__(self, landmarks_dict, segmentation_array, affine):
        self.original_landmarks = landmarks_dict.copy()
        self.segmentation = (segmentation_array > 0).astype(np.uint8)
        self.affine = affine
        self.voxel_size = np.abs(np.diag(affine)[:3])
        
        self._centerline = None
        self._centerline_points = None
        self._bifurcation_points = None
        
        self.refined_landmarks = {}
        self.stats = {}
    
    @property
    def centerline(self):
        if self._centerline is None:
            cleaned = cc3d.largest_k(self.segmentation, k=10, connectivity=26)
            cleaned = (cleaned > 0).astype(np.uint8)
            self._centerline = skeletonize(cleaned).astype(np.uint8)
        return self._centerline
    
    @property
    def centerline_points(self):
        if self._centerline_points is None:
            self._centerline_points = np.array(np.where(self.centerline > 0)).T
        return self._centerline_points
    
    @property
    def bifurcation_points(self):
        if self._bifurcation_points is None:
            kernel = np.ones((3, 3, 3), dtype=np.uint8)
            kernel[1, 1, 1] = 0
            neighbor_count = ndimage.convolve(self.centerline, kernel, mode='constant', cval=0)
            bifurc_mask = (self.centerline > 0) & (neighbor_count > 2)
            coords = np.array(np.where(bifurc_mask)).T
            self._bifurcation_points = self._cluster_points(coords, 3.0) if len(coords) > 0 else np.array([]).reshape(0, 3)
        return self._bifurcation_points
    
    def _cluster_points(self, points, min_dist_mm):
        if len(points) == 0:
            return np.array([])
        min_dist_vox = min_dist_mm / np.mean(self.voxel_size)
        clustered = []
        used = np.zeros(len(points), dtype=bool)
        for i, p in enumerate(points):
            if used[i]:
                continue
            dists = np.linalg.norm(points - p, axis=1)
            cluster_mask = dists < min_dist_vox
            used[cluster_mask] = True
            clustered.append(np.mean(points[cluster_mask], axis=0))
        return np.array(clustered)
    
    def refine_all(self, method='adaptive', search_radius_mm=5.0):
        bifurc_landmarks = ['l-eica', 'r-eica']
        
        for name, coords in self.original_landmarks.items():
            if coords is None or not isinstance(coords, (list, tuple)) or len(coords) != 3:
                self.refined_landmarks[name] = coords
                continue
            
            coords_ijk = ras_to_ijk(np.array(coords), self.affine)
            
            if method == 'adaptive':
                if name in bifurc_landmarks:
                    refined_ijk, dist = self._snap_to_bifurcation(coords_ijk, search_radius_mm * 1.5)
                else:
                    refined_ijk, dist = self._snap_to_centerline(coords_ijk, search_radius_mm)
            elif method == 'bifurcation':
                refined_ijk, dist = self._snap_to_bifurcation(coords_ijk, search_radius_mm)
            else:  # centerline
                refined_ijk, dist = self._snap_to_centerline(coords_ijk, search_radius_mm)
            
            refined_ras = ijk_to_ras(refined_ijk, self.affine).tolist()
            self.refined_landmarks[name] = tuple(refined_ras)
            self.stats[name] = {'displacement_mm': dist, 'method': method}
        
        return self.refined_landmarks
    
    def _snap_to_centerline(self, coords_ijk, search_radius_mm):
        if len(self.centerline_points) == 0:
            return coords_ijk, 0.0
        
        dists = np.linalg.norm(self.centerline_points - coords_ijk, axis=1)
        min_idx = np.argmin(dists)
        min_dist_mm = dists[min_idx] * np.mean(self.voxel_size)
        
        if min_dist_mm > search_radius_mm:
            return coords_ijk, 0.0
        
        return self.centerline_points[min_idx], min_dist_mm
    
    def _snap_to_bifurcation(self, coords_ijk, search_radius_mm):
        if len(self.bifurcation_points) == 0:
            return self._snap_to_centerline(coords_ijk, search_radius_mm)
        
        dists = np.linalg.norm(self.bifurcation_points - coords_ijk, axis=1)
        min_idx = np.argmin(dists)
        min_dist_mm = dists[min_idx] * np.mean(self.voxel_size)
        
        if min_dist_mm > search_radius_mm:
            return self._snap_to_centerline(coords_ijk, search_radius_mm)
        
        return self.bifurcation_points[min_idx], min_dist_mm
    
    def get_stats(self):
        if not self.stats:
            return {}
        displacements = [s['displacement_mm'] for s in self.stats.values()]
        return {
            'mean_displacement_mm': float(np.mean(displacements)),
            'max_displacement_mm': float(np.max(displacements)),
            'per_landmark': self.stats
        }
