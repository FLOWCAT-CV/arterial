#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_laplace
from skimage.measure import label

def get_largest_connected_component(segmentation):
    """
    This function is used to filter the segmentation's smaller components
    to leave the largest component alone.

    Parameters
    ----------
    segmentation : numpy.ndarray or array-like object
        Numpy array with a binary mask.
    
    Returns
    -------
    largest_connected_component : numpy.ndarray or array-like object
        Numpy array with a binary mask of the largest component.

    """
    # Converts the segmentation to different labels in case there 
    # is more than one connected component
    labels = label(segmentation)
    # Assert there is at least one connected component
    assert(labels.max() != 0) 
    # Get largest component
    largest_connected_component = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largest_connected_component  

def slice_cta_head_and_neck(cta_array, cta_affine):
    """
    This funciton enables slicing of head and neck parts of the CTA ({case_id}.nii.gz)
    by using a Laplacian of Gaussian filter (scipy) to perform a segmentation
    of the cranium. That information is used to slice the original CTA
    into two parts: the head CTA and neck CTA.

    Parameters
    ----------
    cta_array : numpy.ndarray or array-like object
        Numpy array with the CTA image.
    cta_affine : numpy.ndarray
        Affine matrix of the CTA.

    Returns
    -------
    cta_head_array : nibabel.nifti1.Nifti1Image
        Nifti object of the head CTA.
    cta_neck_array : nibabel.nifti1.Nifti1Image
        Nifti object of the neck CTA.
    cta_head_affine : numpy.ndarray
        Affine matrix of the head CTA.

    """    
    # Apply laplacian-gaussian filter to upper half of the CTA image
    half_s_coordinate = cta_array.shape[2] // 2
    upper_half_cta_array = cta_array[:, :, half_s_coordinate:]    
    filtered_upper_half_cta_array = gaussian_laplace(upper_half_cta_array, sigma = 0.0001, mode = "nearest")

    # Get cranium binary mask
    tolerance = 0.43 * np.ptp(filtered_upper_half_cta_array)
    threshold = np.min(filtered_upper_half_cta_array) + tolerance
    cranium_mask = np.where(filtered_upper_half_cta_array <= threshold, np.max(cta_array), 0)
    # Get largest connected component
    cranium_mask = get_largest_connected_component(cranium_mask)
    # Get lowest coordinate with a non-zero voxel from cranium mask 
    nonzero_coordinates = np.nonzero(cranium_mask)
    # Get s coordinate for slicing into head and neck
    lower_slicing_i_coordinate = min(nonzero_coordinates[0])
    upper_slicing_i_coordinate = max(nonzero_coordinates[0])
    lower_slicing_j_coordinate = min(nonzero_coordinates[1])
    upper_slicing_j_coordinate = max(nonzero_coordinates[1])
    lower_slicing_k_coordinate = half_s_coordinate + min(nonzero_coordinates[2])
    upper_slicing_k_coordinate = half_s_coordinate + max(nonzero_coordinates[2])
    
    # Slice cta into two (head and neck)
    cta_head_array = cta_array[lower_slicing_i_coordinate:upper_slicing_i_coordinate, 
                               lower_slicing_j_coordinate:upper_slicing_j_coordinate, 
                               lower_slicing_k_coordinate:upper_slicing_k_coordinate]
    
    # For the neck (lower part of the image) we add some extra slices to have some overlap
    # This should smooth edge effects upon merge after separate segmentation
    cta_neck_array = cta_array[:, :, : int(1.1 * lower_slicing_k_coordinate)]
    
    # Update affine and header for head CTA (neck will be fine, 
    # header["dim"] updates automatically when creating the new nifti object)
    cta_head_affine = cta_affine.copy()
    # Get s voxel size
    r_voxel_size = cta_head_affine[0, 0]
    a_voxel_size = cta_head_affine[1, 1]
    s_voxel_size = cta_head_affine[2, 2]
    # Update translation from affine
    cta_head_affine[0, 3] += lower_slicing_i_coordinate * r_voxel_size
    cta_head_affine[1, 3] += lower_slicing_j_coordinate * a_voxel_size
    cta_head_affine[2, 3] += lower_slicing_k_coordinate * s_voxel_size

    return cta_head_array, cta_neck_array, cta_head_affine

def join_head_and_neck_segmentations(cta_array, cta_affine, segmentation_head_array, segmentation_neck_array, head_affine):
    """
    Joins segmentation niftis for head and neck. Since there is an overlap across 
    segmentations, it selects the overlapping slice with the highest Dice similarity
    to join both segmentations.

    It generates an additional nifti file:

    >>> case_dir/{os.path.basename(case_dir)}_segmentation.nii.gz

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    def compute_dice(a, b):
        """
        Computes Dice coefficients between two (assumed) binary arrays.

        Parameters
        ----------
        a : numpy.ndarray or array-like object
            First slice.
        b : numpy.ndarray or array-like object
            Second slice.

        Returns
        -------
        dice : float
            Dice coefficient.

        """
        return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
    cta_shape = cta_array.shape
    
    r_voxel_size, a_voxel_size, s_voxel_size = head_affine[0, 0], head_affine[1, 1], head_affine[2, 2]
    
    head_origin_i = int((head_affine[0, -1] - cta_affine[0, -1]) / r_voxel_size)
    head_origin_j = int((head_affine[1, -1] - cta_affine[1, -1]) / a_voxel_size)
    head_origin_k = int((head_affine[2, -1] - cta_affine[2, -1]) / s_voxel_size)
    
    # Set head_segmentation_array_reduced into original shape
    segmentation_head_array_ = np.zeros([cta_shape[0], cta_shape[1], cta_shape[2] - head_origin_k])
    segmentation_head_array_[head_origin_i:head_origin_i + segmentation_head_array.shape[0], 
                             head_origin_j:head_origin_j + segmentation_head_array.shape[1], 
                             :segmentation_head_array.shape[2]] = segmentation_head_array
    
    # Initialize dice list to get slice with maximum similarity (smoothest transition)
    dice = []
    # Check all slices in the middle for the one with the highest similarity in terms of Dice coefficient
    for idx in range(segmentation_neck_array.shape[2] - head_origin_k):
        dice.append(compute_dice(segmentation_neck_array[:, :, head_origin_k + idx], segmentation_head_array_[:, :, idx]))

    # Get s coordinate for highest similarity
    slice_difference = np.argmax(dice)
    highest_similarity_k_coordinate = head_origin_k + slice_difference

    # Initialize final segmentation array
    segmentation_array = np.zeros_like(cta_array)
    # Add neck CTA segmentation to upper part of the image
    segmentation_array[:, :, highest_similarity_k_coordinate:] = segmentation_head_array_[:, :, slice_difference:]
    # Add head CTA segmentation to lower part of the image
    segmentation_array[:, :, :highest_similarity_k_coordinate] = segmentation_neck_array[:, :, :highest_similarity_k_coordinate]

    segmentation_nifti = nib.Nifti1Image(segmentation_array, cta_affine)

    return segmentation_nifti, segmentation_array