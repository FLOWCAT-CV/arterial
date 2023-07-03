#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_laplace
from skimage.measure import label

def slice_cta_head_and_neck(case_dir):
    """
    This funciton enables slicing of head and neck parts of the CTA ({case_id}.nii.gz)
    by using a Laplacian of Gaussian filter (scipy) to perform a segmentation
    of the cranium. That information is used to slice the original CTA
    into two parts: the head CTA and neck CTA.

    This function generates two additional nifti files:

    >>> case_dir/{case_id}_head.nii.gz
    >>> case_dir/{case_id}_neck.nii.gz

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
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
    
    # Load nifti of full CTA
    cta_nifti = nib.load(os.path.join(case_dir, "{}.nii.gz".format(os.path.basename(case_dir))))
    cta_array = cta_nifti.get_fdata()
    
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
    head_cta_array = cta_array[lower_slicing_i_coordinate:upper_slicing_i_coordinate, lower_slicing_j_coordinate:upper_slicing_j_coordinate, lower_slicing_k_coordinate:upper_slicing_k_coordinate]
    # For the neck (lower part of the image) we add some extra slices to have some overlap
    # This should smooth edge effects upon merge after separate segmentation
    neck_cta_array = cta_array[:, :, : int(1.1 * lower_slicing_k_coordinate)]
    
    # Update affine and header for head CTA (neck will be fine, 
    # header["dim"] updates automatically when creating the new nifti object)
    head_affine = cta_nifti.affine.copy()
    # Get s voxel size
    r_voxel_size = head_affine[0, 0]
    a_voxel_size = head_affine[1, 1]
    s_voxel_size = head_affine[2, 2]
    # Update s translation from affine
    head_affine[0, 3] += lower_slicing_i_coordinate * r_voxel_size
    head_affine[1, 3] += lower_slicing_j_coordinate * a_voxel_size
    head_affine[2, 3] += lower_slicing_k_coordinate * s_voxel_size
    
    # Generate new nifti files
    head_cta_nifti = nib.Nifti1Image(head_cta_array, head_affine)
    head_cta_nifti.set_data_dtype(np.int16)
    neck_cta_nifti = nib.Nifti1Image(neck_cta_array, cta_nifti.affine)
    neck_cta_nifti.set_data_dtype(np.int16)
    # Save new nifti files
    nib.save(head_cta_nifti, os.path.join(case_dir, "{}_head.nii.gz".format(os.path.basename(case_dir))))
    nib.save(neck_cta_nifti, os.path.join(case_dir, "{}_neck.nii.gz".format(os.path.basename(case_dir))))

def join_head_and_neck_segmentations(case_dir):
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
    # Load original CTA
    cta_nifti = nib.load(os.path.join(case_dir, "{}.nii.gz".format(os.path.basename(case_dir))))
    cta_shape = cta_nifti.get_fdata().shape
    # Load head segmentation
    head_segmentation_nifti = nib.load(os.path.join(case_dir, "{}_head_segmentation.nii.gz".format(os.path.basename(case_dir))))
    head_segmentation_array_reduced = head_segmentation_nifti.get_fdata()
    head_header = head_segmentation_nifti.header
    # Load neck segmentation
    neck_segmentation_nifti = nib.load(os.path.join(case_dir, "{}_neck_segmentation.nii.gz".format(os.path.basename(case_dir))))
    neck_segmentation_array = neck_segmentation_nifti.get_fdata()
    neck_header = neck_segmentation_nifti.header
    
    r_voxel_size, a_voxel_size, s_voxel_size = cta_nifti.affine[0, 0], cta_nifti.affine[1, 1], cta_nifti.affine[2, 2]
    
    head_origin_i = int((head_header["qoffset_x"] - neck_header["qoffset_x"]) / r_voxel_size)
    head_origin_j = int((head_header["qoffset_y"] - neck_header["qoffset_y"]) / a_voxel_size)
    head_origin_k = int((head_header["qoffset_z"] - neck_header["qoffset_z"]) / s_voxel_size)
    
    # Set head_segmentation_array_reduced into original shape
    head_segmentation_array = np.zeros([cta_shape[0], cta_shape[1], cta_shape[2] - head_origin_k])
    head_segmentation_array[head_origin_i:head_origin_i + head_segmentation_array_reduced.shape[0], 
                            head_origin_j:head_origin_j + head_segmentation_array_reduced.shape[1], 
                            :head_segmentation_array_reduced.shape[2]] = head_segmentation_array_reduced
    
    # Initialize dice list to get slice with maximum similarity (smoothest transition)
    dice = []
    # Check all slices in the middle for the one with the highest similarity in terms of Dice coefficient
    for idx in range(neck_segmentation_array.shape[2] - head_origin_k):
        dice.append(compute_dice(neck_segmentation_array[:, :, head_origin_k + idx], head_segmentation_array[:, :, idx]))

    # Get s coordinate for highest similarity
    slice_difference = np.argmax(dice)
    highest_similarity_k_coordinate = head_origin_k + slice_difference

    # Initialize final segmentation array
    segmentation_array = np.zeros_like(cta_nifti.get_fdata())
    # Add neck CTA segmentation to upper part of the image
    segmentation_array[:, :, highest_similarity_k_coordinate:] = head_segmentation_array[:, :, slice_difference:]
    # Add head CTA segmentation to lower part of the image
    segmentation_array[:, :, :highest_similarity_k_coordinate] = neck_segmentation_array[:, :, :highest_similarity_k_coordinate]

    # Generate new nifti file for segmentation
    segmentation_nifti = nib.Nifti1Image(segmentation_array, cta_nifti.affine, cta_nifti.header)
    # Save new nifti files
    nib.save(segmentation_nifti, os.path.join(case_dir, "{}_segmentation.nii.gz".format(os.path.basename(case_dir))))