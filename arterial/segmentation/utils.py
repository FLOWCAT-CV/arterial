#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_laplace
from skimage.measure import label, regionprops

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
    
    # Generate new nifti files
    # cta_head_array = nib.Nifti1Image(cta_head_array, cta_head_affine)
    # cta_neck_array = nib.Nifti1Image(cta_neck_array, cta_affine, cta_nifti.header)

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

# def crop_intracranial_cta(case_dir):
#     """
#     This funciton enables slicing of head and neck parts of the CTA ({case_id}.nii.gz)
#     by using a Laplacian of Gaussian filter (scipy) to perform a segmentation
#     of the cranium. That information is used to slice the original CTA
#     into two parts: the head CTA and neck CTA.

#     This function generates two additional nifti files:

#     >>> case_dir/{case_id}_cta_intracranial.nii.gz

#     Parameters
#     ----------
#     case_dir : string or path-like object
#         Path to case directory. 

#     Returns
#     -------

#     """    
#     # Load nifti of full CTA
#     cta_nifti = nib.load(os.path.join(case_dir, "{}_cta.nii.gz".format(os.path.basename(case_dir))))
#     cta_array = cta_nifti.get_fdata()

#     # Get height. If it is shorter than 200 mm, we consider that it is a head CTA and we do not crop
#     height = cta_array.shape[2] * cta_nifti.header.get_zooms()[2]

#     if abs(height) > 200:
#         # Apply laplacian-gaussian filter to upper half of the CTA image
#         half_s_coordinate = cta_array.shape[2] // 2
#         upper_half_cta_array = cta_array[:, :, half_s_coordinate:]    
#         filtered_upper_half_cta_array = gaussian_laplace(upper_half_cta_array, sigma = 0.0001, mode = "nearest")

#         # Get cranium binary mask
#         tolerance = 0.43 * np.ptp(filtered_upper_half_cta_array)
#         threshold = np.min(filtered_upper_half_cta_array) + tolerance
#         cranium_mask = np.where(filtered_upper_half_cta_array <= threshold, np.max(cta_array), 0)
#         # Get largest connected component
#         cranium_mask = get_largest_connected_component(cranium_mask)
#         # Get lowest coordinate with a non-zero voxel from cranium mask 
#         nonzero_coordinates = np.nonzero(cranium_mask)

#         # We will define a flag here so that, in case that we detect a bad cropping, we just select hardcoded coordinates for the CTA cropping
#         bad_cropping = False
#         # In these events, the cranium segmentation was not successful, we will just cut the upper half of the CTA
#         if len(nonzero_coordinates[0]) == 1 or len(nonzero_coordinates[1]) == 1 or len(nonzero_coordinates[2]) == 1:
#             bad_cropping = True
#         if abs(max(nonzero_coordinates[0]) - min(nonzero_coordinates[0])) < 100 or abs(max(nonzero_coordinates[1]) - min(nonzero_coordinates[1])) < 100 or abs(max(nonzero_coordinates[2]) - min(nonzero_coordinates[2])) < 100:
#             bad_cropping = True
        
#         if bad_cropping:
#             lower_slicing_i_coordinate = 50
#             upper_slicing_i_coordinate = cta_array.shape[0] - 50
#             lower_slicing_j_coordinate = 50
#             upper_slicing_j_coordinate = cta_array.shape[1] - 50
#             lower_slicing_k_coordinate = half_s_coordinate
#             upper_slicing_k_coordinate = cta_array.shape[2]
#         else:
#             # Get s coordinate for slicing into head and neck
#             lower_slicing_i_coordinate = min(nonzero_coordinates[0])
#             upper_slicing_i_coordinate = max(nonzero_coordinates[0])
#             lower_slicing_j_coordinate = min(nonzero_coordinates[1])
#             upper_slicing_j_coordinate = max(nonzero_coordinates[1])
#             lower_slicing_k_coordinate = half_s_coordinate + min(nonzero_coordinates[2])
#             upper_slicing_k_coordinate = half_s_coordinate + max(nonzero_coordinates[2])

#         # Slice cta into two (head and neck)
#         intracranial_cta_array = cta_array[lower_slicing_i_coordinate:upper_slicing_i_coordinate, 
#                                            lower_slicing_j_coordinate:upper_slicing_j_coordinate, 
#                                            lower_slicing_k_coordinate:upper_slicing_k_coordinate]
        
#         # Update affine and header for intracranial CTA
#         intracranial_affine = cta_nifti.affine.copy()
#         # Get s voxel size
#         r_voxel_size = intracranial_affine[0, 0]
#         a_voxel_size = intracranial_affine[1, 1]
#         s_voxel_size = intracranial_affine[2, 2]
#         # Update s translation from affine
#         intracranial_affine[0, 3] += lower_slicing_i_coordinate * r_voxel_size
#         intracranial_affine[1, 3] += lower_slicing_j_coordinate * a_voxel_size
#         intracranial_affine[2, 3] += lower_slicing_k_coordinate * s_voxel_size
    
#         # Generate new nifti files
#         intracranial_cta_nifti = nib.Nifti1Image(intracranial_cta_array, intracranial_affine)
#         # Save new nifti files
#         nib.save(intracranial_cta_nifti, os.path.join(case_dir, "{}_cta_intracranial.nii.gz".format(os.path.basename(case_dir))))
#     else:
#         nib.save(cta_nifti, os.path.join(case_dir, "{}_cta_intracranial.nii.gz".format(os.path.basename(case_dir))))


# def recenter_thrombus_patch(case_dir):
#     """
#     This function re-centers the thrombus patch in the CTA and NCCT nifties. It loads the predicted thrombus,
#     gets the centroid of the thrombus, and then creates new patches for CTA and NCCT based on the thrombus centroid.
#     If no thrombus is found, it prints a message indicating so.

#     This function ovverwrites two additional nifti files:

#     >>> case_dir/{case_id}_cta_thrombus_patch.nii.gz
#     >>> case_dir/{case_id}_ncct_thrombus_patch.nii.gz

#     Parameters
#     ----------
#     case_dir : string or path-like object
#         Path to case directory. 

#     Returns
#     -------

#     """
#     # Load CTA patch for patch shape (assumed to be divisible by 2)
#     cta_patch = nib.load(os.path.join(case_dir, "{}_cta_thrombus_patch.nii.gz".format(os.path.basename(case_dir))))
#     affine_patch = cta_patch.affine
    
#     # Load CTA and NCCT nifties
#     cta_nifti = nib.load(os.path.join(case_dir, "{}_cta.nii.gz".format(os.path.basename(case_dir))))
#     cta_nifti_data = cta_nifti.get_fdata()
#     ncct_nifti = nib.load(os.path.join(case_dir, "{}_ncct.nii.gz".format(os.path.basename(case_dir))))
#     ncct_nifti_data = ncct_nifti.get_fdata()

#     # Load predicted thrombus
#     thrombus_segmentation_nifti = nib.load(os.path.join(case_dir, '{}_thrombus_segmentation.nii.gz'.format(os.path.basename(case_dir))))
#     thrombus_segmentation = thrombus_segmentation_nifti.get_fdata()

#     # Get thrombus centroid (largest component)
#     if not thrombus_segmentation.sum() == 0:
#         label_mask = label(thrombus_segmentation.astype(np.int))
#         properties = regionprops(label_mask)
#         thrombus_centroid = properties[0].centroid
        
#         # Get new patches for cta and ncct
#         cta_patch_data = cta_nifti_data[int(thrombus_centroid[0] - cta_patch.shape[0] / 2):int(thrombus_centroid[0] + cta_patch.shape[0] / 2),
#                                         int(thrombus_centroid[1] - cta_patch.shape[1] / 2):int(thrombus_centroid[1] + cta_patch.shape[1] / 2),
#                                         int(thrombus_centroid[2] - cta_patch.shape[2] / 2):int(thrombus_centroid[2] + cta_patch.shape[2] / 2)]
#         ncct_patch_data = ncct_nifti_data[int(thrombus_centroid[0] - cta_patch.shape[0] / 2):int(thrombus_centroid[0] + cta_patch.shape[0] / 2),
#                                           int(thrombus_centroid[1] - cta_patch.shape[1] / 2):int(thrombus_centroid[1] + cta_patch.shape[1] / 2),
#                                           int(thrombus_centroid[2] - cta_patch.shape[2] / 2):int(thrombus_centroid[2] + cta_patch.shape[2] / 2)]
        
#         # Save new patches
#         cta_patch = nib.Nifti1Image(cta_patch_data, affine_patch)
#         ncct_patch = nib.Nifti1Image(ncct_patch_data, affine_patch)
#         nib.save(cta_patch, os.path.join(case_dir, "{}_cta_thrombus_patch.nii.gz".format(os.path.basename(case_dir))))
#         nib.save(ncct_patch, os.path.join(case_dir, "{}_ncct_thrombus_patch.nii.gz".format(os.path.basename(case_dir))))

#     else:
#         print("No thrombus was found in the segmentation.")