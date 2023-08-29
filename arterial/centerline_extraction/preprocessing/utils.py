#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import numpy as np

from scipy.ndimage import gaussian_filter

def get_bounding_box_limits_3d(img):
    """
    Computes bounding box (only z axis) of a numpy array (expects an array 
    with zeros as background).

    Parameters
    ----------
    img : numpy.array or array-like object
        3D numpy binary (0, 1) array.

    Returns
    -------
    min_lr : integer
        Lower bound on axis x, LR (in voxel coordinates).
    max_lr : integer
        Upper bound on axis x, LR (in voxel coordinates).
    min_pa : integer
        Lower bound on axis y, PA (in voxel coordinates).
    max_pa : integer
        Upper bound on axis y, PA (in voxel coordinates).
    min_is : integer
        Lower bound on axis z, IS (in voxel coordinates).
    max_is : integer
        Upper bound on axis z, IS (in voxel coordinates).

    """
    axis_left_right = np.any(img, axis=(0, 1))
    axis_posterior_anterior = np.any(img, axis=(0, 2))
    axis_inferior_superior = np.any(img, axis=(1, 2))

    min_lr, max_lr = np.where(axis_left_right)[0][[0, -1]]
    min_pa, max_pa = np.where(axis_posterior_anterior)[0][[0, -1]]
    min_is, max_is = np.where(axis_inferior_superior)[0][[0, -1]]

    return min_lr, max_lr, min_pa, max_pa, min_is, max_is

def patchwise_smoothing(masked_volume_array, patch_shape = (50, 50, 50), max_sigma = 1):
    """
    Applies Gaussian smoothing to a 3D volume in a patchwise manner.

    Parameters
    ----------
    masked_volume_array : numpy.array or array-like object
        3D numpy binary (0, 1) array representing the volume to be smoothed.
    patch_shape : tuple of integers, optional
        The shape of the patches to be used for smoothing. Default is (50, 50, 50).
    max_sigma : float, optional
        The maximum sigma value to be used for Gaussian smoothing. Default is 1.

    Returns
    -------
    smoothed_volume_array : numpy.array
        The smoothed 3D volume.

    """
    
    # Initialize smoothed_volume_array with the same shape as masked_volume_array
    smoothed_volume_array = np.zeros_like(masked_volume_array)
    # Iterate over the volume with strides equal to the patch size
    for i in range(0, masked_volume_array.shape[0], patch_shape[0]):
        for j in range(0, masked_volume_array.shape[1], patch_shape[1]):
            for k in range(0, masked_volume_array.shape[2], patch_shape[2]):
                # Extract the current patch from the volume
                patch = masked_volume_array[i:i+patch_shape[0], j:j+patch_shape[1], k:k+patch_shape[2]]
                # Compute the density of ones in the patch
                density = np.count_nonzero(patch) / patch.size
                # Compute the sigma for Gaussian smoothing
                sigma = max_sigma * density
                # Apply Gaussian smoothing to the patch
                smoothed_patch = gaussian_filter(patch.astype(float), sigma)
                # Binarize the smoothed patch with a threshold of 0.5
                smoothed_patch = np.where(smoothed_patch > 0.1, 1, 0)
                # Assign the smoothed patch back to smoothed_volume_array
                smoothed_volume_array[i:i+patch_shape[0], j:j+patch_shape[1], k:k+patch_shape[2]] = smoothed_patch

    return smoothed_volume_array

def get_compatible_patch_shape(volume_array_shape, desired_patch_shape):
    """
    Computes a patch shape that is compatible with the volume array shape.

    Parameters
    ----------
    volume_array_shape : tuple of integers
        The shape of the volume array.
    desired_patch_shape : tuple of integers
        The desired patch shape.

    Returns
    -------
    compatible_shape : tuple of integers
        The computed patch shape that is compatible with the volume array shape.

    """
    compatible_shape = []
    for v, b in zip(volume_array_shape, desired_patch_shape):
        while v % b != 0:
            b -= 1
        compatible_shape.append(b)
    return tuple(compatible_shape)