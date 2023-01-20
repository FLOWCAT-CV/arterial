#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import numpy as np

from scipy.optimize import curve_fit

def get_bounding_box_limits_3d(img):
    ''' 
    Computes bounding box (only z axis) of a numpy array (expects an array 
    with zeros as background).

    Parameters
    ----------
    img : numpy.array or array-like object
        3D numpy binary (0, 1) array.

    Returns
    -------
    minLR : integer
        Lower bound on axis x, LR (in voxel coordinates).
    maxLR : integer
        Upper bound on axis x, LR (in voxel coordinates).
    minPA : integer
        Lower bound on axis y, PA (in voxel coordinates).
    maxPA : integer
        Upper bound on axis y, PA (in voxel coordinates).
    minIS : integer
        Lower bound on axis z, IS (in voxel coordinates).
    maxIS : integer
        Upper bound on axis z, IS (in voxel coordinates).

    '''
    axis_left_right = np.any(img, axis=(0, 1))
    axis_posterior_anterior = np.any(img, axis=(0, 2))
    axis_inferior_superior = np.any(img, axis=(1, 2))

    min_lr, max_lr = np.where(axis_left_right)[0][[0, -1]]
    min_pa, max_pa = np.where(axis_posterior_anterior)[0][[0, -1]]
    min_is, max_is = np.where(axis_inferior_superior)[0][[0, -1]]

    return min_lr, max_lr, min_pa, max_pa, min_is, max_is