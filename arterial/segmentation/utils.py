#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import math
import torch

import cc3d
import numpy as np
import nibabel as nib
from time import time
from scipy.ndimage import gaussian_laplace

from arterial import model_registry
from arterial.segmentation.inference import get_nnunet_predictor, run_nnunet_prediction

# Default upper limit for the number of CPU threads used per process
MAX_NUM_THREADS = 8
CGROUP_CPU_MAX_PATH = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_CPU_DIR = "/sys/fs/cgroup/cpu"

def get_usable_cores():
    """
    Returns the number of CPU cores this process may use. It respects the CPU affinity (taskset, SLURM)
    and the cgroup CPU quota (containers), falling back to os.cpu_count() where affinity is not available.

    Returns
    -------
    usable_cores : int
        Number of usable CPU cores (at least 1).

    """
    try:
        usable_cores = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        usable_cores = os.cpu_count() or 1

    # cgroup v2 ("max 100000" when unlimited) or cgroup v1 (quota -1 when unlimited)
    quota, period = None, None
    try:
        with open(CGROUP_CPU_MAX_PATH) as f:
            quota, period = f.read().split()[:2]
    except (OSError, ValueError):
        try:
            with open(os.path.join(CGROUP_V1_CPU_DIR, "cpu.cfs_quota_us")) as f:
                quota = f.read().strip()
            with open(os.path.join(CGROUP_V1_CPU_DIR, "cpu.cfs_period_us")) as f:
                period = f.read().strip()
        except OSError:
            pass
    if quota not in [None, "max", "-1"] and period is not None and float(period) > 0:
        usable_cores = min(usable_cores, max(1, math.ceil(float(quota) / float(period))))

    return max(1, usable_cores)

def get_num_threads(num_threads=None):
    """
    Resolves the number of CPU threads to use. The order of precedence is the num_threads argument, the
    ARTERIAL_NUM_THREADS environment variable and, by default, the usable cores capped at MAX_NUM_THREADS.

    Parameters
    ----------
    num_threads : int, optional
        Number of threads requested explicitly.

    Returns
    -------
    num_threads : int
        Number of CPU threads to use (at least 1).
    usable_cores : int
        Number of usable CPU cores, as given by get_usable_cores.
    source : string
        Where num_threads comes from: "argument", "ARTERIAL_NUM_THREADS" or "default".

    """
    usable_cores = get_usable_cores()
    if num_threads is not None:
        source = "argument"
    elif os.environ.get("ARTERIAL_NUM_THREADS"):
        num_threads, source = int(os.environ["ARTERIAL_NUM_THREADS"]), "ARTERIAL_NUM_THREADS"
    else:
        num_threads, source = min(usable_cores, MAX_NUM_THREADS), "default"

    return max(1, int(num_threads)), usable_cores, source

def compute_cranium_mask(cta_array):
    """
    Returns a binary numpy mask of the cranium, derived from a Laplacian-of-Gaussian
    filter on the upper half of the CTA followed by largest-connected-component
    cleanup. The returned mask is defined on the upper-half voxel grid only (its
    shape is ``(cta_array.shape[0], cta_array.shape[1], cta_array.shape[2] // 2)``)
    — callers that need a full-CTA-shape mask should embed it back into the upper
    half themselves.

    Parameters
    ----------
    cta_array : numpy.ndarray
        3D numpy array with the CTA image.

    Returns
    -------
    cranium_mask : numpy.ndarray
        Binary mask of the cranium on the upper-half voxel grid.

    """
    half_s_coordinate = cta_array.shape[2] // 2
    # Filter in floating point: integer CTAs would be filtered in integer arithmetic and yield an empty mask.
    upper_half_cta_array = np.asarray(cta_array[:, :, half_s_coordinate:], dtype=np.float32)
    filtered_upper_half_cta_array = gaussian_laplace(upper_half_cta_array, sigma=0.0001, mode="nearest")

    tolerance = 0.43 * np.ptp(filtered_upper_half_cta_array)
    threshold = np.min(filtered_upper_half_cta_array) + tolerance
    cranium_mask = np.where(filtered_upper_half_cta_array <= threshold, np.max(cta_array), 0)
    cranium_mask = get_largest_connected_component(cranium_mask)

    return cranium_mask


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
    # Assert there is at least one connected component
    assert(np.any(segmentation))
    # Get largest component (26-connectivity, as skimage.measure.label for 3D volumes)
    largest_connected_component = cc3d.largest_k(np.asarray(segmentation) != 0, k=1, connectivity=26) > 0

    return largest_connected_component  

def run_cranium_segmentation_totalsegmentator(cta_array, cta_affine, resampling="torch", num_threads=None):
    """
    Runs TotalSegmentator for cranium segmentation. If the scan is long (axial length > 240 mm),
    it only considers the upper half of the scan for speed. Otherwise, it considers the whole scan.

    If the spacing coeff are all within 0.3 mm of 0.5 mm, we can ignore preprocessing. 
    To do that, we have to set the spacing to 0.5 mm artificially.

    Parameters
    ----------
    cta_array : numpy.ndarray or array-like object
        Numpy array with the CTA image.
    cta_affine : numpy.ndarray
        Affine matrix of the CTA.
    resampling : string
        Resampling used by nnunet: "scipy" (CPU) or "torch" (GPU).
    num_threads : int, optional
        CPU threads used by the torch resampler.

    Returns
    -------
    cranium_mask : numpy.ndarray or array-like object
        Numpy array with the cranium mask.

    """
    start = time()
    # If axial length of scan is more than 240mm, we can ignore lower part of the scan for speed
    final_mask = np.zeros_like(cta_array)
    only_upper_half = False
    if cta_affine[2, 2] * cta_array.shape[2] > 240:
        print("Long scan (axial >240 mm), assuming head-and-neck CTA and only considering upper half for cranium segmentation")
        cta_array = cta_array[:, :, cta_array.shape[2] // 2:]
        only_upper_half = True

    # If spacing coeff are all within 0.3 mm of 0.5 mm, we can ignore preprocessing. To do that, we have to set the spacing to 0.5 mm artificially
    props = {"spacing": tuple(np.abs(np.diag(cta_affine, k=0)[:3][::-1]))}
    if np.allclose(np.abs(np.diag(cta_affine, k=0)[:3][::-1]), 0.5, atol=0.3):
        print("Spacing coeff are all within 0.3 mm of 0.5 mm, disabling preprocessing")
        # We manually set the spacing to 0.5 mm to avoid preprocessing
        props['spacing'] = (0.5, 0.5, 0.5)

    img = np.expand_dims(cta_array.transpose([2, 1, 0]), axis=0).astype(np.float32)  # reverse axis order to match SITK (from nnunetv2 repo)

    # Read device
    device = torch.device("cpu") # This will take forever on 'cpu', only to be used for debugging
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")

    # Get the nnUNetPredictor (built and loaded only once per process)
    predictor = get_nnunet_predictor(
        model_registry.model_path("segmentation", "totalsegmentator_mandible", "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres"),
        use_folds=("0",),
        device=device,
    )

    # Perform inference
    segmentation_array = run_nnunet_prediction(predictor, img, props, False, resampling, num_threads)
    # Reverse axis order to match nibabel
    segmentation_array = segmentation_array.transpose([2, 1, 0])

    # Add segmentation to final mask
    if only_upper_half:
        final_mask[:, :, final_mask.shape[2] // 2:] = segmentation_array
    else:
        final_mask = segmentation_array
    
    # Get skull mask
    cranium_mask = final_mask == 3
    # Get largest connected component
    cranium_mask = get_largest_connected_component(cranium_mask)

    print("Time needed for cranium segmentation: {:.2f} s".format(time() - start))

    return cranium_mask

def slice_cta_head_and_neck(cta_array, cta_affine, use_laplacian=False, return_bounding_box=False, resampling="torch", num_threads=None):
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
    use_laplacian : bool, optional
        Whether to use a Laplacian of Gaussian filter to perform a segmentation
        of the cranium.
    return_bounding_box : bool, optional
        Whether to return the bounding box of the head and neck parts of the CTA.
        Default is False.
    resampling : string
        Resampling used by nnunet for the cranium segmentation: "scipy" (CPU) or "torch" (GPU).
    num_threads : int, optional
        CPU threads used by the torch resampler.

    Returns
    -------
    cta_head_array : nibabel.nifti1.Nifti1Image
        Nifti object of the head CTA.
    cta_neck_array : nibabel.nifti1.Nifti1Image
        Nifti object of the neck CTA.
    cta_head_affine : numpy.ndarray
        Affine matrix of the head CTA.
    bounding_box : tuple
        Tuple with the bounding box of the head and neck parts of the CTA.
        Default is None.
    """    
    bounding_box = None
    if use_laplacian:
        half_s_coordinate = cta_array.shape[2] // 2
        cranium_mask = compute_cranium_mask(cta_array)
        # Get lowest coordinate with a non-zero voxel from cranium mask
        nonzero_coordinates = np.nonzero(cranium_mask)
        # Get s coordinate for slicing into head and neck
        lower_slicing_i_coordinate = min(nonzero_coordinates[0])
        upper_slicing_i_coordinate = max(nonzero_coordinates[0])
        lower_slicing_j_coordinate = min(nonzero_coordinates[1])
        upper_slicing_j_coordinate = max(nonzero_coordinates[1])
        lower_slicing_k_coordinate = half_s_coordinate + min(nonzero_coordinates[2])
        upper_slicing_k_coordinate = half_s_coordinate + max(nonzero_coordinates[2])
    else:
        # Run TotalSegmentator directly, getting all head structures
        try:
            cranium_mask = run_cranium_segmentation_totalsegmentator(cta_array, cta_affine, resampling, num_threads)
        except Exception as e:
            print(f"Error running TotalSegmentator: {e}")
            print("TotalSegmentator failed, using laplacian-gaussian filter instead...")
            return slice_cta_head_and_neck(cta_array, cta_affine, use_laplacian=True, return_bounding_box=return_bounding_box)

        # Get lowest coordinate with a non-zero voxel from cranium mask
        nonzero_coordinates = np.nonzero(cranium_mask)
        # Get s coordinate for slicing into head and neck
        lower_slicing_i_coordinate = min(nonzero_coordinates[0])
        upper_slicing_i_coordinate = max(nonzero_coordinates[0])
        lower_slicing_j_coordinate = min(nonzero_coordinates[1])
        upper_slicing_j_coordinate = max(nonzero_coordinates[1])
        lower_slicing_k_coordinate = min(nonzero_coordinates[2])
        upper_slicing_k_coordinate = max(nonzero_coordinates[2])

    # Sanity-check the slicing bounds. If the cranium mask reaches near k=0
    # of the full volume (TotalSegmentator path) or the TS-derived bounds
    # otherwise leave one of the slabs degenerately thin, fall back to the
    # Laplacian path — its mask is constrained to the upper half of the volume
    # and is robust to that failure mode. nnUNet's preprocessing crashes with
    # an opaque "zero-size array to reduction operation minimum" inside skimage
    # resize when the neck slab is empty, hence the explicit guard here.
    MIN_SLAB_VOXELS = 32
    neck_slab_voxels = int(1.1 * lower_slicing_k_coordinate)
    head_slab_voxels = upper_slicing_k_coordinate - lower_slicing_k_coordinate
    if neck_slab_voxels < MIN_SLAB_VOXELS or head_slab_voxels < MIN_SLAB_VOXELS:
        if not use_laplacian:
            print(f"Cranium mask produces a degenerate slab "
                  f"(lower_k={lower_slicing_k_coordinate}, upper_k={upper_slicing_k_coordinate}, "
                  f"neck_voxels={neck_slab_voxels}, head_voxels={head_slab_voxels}); "
                  f"falling back to Laplacian-gaussian cranium estimator.")
            return slice_cta_head_and_neck(cta_array, cta_affine, use_laplacian=True, return_bounding_box=return_bounding_box)
        raise ValueError(
            f"Both TotalSegmentator and Laplacian cranium estimators produced a "
            f"degenerate head/neck slab (lower_k={lower_slicing_k_coordinate}, "
            f"upper_k={upper_slicing_k_coordinate}, neck_voxels={neck_slab_voxels}, "
            f"head_voxels={head_slab_voxels}). Cannot proceed with head/neck split."
        )

    if return_bounding_box:
        bounding_box = (lower_slicing_i_coordinate, upper_slicing_i_coordinate, lower_slicing_j_coordinate, upper_slicing_j_coordinate, lower_slicing_k_coordinate, upper_slicing_k_coordinate)

    # Slice cta into two (head and neck)
    cta_head_array = cta_array[
        lower_slicing_i_coordinate:upper_slicing_i_coordinate,
        lower_slicing_j_coordinate:upper_slicing_j_coordinate,
        lower_slicing_k_coordinate:upper_slicing_k_coordinate
        ]

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

    return cta_head_array, cta_neck_array, cta_head_affine, bounding_box

def join_head_and_neck_segmentations(cta_array, cta_affine, segmentation_head_array, segmentation_neck_array, head_affine, is_probabilities=False):
    """
    Joins segmentation niftis for head and neck. Since there is an overlap across 
    segmentations, it selects the overlapping slice with the highest Dice similarity
    to join both segmentations.

    This function joins the head and neck segmentations into a single segmentation.
    If the segmentations are probabilities, it joins them using the highest probability.

    Parameters
    ----------
    cta_array : numpy.ndarray or array-like object
        Numpy array with the CTA image.
    cta_affine : numpy.ndarray
        Affine matrix of the CTA.
    segmentation_head_array : numpy.ndarray or array-like object
        Numpy array with the head segmentation.
    segmentation_neck_array : numpy.ndarray or array-like object
        Numpy array with the neck segmentation.
    head_affine : numpy.ndarray
        Affine matrix of the head.
    is_probabilities : bool
        Whether the segmentations are probabilities.

    Returns
    -------
    segmentation_nifti : nibabel.nifti1.Nifti1Image
        Nifti image with the joined segmentation mask.
    segmentation_array : numpy array
        3D numpy array representing the joined segmentation mask.

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
    if is_probabilities:
        # Convert the probabilities to binary masks
        segmentation_head_array_binary = np.zeros_like(segmentation_head_array_)
        segmentation_head_array_binary[segmentation_head_array_ >= 0.5] = 1
        segmentation_neck_array_binary = np.zeros_like(segmentation_neck_array)
        segmentation_neck_array_binary[segmentation_neck_array >= 0.5] = 1
        for idx in range(segmentation_neck_array_binary.shape[2] - head_origin_k):   
            dice.append(compute_dice(segmentation_neck_array_binary[:, :, head_origin_k + idx], segmentation_head_array_binary[:, :, idx]))
    else:
        for idx in range(segmentation_neck_array.shape[2] - head_origin_k):
            dice.append(compute_dice(segmentation_neck_array[:, :, head_origin_k + idx], segmentation_head_array_[:, :, idx]))

    if len(dice) == 0:
        raise ValueError("Head and neck segmentations do not overlap; cannot find a transition slice.")
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