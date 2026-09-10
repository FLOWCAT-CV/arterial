#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import torch

import numpy as np
import nibabel as nib

from arterial.landmark_detection.model import load_trained_model_seg
from arterial.landmark_detection.utils import preprocess_for_landmark_detection, postprocess_preds, resample_mask_to_original_cta, refine_landmarks_with_segmentation
from arterial.centerline_extraction.utils import build_endpoints_json


def infer_landmarks_from_array(cta_array, cta_affine, mode="extracranial_vessels", 
                                return_mask=False, segmentation_array=None,
                                use_segmentation_model=False, refine_with_segmentation=False,
                                refinement_method='adaptive', refinement_radius_mm=20.0):
    """
    Infer landmarks from a CTA volume.

    Parameters
    ----------
    cta_array : np.ndarray
        CTA data.
    cta_affine : np.ndarray
        Affine transformation matrix.
    mode : str
        Mode: 'extracranial_vessels' or 'intracranial_vessels'.
    return_mask : bool
        Whether to return the predicted mask.
    segmentation_array : np.ndarray, optional
        Vessel segmentation mask. Required if use_segmentation_model=True or refine_with_segmentation=True.
    use_segmentation_model : bool
        If True, use 2-channel model (CTA + segmentation). Requires segmentation_array.
    refine_with_segmentation : bool
        If True, refine landmarks by snapping to vessel centerline/bifurcations.
        Requires segmentation_array.
    refinement_method : str
        Refinement method: 'centerline', 'bifurcation', 'adaptive'.
    refinement_radius_mm : float
        Maximum search radius for refinement in mm.

    Returns
    -------
    landmarks_ras_mm_dict : dict
        Dictionary containing the landmarks in RAS coordinates.
    landmarks_slicer_json : dict
        Dictionary containing the landmarks in Slicer format.
    predicted_mask_nib : nibabel.Nifti1Image or None
        Predicted mask. None if return_mask is False.
    """
    # Validate inputs
    if use_segmentation_model and segmentation_array is None:
        raise ValueError("segmentation_array is required when use_segmentation_model=True")
    if refine_with_segmentation and segmentation_array is None:
        raise ValueError("segmentation_array is required when refine_with_segmentation=True")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine model path and input channels
    arterial_dir = os.environ["arterial_dir"]
    
    if use_segmentation_model:
        model_path = os.path.join(arterial_dir, "landmark_detection/models/six_landmarks_2ch.pth")
        in_channels = 2
        print("Using 2-channel model (CTA + segmentation)")
    else:
        model_path = os.path.join(arterial_dir, "landmark_detection/models/six_landmarks_11_7.pth")
        in_channels = 1
        print("Using 1-channel model (CTA only)")

    # Load model
    model = load_trained_model_seg(model_path, device, in_channels=in_channels)
    model.eval()

    # Prepare input
    print("Preprocessing volume for landmark detection...")
    if use_segmentation_model:
        preprocessed_volume, preprocessed_affine = preprocess_for_landmark_detection(
            cta_array, cta_affine, segmentation_array=segmentation_array, device=device
        )
    else:
        preprocessed_volume, preprocessed_affine = preprocess_for_landmark_detection(
            cta_array, cta_affine, device=device
        )

    # Perform inference
    print("Performing landmark detection...")
    with torch.no_grad():
        pred = model(preprocessed_volume)

    # Postprocess predictions
    print("Postprocessing predictions...")
    landmarks_ras_mm, predicted_mask_array = postprocess_preds(pred, preprocessed_affine, return_mask)
    
    # Define landmark labels based on mode
    if mode == "intracranial_vessels":
        landmarks_ras_mm = np.delete(landmarks_ras_mm, [2, 3], axis=0)
        landmarks_labels = ["l-tica", "r-tica", "r-mca", "l-mca"]
    else:
        landmarks_labels = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
    
    landmarks_ras_mm_dict = {label: tuple(landmarks_ras_mm[i]) for i, label in enumerate(landmarks_labels)}

    # Refine landmarks using segmentation if requested
    if refine_with_segmentation:
        print(f"Refining landmarks using segmentation (method={refinement_method}, radius={refinement_radius_mm}mm)...")
        landmarks_ras_mm_dict, refinement_stats = refine_landmarks_with_segmentation(
            landmarks_ras_mm_dict,
            segmentation_array,
            cta_affine,
            method=refinement_method,
            search_radius_mm=refinement_radius_mm
        )
        if refinement_stats:
            print(f"  Mean displacement: {refinement_stats.get('mean_displacement_mm', 0):.2f}mm")

    # Build Slicer-format JSON
    # Convert dict values back to array for build_endpoints_json
    refined_coords = np.array([landmarks_ras_mm_dict[label] for label in landmarks_labels])
    landmarks_slicer_json = build_endpoints_json(refined_coords, cta_affine, landmarks_labels)
    
    # Handle predicted mask
    if return_mask:
        print("Resampling mask back to CTA native space...")
        resampled_predicted_mask = resample_mask_to_original_cta(
            predicted_mask_array, preprocessed_affine, cta_array, cta_affine
        )
        predicted_mask_nib = nib.Nifti1Image(resampled_predicted_mask, cta_affine)
    else:
        predicted_mask_nib = None

    return landmarks_ras_mm_dict, landmarks_slicer_json, predicted_mask_nib
