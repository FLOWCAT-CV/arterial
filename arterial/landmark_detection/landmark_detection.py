#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import torch

import numpy as np
import nibabel as nib

from arterial.landmark_detection.model import load_trained_model_seg
from arterial.landmark_detection.utils import preprocess_for_landmark_detection, postprocess_preds, resample_mask_to_original_cta
from arterial.centerline_extraction.utils import build_endpoints_json

def infer_landmarks_from_array(cta_array, cta_affine, mode="extracranial_vessels", return_mask=False):
    """
    Infer landmarks from a CTA volume.

    Parameters
    ----------
    cta_array: np.ndarray
        CTA data.
    cta_affine: np.ndarray
        Affine transformation matrix.
    return_mask: bool
        Whether to return the predicted mask.

    Returns
    -------
    landmarks_ras_mm_dict: dict
        Dictionary containing the landmarks in RAS coordinates.
    landmarks_slicer_json: dict
        Dictionary containing the landmarks in Slicer format.
    predicted_mask_nib: nibabel.Nifti1Image, None
        Predicted mask. None if return_mask is False.
        
    """
    # Read device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")

    # Load model
    model_path = os.environ["arterial_dir"] + "/landmark_detection/models/six_landmarks_11_7.pth"
    model = load_trained_model_seg(model_path, device)
    model.eval()

    # Prepare input
    print("Preprocessing CTA volume for landmark detection...")
    preprocessed_volume, preprocessed_affine = preprocess_for_landmark_detection(cta_array, cta_affine, device=device)

    # Perform inference
    print("Performing landmark detection on preprocessed CTA volume...")
    with torch.no_grad():
        pred = model(preprocessed_volume)

    # Postprocess predictions to get landmarks in RAS coordinates
    print("Postprocessing predictions to get landmarks in RAS coordinates...")
    landmarks_ras_mm, predicted_mask_array = postprocess_preds(pred, preprocessed_affine, return_mask)
    if mode == "intracranial_vessels":
        landmarks_ras_mm = np.delete(landmarks_ras_mm, [2, 3], axis=0)
        landmarks_labels = ["l-tica", "r-tica", "r-mca", "l-mca"]
    else:
        landmarks_labels = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
    landmarks_ras_mm_dict = {label: tuple(landmarks_ras_mm[i]) for i, label in enumerate(landmarks_labels)}

    # TODO: implement the robust endpoint relocation step here if segmentation is available

    # We build the json in Slicer format as well
    landmarks_slicer_json = build_endpoints_json(landmarks_ras_mm, cta_affine, landmarks_labels) # affine only used for orientation, can use either preprocessed or original affine

    
    if return_mask:
        print("Resampling mask back to CTA native space...")
        resampled_predicted_mask = resample_mask_to_original_cta(predicted_mask_array, preprocessed_affine, cta_array, cta_affine)
        predicted_mask_nib = nib.Nifti1Image(resampled_predicted_mask, cta_affine)
    else:
        predicted_mask_nib = None

    return landmarks_ras_mm_dict, landmarks_slicer_json, predicted_mask_nib