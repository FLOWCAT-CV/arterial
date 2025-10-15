#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import torch

import nibabel as nib

from arterial.landmark_extraction.model import load_trained_model_seg
from arterial.landmark_extraction.utils import preprocess_for_landmark_extraction, postprocess_preds, resample_mask_to_original_cta
from arterial.centerline_extraction.utils import build_endpoints_json

def infer_landmarks_from_array(cta_array, cta_affine, return_mask=False):
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
    model_path = os.environ["arterial_dir"] + "/landmark_extraction/models/six_landmarks_11_7.pth"
    model = load_trained_model_seg(model_path, device)
    model.eval()

    # Prepare input
    print("Preprocessing CTA volume for landmark extraction...")
    preprocessed_volume, preprocessed_affine = preprocess_for_landmark_extraction(cta_array, cta_affine, device=device)

    # Perform inference
    print("Performing landmark extraction on preprocessed CTA volume...")
    with torch.no_grad():
        pred = model(preprocessed_volume)

    # Postprocess predictions to get landmarks in RAS coordinates
    print("Postprocessing predictions to get landmarks in RAS coordinates...")
    landmarks_ras_mm, predicted_mask_array = postprocess_preds(pred, preprocessed_affine, return_mask)
    landmarks_labels = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
    landmarks_ras_mm_dict = {label: tuple(landmarks_ras_mm[i]) for i, label in enumerate(landmarks_labels)}

    # We build the json in Slicer format as well
    landmarks_slicer_json = build_endpoints_json(landmarks_ras_mm, cta_affine, landmarks_labels) # affine only used for orientation, can use either preprocessed or original affine

    # TODO: implement a sanity check to remove landmarks that are not found? 
    #       What happens when I pass a head CTA without the Neck? Probably that should be introdued during training? 
    #       Patch-based method to account for this variability with the same model? Monai supports sliding window inference.
    
    if return_mask:
        print("Resampling mask back to CTA native space...")
        resampled_predicted_mask = resample_mask_to_original_cta(predicted_mask_array, preprocessed_affine, cta_array, cta_affine)
        predicted_mask_nib = nib.Nifti1Image(resampled_predicted_mask, cta_affine)
    else:
        predicted_mask_nib = None

    return landmarks_ras_mm_dict, landmarks_slicer_json, predicted_mask_nib