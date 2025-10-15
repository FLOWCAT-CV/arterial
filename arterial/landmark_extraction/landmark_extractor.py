#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import numpy as np

from arterial.landmark_extraction.landmark_extraction import infer_landmarks_from_array

from arterial.io.load_and_save_operations import load_nifti, save_nifti, save_json

class LandmarkExtractor:
    """
    Class for automating landmark extraction from CTA images.

    Parameters
    ----------
        model_path (str): Path to the trained model file.
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to None, which auto-selects.
    
    Methods
    -------
        infer_folder(folder_path, output_folder, save_mask=True, save_json=True):
            Perform inference on a folder containing 'cta.nii.gz' and save results.
        infer_from_array(data, affine=None, output_folder=None, save_mask=False, save_json=False):
            Perform inference on numpy array or nibabel image and return landmarks.
    """
    def __init__(self,
                 case_dir,
                 mode = "extracranial_vessels",
                 cta_nifti_path = None
                 ):
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels"], "mode should be either 'extracranial_vessels'. 'intracranial_vessels' is not supported yet."

        self.case_dir = case_dir
        self.mode = mode
        if cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        else:
            self.cta_nifti_path = cta_nifti_path

        self.cta_nifti = None
        self.cta_array = None
        self.cta_affine = None

        self.landmarks_ras_mm_dict = None
        self.landmarks_ras_json_path = os.path.join(self.case_dir, self.mode, "landmarks.json")
        self.landmarks_slicer_json = None
        self.landmarks_slicer_json_path = os.path.join(self.case_dir, self.mode, "landmarks_slicer.json")

        self.predicted_mask_nifti = None
        self.predicted_mask_nifti_path = os.path.join(self.case_dir, self.mode, "predicted_mask.nii.gz")

    def detect_landmarks_on_cta(self, return_mask=False, save=True):
        """
        Detect landmarks on CTA and save results.

        Parameters
        ----------
        return_mask: bool
            Whether to return the predicted mask.
        save: bool
            Whether to save the results.
        
        Returns
        -------
            None

        """
        if save: os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        if self.cta_array is None or self.cta_affine is None: self.load_cta_nifti()

        print(f"Detecting landmarks on CTA...")
        self.landmarks_ras_mm_dict, self.landmarks_slicer_json, self.predicted_mask_nib = infer_landmarks_from_array(self.cta_array, self.cta_affine, return_mask)

        if save:
            save_json(self.landmarks_ras_mm_dict, self.landmarks_ras_json_path)
            save_json(self.landmarks_slicer_json, self.landmarks_slicer_json_path)
            if return_mask:
                save_nifti(self.predicted_mask_nib, self.predicted_mask_nifti_path)

    def load_cta_nifti(self):
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine