#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import numpy as np

from arterial.landmark_detection.landmark_detection import infer_landmarks_from_array

from arterial.io.load_and_save_operations import load_nifti, save_nifti, save_json

class LandmarkDetector:
    """
    Class for automating landmark detection from CTA images.

    """
    def __init__(self,
                 case_dir,
                 mode = "extracranial_vessels",
                 cta_nifti_path = None
                 ):
        """
        Initializes object of the LandmarkDetector class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory.
        mode : string, optional
            Mode of the vessel labeller. The default is "extracranial_vessels", it can also be "intracranial_vessels".
        cta_nifti_path : string or path-like object, optional
            Path to the CTA nifti file. If not provided, it will be assumed that the CTA nifti file is in the case_dir, with the name 'cta.nii.gz'.


        """
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."

        self.case_dir = case_dir
        self.mode = mode
        if cta_nifti_path is None:
            print(f"'cta_nifti_path' not provided. Assuming it is in {self.case_dir}/cta.nii.gz")
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

        """
        if save: os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        if self.cta_array is None or self.cta_affine is None: self._load_cta_nifti_from_file()

        print(f"Detecting landmarks on CTA...")
        self.landmarks_ras_mm_dict, self.landmarks_slicer_json, self.predicted_mask_nib = infer_landmarks_from_array(self.cta_array, self.cta_affine, self.mode, return_mask)
        if save:
            print(f"Saving landmarks in RAS coordinates to {self.landmarks_ras_json_path}")
            save_json(self.landmarks_ras_mm_dict, self.landmarks_ras_json_path)
            print(f"Saving landmarks in Slicer format to {self.landmarks_slicer_json_path}")
            save_json(self.landmarks_slicer_json, self.landmarks_slicer_json_path)
            if return_mask:
                print(f"Saving predicted mask to {self.predicted_mask_nifti_path}")
                save_nifti(self.predicted_mask_nib, self.predicted_mask_nifti_path)

    def _load_cta_nifti_from_file(self):
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine