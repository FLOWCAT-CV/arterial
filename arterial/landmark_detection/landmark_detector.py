#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import numpy as np

from arterial.landmark_detection.landmark_detection import infer_landmarks_from_array
from arterial.io.load_and_save_operations import load_nifti, save_nifti, save_json


class LandmarkDetector:
    """
    Class for automating landmark detection from CTA images.
    
    Supports two modes:
    1. CTA-only detection (default): Uses single-channel model
    2. CTA + Segmentation detection: Uses two-channel model for potentially better accuracy
    
    Additionally supports post-processing refinement that snaps landmarks to 
    vessel centerline/bifurcations using the segmentation mask.
    """
    
    def __init__(self,
                 case_dir,
                 mode="extracranial_vessels",
                 cta_nifti_path=None,
                 segmentation_nifti_path=None
                 ):
        """
        Initializes object of the LandmarkDetector class.

        Parameters
        ----------
        case_dir : str
            Path to case directory where results will be saved.
        mode : str, optional
            Mode of detection. Options: 'extracranial_vessels' (default), 'intracranial_vessels'.
        cta_nifti_path : str, optional
            Path to the CTA nifti file. If not provided, assumes '{case_dir}/cta.nii.gz'.
        segmentation_nifti_path : str, optional
            Path to the vessel segmentation nifti file. If not provided, will look in
            '{case_dir}/{mode}/segmentation.nii.gz' when needed.
        """
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], \
            "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."

        self.case_dir = case_dir
        self.mode = mode
        
        # CTA path
        if cta_nifti_path is None:
            print(f"'cta_nifti_path' not provided. Assuming it is in {self.case_dir}/cta.nii.gz")
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        else:
            self.cta_nifti_path = cta_nifti_path

        # Segmentation path
        if segmentation_nifti_path is None:
            self.segmentation_nifti_path = os.path.join(self.case_dir, self.mode, "segmentation.nii.gz")
        else:
            self.segmentation_nifti_path = segmentation_nifti_path

        # Data holders
        self.cta_nifti = None
        self.cta_array = None
        self.cta_affine = None
        
        self.segmentation_nifti = None
        self.segmentation_array = None

        # Results
        self.landmarks_ras_mm_dict = None
        self.landmarks_ras_json_path = os.path.join(self.case_dir, self.mode, "landmarks.json")
        self.landmarks_slicer_json = None
        self.landmarks_slicer_json_path = os.path.join(self.case_dir, self.mode, "landmarks_slicer.json")

        self.predicted_mask_nifti = None
        self.predicted_mask_nifti_path = os.path.join(self.case_dir, self.mode, "landmarks_mask.nii.gz")

    def detect_landmarks_on_cta(self, return_mask=False, save=True,
                                 use_segmentation_model=True,
                                 refine_with_segmentation=True,
                                 refinement_method='adaptive',
                                 refinement_radius_mm=5.0):
        """
        Detect landmarks on CTA and optionally save results.

        Parameters
        ----------
        return_mask : bool
            Whether to return/save the predicted landmark mask.
        save : bool
            Whether to save the results to disk.
        use_segmentation_model : bool
            If True, use the 2-channel model (CTA + segmentation).
            Requires segmentation to be available.
        refine_with_segmentation : bool
            If True, refine landmarks by snapping to vessel centerline/bifurcations.
            Requires segmentation to be available.
        refinement_method : str
            Method for refinement: 'centerline', 'bifurcation', or 'adaptive'.
            'adaptive' (default) uses bifurcation for eICA landmarks, centerline for others.
        refinement_radius_mm : float
            Maximum search radius for refinement in mm. Default: 5.0.

        Returns
        -------
        landmarks_ras_mm_dict : dict
            Dictionary of landmark names to RAS coordinates.
        """
        if save:
            os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        
        # Load CTA
        if self.cta_array is None or self.cta_affine is None:
            self._load_cta_nifti_from_file()

        # Load segmentation if needed
        segmentation_array = None
        if use_segmentation_model or refine_with_segmentation:
            segmentation_array = self._load_segmentation_if_available()
        if segmentation_array is None:
            if use_segmentation_model:
                print("WARNING: Segmentation not found. Falling back to CTA-only model.")
                use_segmentation_model = False
            if refine_with_segmentation:
                print("WARNING: Segmentation not found. Skipping refinement.")
                refine_with_segmentation = False

        # Detect landmarks
        print(f"Detecting landmarks on CTA...")
        if use_segmentation_model:
            print("  Using 2-channel model (CTA + segmentation)")
        else:
            print("  Using 1-channel model (CTA only)")
        
        if refine_with_segmentation:
            print(f"  Will refine with segmentation (method={refinement_method})")

        self.landmarks_ras_mm_dict, self.landmarks_slicer_json, self.predicted_mask_nifti = \
            infer_landmarks_from_array(
                self.cta_array, 
                self.cta_affine, 
                self.mode, 
                return_mask,
                segmentation_array=segmentation_array,
                use_segmentation_model=use_segmentation_model,
                refine_with_segmentation=refine_with_segmentation,
                refinement_method=refinement_method,
                refinement_radius_mm=refinement_radius_mm
            )

        # Save results
        if save:
            print(f"Saving landmarks in RAS coordinates to {self.landmarks_ras_json_path}")
            save_json(self.landmarks_ras_mm_dict, self.landmarks_ras_json_path)
            
            print(f"Saving landmarks in Slicer format to {self.landmarks_slicer_json_path}")
            save_json(self.landmarks_slicer_json, self.landmarks_slicer_json_path)
            
            if return_mask and self.predicted_mask_nifti is not None:
                print(f"Saving predicted mask to {self.predicted_mask_nifti_path}")
                save_nifti(self.predicted_mask_nifti, self.predicted_mask_nifti_path)

        return self.landmarks_ras_mm_dict

    def _load_cta_nifti_from_file(self):
        """Load CTA from file."""
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine

    def _load_segmentation_if_available(self):
        """Load segmentation if available, return None otherwise."""
        if self.segmentation_array is not None:
            return self.segmentation_array
            
        if not os.path.isfile(self.segmentation_nifti_path):
            print(f"Segmentation not found at {self.segmentation_nifti_path}")
            return None
        
        print(f"Loading segmentation from {self.segmentation_nifti_path}")
        self.segmentation_nifti = load_nifti(self.segmentation_nifti_path)
        self.segmentation_array = self.segmentation_nifti.get_fdata()
        return self.segmentation_array

    def _load_cta_nifti_from_nib(self, cta_nifti):
        self.cta_nifti = cta_nifti
        self.cta_array = cta_nifti.get_fdata()
        self.cta_affine = cta_nifti.affine
        self.image_shape = self.cta_array.shape
    
    def _load_segmentation_nifti_from_nib(self, segmentation_nifti):
        self.segmentation_nifti = segmentation_nifti
        self.segmentation_array = segmentation_nifti.get_fdata()
        self.segmentation_affine = segmentation_nifti.affine
        self.image_shape = self.segmentation_array.shape
