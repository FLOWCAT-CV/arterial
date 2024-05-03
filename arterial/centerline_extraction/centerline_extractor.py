#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.centerline_extraction.utils import volume_sanity_check
from arterial.centerline_extraction.preprocessing.preprocessing import preprocess_segmentation_for_centerline_extraction
from arterial.centerline_extraction.centerline_extraction import get_robuts_endpoints, extract_centerlines
from arterial.centerline_extraction.postprocessing.branch_and_clipped_model_extraction import extract_branch_model, unify_branch_models, extract_clipped_model, unify_clipped_models
from arterial.centerline_extraction.postprocessing.postprocessing import compute_centerline_segments_array
from arterial.io.load_and_save_operations import *

class CenterlineExtractor():
    """
    CenterlineExtractor class to perform centerline extraction over predicted
    segmentation.   
        
    """
    def __init__(self, 
                 case_dir, 
                 mode = "extracranial_vessels", 
                 segmentation_nifti_path = None,
                 fast_segmentation = False
                 ):
        """
        Initializes object of the CenterlineExtractor class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory.
        mode: string, default = "extracranial_vessels"
            Determines whether the centerline is extracted from `extracranial_vessels` or `intracranial_vessels`. 
        segmentation_nifti_path : string or path-like object, default = None
            Path to segmentation nifti file. If None, it is assumed that the segmentation nifti file is located
            in the case_dir, with the name `{mode}_segmentation.nii.gz`.
        fast_segmentation : bool, default = False
            Boolean variable to be used when running analysis derived from fast segmentation (lowres).
            In this case, segmentation of the cerebral arteries is less reliable, so a higher fraction
            of intracranial slices is ignored, facilitating centerline extraction.

        
        """
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."
        
        self.case_dir = case_dir
        self.mode = mode
        self.fast_segmentation = fast_segmentation 

        if segmentation_nifti_path is None:
            self.segmentation_nifti_path = os.path.join(self.case_dir, self.mode, "segmentation.nii.gz")
        else:
            self.segmentation_nifti_path = segmentation_nifti_path
        self.segmentation_nifti = None
        self.segmentation_array = None
        self.segmentation_affine = None
        self.image_shape = None

        self.centerlines_dir_path = os.path.join(self.case_dir, self.mode, "centerlines")
        self.segmentations_dir_path = os.path.join(self.case_dir, self.mode, "segmentations")
        self.branch_models_dir_path = os.path.join(self.case_dir, self.mode, "branch_models")
        self.clipped_models_dir_path = os.path.join(self.case_dir, self.mode, "clipped_models")

        self.segmentation_model_list = []
        self.endpoints_list = []
        self.centerline_model_list = []
        self.voronoi_diagrams_list = []
        self.branch_model_list = []
        self.clipped_model_list = []

        self.segmentation_path = os.path.join(self.case_dir, self.mode, "segmentation.vtk")
        self.branch_model_path = os.path.join(self.case_dir, self.mode, "branch_model.vtk")
        self.clipped_model_path = os.path.join(self.case_dir, self.mode, "clipped_model.vtk")

        self.segmentation_model = None
        self.branch_model = None
        self.clipped_model = None


        self.centerline_segments_array_path = os.path.join(self.case_dir, self.mode, "centerline_segments_array.npy")
        self.centerline_segments_array = None

    def perform_preprocessing(self, save=True):
        """
        Performs preprocessing of the segmentation nifti file to generate a vtkpolydata 
        of the segmentation's surface mode, as well as the segmentation model list (each of the
        large islands in the segmentation).

        Parameters
        ----------
        save : bool, default = True
            Boolean variable to determine whether to save the segmentation models in the case directory.

        Returns
        -------
        
        """
        if self.segmentation_array is None or self.segmentation_affine is None:
            self.load_segmentation_nifti()
        volume_sanity_check(self.segmentation_array, self.segmentation_affine)
        self.segmentation_model, self.segmentation_model_list = preprocess_segmentation_for_centerline_extraction(self.segmentation_array, self.segmentation_affine)
        
        if save:
            save_vtkpolydata(self.segmentation_model, self.segmentation_path)
            for idx, segmentation_model in enumerate(self.segmentation_model_list):
                save_vtkpolydata(segmentation_model, os.path.join(self.segmentations_dir_path, f"segmentation_{idx}.vtk"))
    
    def perform_centerline_extraction(self, save=True):
        """
        Given that segmentation models have been previously extracted, this function
        automatically detects endpoints based on the extraction of the centerline network (VMTK)
        from the segmentation models and relocates endpoints if needed for robust centerline extraction.
        Then, it runs the extraction of centerline models using VMTK from the segmentation models.

        This full implementation is based on the VTMK package and the VMTK Slicer extension:

        > https://github.com/vmtk/vmtk
        > https://github.com/vmtk/SlicerExtension-VMTK/

        Parameters
        ----------

        Returns
        -------

        """
        if save:
            os.makedirs(self.centerlines_dir_path, exist_ok=True)
            os.makedirs(self.segmentations_dir_path, exist_ok=True)

        if self.segmentation_model is None:
            self.perform_preprocessing()
        if self.segmentation_array is None or self.segmentation_affine is None:
            self.load_segmentation_nifti()

        for idx, segmentation_model_idx in enumerate(self.segmentation_model_list):
            self.endpoints_list.append(get_robuts_endpoints(segmentation_model_idx, self.segmentation_array, self.segmentation_affine))
            centerlines, voronoi_diagram = extract_centerlines(segmentation_model_idx, self.endpoints_list[idx])
            self.centerline_model_list.append(centerlines)
            self.voronoi_diagrams_list.append(voronoi_diagram)
    
        if save:
            for idx, centerline_model in enumerate(self.centerline_model_list):
                save_vtkpolydata(centerline_model, os.path.join(self.centerlines_dir_path, f"centerlines_{idx}.vtk"))
        
    def perform_branch_model_extraction(self, save=True):
        """
        Given that centerline models have been previously extracted, this function
        runs the extraction of branch models using VMTK from the centerline models.

        At the end of the execution, all separate branch models are unified into a single
        branch model, which is stored in self.branch_model and saved in the case directory.
        
        Parameters
        ----------
        save : bool, default = True
            Boolean variable to determine whether to save the branch models in the case directory.

        Returns
        -------

        """
        if len(self.centerline_model_list) == 0: 
            raise ValueError("Centerline model list is empty. Please run self.perform_centerline_extraction() first.")
        
        if save: os.makedirs(self.branch_models_dir_path, exist_ok=True)

        for idx, centerline_model in enumerate(self.centerline_model_list):
            print(f"Extracting branch model from centerline model {idx}...")
            branch_model_ = extract_branch_model(centerline_model)
            if branch_model_ is None and idx == 0:
                raise ValueError("Branch model could not be extracted for first model. Interrupting computation, as this"\
                                 "is a critical error for all the posterior pipeline.")
            if branch_model_ is not None:
                self.branch_model_list.append(branch_model_)
                if save:
                    save_vtkpolydata(branch_model_, os.path.join(self.branch_models_dir_path, f"branch_model_{idx}.vtk"))
        
        print("Unifying branch models...")
        self.branch_model = unify_branch_models(self.branch_model_list)
        if save:
            save_vtkpolydata(self.branch_model, self.branch_model_path)

    def perform_clipped_model_extraction(self, save=True):
        """
        Given that segmentation models and branch models have been previously extracted, this function
        runs the extraction of clipped models using VMTK from the segmentation models and branch models.

        At the end of the execution, all separate clipped models are unified into a single
        clipped model, which is stored in self.clipped_model and saved in the case directory.
        
        Parameters
        ----------
        save : bool, default = True
            Boolean variable to determine whether to save the clipped models in the case directory.

        Returns
        -------

        """
        if len(self.segmentation_model_list) == 0: 
            raise ValueError("Segmentation model list is empty. Please run self.perform_centerline_extraction() first.")
        if len(self.branch_model_list) == 0: 
            raise ValueError("Branch model is empty. Please run perform_branch_model_extraction() first.")

        if save: os.makedirs(self.clipped_models_dir_path, exist_ok=True)

        for idx, segmentation_model in enumerate(self.segmentation_model_list):
            print(f"Extracting clipped model from segmentation model {idx}...")
            clipped_model_ = extract_clipped_model(segmentation_model, self.branch_model_list[idx])
            if clipped_model_ is not None:
                self.clipped_model_list.append(clipped_model_)
                if save:
                    save_vtkpolydata(clipped_model_, os.path.join(self.clipped_models_dir_path, f"clipped_model_{idx}.vtk"))

        print("Unifying clipped models...")
        self.clipped_model = unify_clipped_models(self.clipped_model_list)
        if save:
            save_vtkpolydata(self.clipped_model, self.clipped_model_path)

    def perform_centerline_postprocessing(self, save=True):
        """
        Runs postprocessing of the centerlines model to generate a numpy array
        with the coordinates and radii of each centerline point for each non-overlapping
        centerline segment.

        Parameters
        ----------
        save : bool, default = True
            Boolean variable to determine whether to save the centerline segments array in the case directory.

        Returns
        -------
        
        """
        if len(self.centerline_model_list) == 0: 
            raise ValueError("Centerline model list is empty. Please run self.perform_centerline_extraction() first.")
        
        if self.segmentation_nifti is None or self.affine is None or self.image_shape is None:
            self.load_segmentation_nifti()

        print("Computing centerline segments array...")
        self.centerline_segments_array = compute_centerline_segments_array(
            self.centerline_model_list, 
            self.affine, 
            self.image_shape, 
            self.mode
            )

        if save:
            save_numpy(self.centerline_segments_array, self.centerline_segments_array_path)

    def load_segmentation_nifti(self):
        if not os.path.isfile(self.segmentation_nifti_path): 
            raise FileNotFoundError(f"Segmentation nifti file not found: {self.segmentation_nifti_path}. \nPlease run segmentation first.")

        self.segmentation_nifti = load_nifti(self.segmentation_nifti_path)
        self.segmentation_array = self.segmentation_nifti.get_fdata()
        self.segmentation_affine = self.segmentation_nifti.affine
        self.image_shape = self.segmentation_nifti.shape

    def load_centerline_model_list(self):
        if not os.path.isdir(self.centerlines_dir_path): 
            raise FileNotFoundError(f"Centerlines directory not found: {self.centerlines_dir_path}. \nPlease run centerline extraction first.")
        if len(os.listdir(self.centerlines_dir_path)) == 0: 
            raise ValueError(f"Centerlines directory is empty: {self.centerlines_dir_path}. \nPlease run centerline extraction first.")

        self.centerline_model_list = load_vtk_list_from_dir(self.centerlines_dir_path)
    
    def load_segmentation_model_list(self):
        if not os.path.isdir(self.segmentations_dir_path): 
            raise FileNotFoundError(f"Segmentations directory not found: {self.segmentations_dir_path}. \nPlease run centerline extraction first.")
        if len(os.listdir(self.segmentations_dir_path)) == 0: 
            raise ValueError(f"Segmentations directory is empty: {self.segmentations_dir_path}. \nPlease run centerline extraction first.")

        self.segmentation_model_list = load_vtk_list_from_dir(self.segmentations_dir_path)

    def load_segmentation(self):
        if not os.path.isfile(self.segmentation_path):
            raise FileNotFoundError(f"Segmentation file not found: {self.segmentation_path}. \nPlease run centerline extraction first.")
        self.segmentation_model = load_vtkpolydata(self.segmentation_path)

    def load_branch_model_list(self):
        if not os.path.isdir(self.branch_models_dir_path):
            raise FileNotFoundError(f"Branch models directory not found: {self.branch_models_dir_path}. \nPlease run centerline extraction and branch model extraction first.")
        if len(os.listdir(self.branch_models_dir_path)) == 0: 
            raise ValueError(f"Branch models directory is empty: {self.branch_models_dir_path}. \nPlease run centerline extraction and branch model extraction first.")
        self.branch_model_list = load_vtk_list_from_dir(self.branch_models_dir_path)
    
    def load_branch_model(self):
        if not os.path.isfile(self.branch_model_path):
            raise FileNotFoundError(f"Branch model file not found: {self.branch_model_path}. \nPlease run centerline extraction and branch model extraction first.")
        self.branch_model = load_vtkpolydata(self.branch_model_path)

    def load_clipped_model_list(self):
        if not os.path.isdir(self.clipped_models_dir_path):
            raise FileNotFoundError(f"Clipped models directory not found: {self.clipped_models_dir_path}. \nPlease run centerline extraction and clipped model extraction first.")
        if len(os.listdir(self.clipped_models_dir_path)) == 0:
            raise ValueError(f"Clipped models directory is empty: {self.clipped_models_dir_path}. \nPlease run centerline extraction and clipped model extraction first.")
        self.clipped_model_list = load_vtk_list_from_dir(self.clipped_models_dir_path)
    
    def load_clipped_model(self):
        if not os.path.isfile(self.clipped_model_path):
            raise FileNotFoundError(f"Clipped model file not found: {self.clipped_model_path}. \nPlease run centerline extraction and clipped model extraction first.")
        self.clipped_model = load_vtkpolydata(self.clipped_model_path)

    def set_case_dir(self, case_dir):
        if not isinstance(case_dir, str):
            raise ValueError("case_dir should be a string.")
        self.case_dir = case_dir

    def set_mode(self, mode):
        if mode not in ["extracranial_vessels", "intracranial_vessels"]:
            raise ValueError("mode should be either 'extracranial_vessels' or 'intracranial_vessels'.")
        self.mode = mode

    def set_segmentation_nifti_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segmentation_nifti_path = path

    def set_fast_segmentation(self, fast_segmentation):
        if not isinstance(fast_segmentation, bool):
            raise ValueError("fast_segmentation should be a boolean variable.")
        self.fast_segmentation = fast_segmentation
        