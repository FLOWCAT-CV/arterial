#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.centerline_extraction.utils import volume_sanity_check
from arterial.centerline_extraction.run_centerline_extraction_slicer import perform_preprocessing_and_centerline_extraction
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
            in the case directory, with the name {mode}_segmentation.nii.gz.
        fast_segmentation : bool, default = False
            Boolean variable to be used when running analysis derived from fast segmentation (lowres).
            In this case, segmentation of the cerebral arteries is less reliable, so a higher fraction
            of intracranial slices is ignored, facilitating centerline extraction.

        Returns
        -------
        
        """
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."
        
        self.case_dir = case_dir
        if not os.path.isdir(self.case_dir): os.makedirs(self.case_dir, exist_ok=True)
        self.mode = mode
        self.fast_segmentation = fast_segmentation 

        if segmentation_nifti_path is None:
            self.segmentation_nifti_path = os.path.join(self.case_dir, f"{self.mode}_segmentation.nii.gz")
        else:
            self.segmentation_nifti_path = segmentation_nifti_path
        self.segmentation_nifti = None
        self.affine = None
        self.image_shape = None

        self.centerlines_dir_path = os.path.join(self.case_dir, "centerlines")
        self.segmentations_dir_path = os.path.join(self.case_dir, "segmentations")
        self.branch_models_dir_path = os.path.join(self.case_dir, "branch_models")
        self.clipped_models_dir_path = os.path.join(self.case_dir, "clipped_models")

        self.centerline_model_list = []
        self.segmentation_model_list = []
        self.branch_model_list = []
        self.clipped_model_list = []

        self.segmentation_path = os.path.join(self.case_dir, f"{self.mode}_segmentation.vtk")
        self.branch_model_path = os.path.join(self.case_dir, f"{self.mode}_branch_model.vtk")
        self.clipped_model_path = os.path.join(self.case_dir, f"{self.mode}_clipped_model.vtk")

        self.segmentation = None
        self.branch_model = None
        self.clipped_model = None

        self.centerline_segments_array_path = os.path.join(self.case_dir, f"{self.mode}_centerline_segments_array.npy")
        self.centerline_segments_array = None
    
    def perform_centerline_extraction(self):
        """
        Runs preprocessing and centerline extraction, including analysis for circular
        centerlines, all using Slicer and SlicerVMTK functions. Since PythonSlicer
        functions and Slicer GUI elements are used to run the analysis, there is a need for an 
        intermediate script that runs a command line command. We use os.system() to do that.        

        At the end of the execution, the following files should be generated:

        >>> case_dir/centerlines/{self.mode}_centerlines_{idx}.vtk
        >>> case_dir/segmentations/{self.mode}_segmentation_{idx}.vtk
        >>> case_dir/{self.mode}_segmentation.vtk
        >>> case_dir/{self.mode}_segmentation.stl
        
        Acts as a wrapper for the arterial.centerline_extraction.run_centerline_extraction_slicer.
            perform_preprocessing_and_centerline_extraction() function.

        Parameters
        ----------

        Returns
        -------

        """
        if not os.path.isfile(self.segmentation_nifti_path): 
            raise FileNotFoundError(f"Segmentation nifti file not found: {self.segmentation_nifti_path}. \nPlease run segmentation first.")
    
        if self.mode == "extracranial_vessels":
            self.load_segmentation_nifti()
            # Perform volume sanity check
            volume_sanity_check(self.segmentation_nifti)

        # For preprocessing and centerline extraction we have to rely on using Slicer and SlicerVMTK functions
        # This is due to the fact that the AutoDetectEndpoints functionality from VMTK is only available in the
        # SlicerVMTK extension, and not in the VMTK library.
        # Thus, this function is a wrapper for a CLI call to a Python script that runs Slicer to obtain 3D models
        # resulting from the segementer.
        print("Performing preprocessing and centerline extraction in Slicer...")
        perform_preprocessing_and_centerline_extraction(self.case_dir, self.segmentation_nifti_path, self.mode, self.fast_segmentation)

        self.centerline_model_list = load_vtk_list(self.centerlines_dir_path, self.mode)
        self.segmentation_model_list = load_vtk_list(self.segmentations_dir_path, self.mode)

        self.segmentation = load_vtkpolydata(self.segmentation_path)
        
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
                    save_vtkpolydata(branch_model_, os.path.join(self.branch_models_dir_path, f"{self.mode}_branch_model_{idx}.vtk"))
        
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
                    save_vtkpolydata(clipped_model_, os.path.join(self.clipped_models_dir_path, f"{self.mode}_clipped_model_{idx}.vtk"))

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
        self.affine = self.segmentation_nifti.affine
        self.image_shape = self.segmentation_nifti.shape

    def load_centerline_model_list(self):
        if not os.path.isdir(self.centerlines_dir_path): 
            raise FileNotFoundError(f"Centerlines directory not found: {self.centerlines_dir_path}. \nPlease run centerline extraction first.")
        if len(os.listdir(self.centerlines_dir_path)) == 0: 
            raise ValueError(f"Centerlines directory is empty: {self.centerlines_dir_path}. \nPlease run centerline extraction first.")

        self.centerline_model_list = load_vtk_list(self.centerlines_dir_path)
    
    def load_segmentation_model_list(self):
        if not os.path.isdir(self.segmentations_dir_path): 
            raise FileNotFoundError(f"Segmentations directory not found: {self.segmentations_dir_path}. \nPlease run centerline extraction first.")
        if len(os.listdir(self.segmentations_dir_path)) == 0: 
            raise ValueError(f"Segmentations directory is empty: {self.segmentations_dir_path}. \nPlease run centerline extraction first.")

        self.segmentation_model_list = load_vtk_list(self.segmentations_dir_path)

    def load_segmentation(self):
        assert os.path.isfile(self.segmentation_path), f"Segmentation file not found: {self.segmentation_path}. \nPlease run centerline extraction first."

        self.segmentation = load_vtkpolydata(self.segmentation_path)

    def load_branch_model_list(self):
        if not os.path.isdir(self.branch_models_dir_path):
            raise FileNotFoundError(f"Branch models directory not found: {self.branch_models_dir_path}. \nPlease run centerline extraction first.")
        if len(os.listdir(self.branch_models_dir_path)) == 0: 
            raise ValueError(f"Branch models directory is empty: {self.branch_models_dir_path}. \nPlease run centerline extraction first.")

        self.branch_model_list = load_vtk_list(self.branch_models_dir_path)
    
    def load_branch_model(self):
        if not os.path.isfile(self.branch_model_path):
            raise FileNotFoundError(f"Branch model file not found: {self.branch_model_path}. \nPlease run centerline extraction first.")

        self.branch_model = load_vtkpolydata(self.branch_model_path)

    def load_clipped_model_list(self):
        if not os.path.isdir(self.clipped_models_dir_path):
            raise FileNotFoundError(f"Clipped models directory not found: {self.clipped_models_dir_path}. \nPlease run centerline extraction first.")
        if len(os.listdir(self.clipped_models_dir_path)) == 0:
            raise ValueError(f"Clipped models directory is empty: {self.clipped_models_dir_path}. \nPlease run centerline extraction first.")

        self.clipped_model_list = load_vtk_list(self.clipped_models_dir_path)
    
    def load_clipped_model(self):
        if not os.path.isfile(self.clipped_model_path):
            raise FileNotFoundError(f"Clipped model file not found: {self.clipped_model_path}. \nPlease run centerline extraction first.")

        self.clipped_model = load_vtkpolydata(self.clipped_model_path)