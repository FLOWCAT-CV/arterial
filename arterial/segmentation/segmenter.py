#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import nibabel as nib
from arterial.segmentation.inference import perform_single_inference_nnunet
from arterial.segmentation.utils import slice_cta_head_and_neck, join_head_and_neck_segmentations
from arterial.io.load_and_save_operations import *

class VesselSegmenter():
    """ 
    Segmenter class to perform vessel segmentation prediction over CTA. the nnunetv2 [1] is used as the
    segmentation background framework.
    
    There are two modes available:

    * `extracranial_vessels`: Segmentation of the extracranial vessels. Performs inference over the
        whole CTA volume. If fast_segmentation=True, the 3d_lowres variant from nnunetv2 is used 
        for the whole volume. Otherwise, the 3d_fullres variant is used for the head and the 3d_lowres
        for the neck.
    * `intracranial_vessels`: Segmentation of the intracranial vessels. Performs inference over the
        intracranial part of the CTA volume. The 3d_fullres variant from nnunetv2 is used.

    References:
    [1] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

    """
    def __init__(self, 
                 case_dir,
                 mode = "extracranial_vesslels",
                 cta_nifti_path = None,
                 fast_segmentation = False,
                 use_vanilla_nnunet = True,
                 no_slicing = False
                 ):
        """
        Initializes object of the Segmenter class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory.
        mode: string, default = "extracranial_vessels"
            Determines whether the centerline is extracted from `extracranial_vessels` or `intracranial_vessels`. 
        cta_nifti_path : string or path-like object, default = None
            Path to the original CTA nifti file. If not provided, the CTA should be in nifti format, the
            name convention used should be `cta.nii.gz` and it should be located in the self.case_dir directory.
        fast_segmentation : bool, default = False
            Boolean variable to be used when running analysis derived from fast segmentation (lowres).
            In this case, segmentation of the cerebral arteries is less reliable, so a higher fraction
            of intracranial slices is ignored, facilitating centerline extraction.
        use_vanilla_nnunet : bool, default = True
            Boolean variable to determine whether the vanilla nnunet model is used. If False, the nnunet model
            trained with the nnunetClDiceLossTrainer is used.
        
        """
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."

        self.case_dir = case_dir
        self.mode = mode
        if cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        else:
            self.cta_nifti_path = cta_nifti_path
        self.fast_segmentation = fast_segmentation

        self.cta_nifti = None
        self.cta_array = None
        self.cta_affine = None

        self.segmentation_nifti = None
        self.segmentation_nifti_path = os.path.join(self.case_dir, self.mode, "segmentation.nii.gz")
        self.segmentation_array = None

        self.cta_head_array = None
        self.cta_neck_array = None
        self.cta_head_affine = None
        self.segmentation_head_array = None
        self.segmentation_neck_array = None

        self.use_vanilla_nnunet = use_vanilla_nnunet
        self.no_slicing = no_slicing

    def segment_vessels_from_cta(self, save=True):
        """
        This method calls perform_inference to perform inference using a trained nnunetv2
        model over the original CTA. At the end of the segmentation prediction, a nifti file 
        with the format:
        
        >>> {self.mode}_segmentation.nii.gz
        
        should be generated in the self.case_dir. 

        In the case of `extracranial_vessels` and fast_segmentation=True, the 3d_lowres variant
        from nnunetv2 is used. In the case of `extracranial_vessels` and fast_segmentation=False, or
        `intracranial_vessels`, the 3d_fullres variant is used for the intracranial part of the image (head)
        while the 3d_lowres is used for the rest of the image (neck).

        Parmeters
        ---------

        Returns
        -------

        """
        if save: os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        if self.cta_array is None or self.cta_affine is None: self.load_cta_nifti()

        if self.mode == "extracranial_vessels":
            if self.fast_segmentation:
                print("Performing fast segmentation...")
                self.segmentation_nifti, self.segmentation_array = perform_single_inference_nnunet(self.cta_array, self.cta_affine, self.mode, "3d_lowres", self.use_vanilla_nnunet)
            else:
                print("Slicing CTA into head and neck...")
                self.slice_cta()
                print("Performing segmentation (head)...")
                _, self.segmentation_head_array =  perform_single_inference_nnunet(self.cta_head_array, self.cta_head_affine, "intracranial_vessels", "3d_fullres", self.use_vanilla_nnunet)
                print("Performing segmentation (neck)...")
                _, self.segmentation_neck_array =  perform_single_inference_nnunet(self.cta_neck_array, self.cta_affine, self.mode, "3d_lowres", self.use_vanilla_nnunet)
                print("Joining segmentations...")
                self.segmentation_nifti, self.segmentation_array = join_head_and_neck_segmentations(self.cta_array, self.cta_affine, self.segmentation_head_array, self.segmentation_neck_array, self.cta_head_affine)
            
        elif self.mode == "intracranial_vessels":   
            if not self.no_slicing:
                print("Slicing CTA for intracranial vessel segmentation...")
                self.slice_cta()
                self.save_head_cta_nifti()
            else:
                print("No slicing is True. Using the whole CTA volume for intracranial vessels segmentation (nnunet trained at full resolution).")
                self.cta_head_array = self.cta_array
                self.cta_head_affine = self.cta_affine
            print("Performing segmentation...")
            self.segmentation_nifti, self.segmentation_array = perform_single_inference_nnunet(self.cta_head_array, self.cta_head_affine, self.mode, "3d_fullres", self.use_vanilla_nnunet)
        
        if save:
            print("Saving segmentation...")
            save_nifti(self.segmentation_nifti,  self.segmentation_nifti_path)

    def slice_cta(self):
        """
        Slices head-and-neck CTA volumes into two separate volumes: head and neck.

        A Laplacian of Gaussian filter is applied for robust skull segmentation. This information
        is used to define the head-and-neck separation.

        Parameters
        ----------

        Returns
        -------
        
        """
        if self.cta_array is None or self.cta_affine is None:
            self.load_cta_nifti()

        self.cta_head_array, self.cta_neck_array, self.cta_head_affine = slice_cta_head_and_neck(self.cta_array, self.cta_affine)

    def load_cta_nifti(self):
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        
        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine

    def set_case_dir(self, case_dir):
        if not isinstance(case_dir, str):
            raise ValueError("case_dir should be a string.")
        self.case_dir = case_dir

    def set_cta_nifti_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.cta_nifti_path = path

    def set_mode(self, mode):
        if mode not in ["extracranial_vessels", "intracranial_vessels"]:
            raise ValueError("mode should be either 'extracranial_vessels' or 'intracranial_vessels'.")
        self.mode = mode

    def set_fast_segmentation(self, fast_segmentation):
        if not isinstance(fast_segmentation, bool):
            raise ValueError("fast_segmentation should be a boolean variable.")
        self.fast_segmentation = fast_segmentation

    def set_segmentation_nifti_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segmentation_nifti_path = path

    def save_segmentation_nifti(self, path=None):
        if self.segmentation_nifti is None:
            raise ValueError("Segmentation nifti is not available. Please run segment_vessels_from_cta method first.")
        if path is None:
            path = self.segmentation_nifti_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_nifti(self.segmentation_nifti, self.segmentation_nifti_path)

    def save_head_cta_nifti(self, path=None):
        if self.cta_head_array is None or self.cta_head_affine is None:
            raise ValueError("Head CTA array is not available. Please run slice_cta method first.")
        if path is None:
            path = os.path.join(self.case_dir, "head_cta.nii.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_nifti(nib.Nifti1Image(self.cta_head_array, self.cta_head_affine), path)
    