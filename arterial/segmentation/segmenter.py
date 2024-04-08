#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
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
                 fast_segmentation = False
                 ):
        """
        Initializes object of the Segmenter class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory.
        mode: string, default = "extracranial_vessels"
            Determines whether the centerline is extracted from `extracranial_vessels`, `intracranial_vessels` 
            or `thrombus`. 
        cta_nifti_path : string or path-like object, default = None
            Path to the original CTA nifti file. If not provided, the CTA should be in nifti format, the
            name convention used should be:

            >>> cta.nii.gz

            and it should be located in the self.case_dir directory.
        fast_segmentation : bool, default = False
            Boolean variable to be used when running analysis derived from fast segmentation (lowres).
            In this case, segmentation of the cerebral arteries is less reliable, so a higher fraction
            of intracranial slices is ignored, facilitating centerline extraction.

        Returns
        -------
        
        """
        if case_dir is None:
            raise ValueError("case_dir should be provided as the directory where all results will be saved.")
        if mode not in ["extracranial_vessels", "intracranial_vessels"]:
            raise ValueError("mode should be either 'extracranial_vessels' or 'intracranial_vessels'.")

        self.case_dir = case_dir
        os.makedirs(self.case_dir, exist_ok=True)
        self.mode = mode
        if cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        else:
            self.cta_nifti_path = cta_nifti_path
        self.fast_segmentation = fast_segmentation

        if not os.path.exists(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found at {self.cta_nifti_path}. Please provide a valid path.")

        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine

        self.segmentation_nifti = None
        self.segmentation_array = None

        self.cta_head_array = None
        self.cta_neck_array = None
        self.cta_head_affine = None
        self.segmentation_head_array = None
        self.segmentation_neck_array = None

    def segment_vessels_from_cta(self):
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
        if self.mode == "extracranial_vessels":
            if self.fast_segmentation:
                print("Performing fast segmentation...")
                self.segmentation_nifti, self.segmentation_array = perform_single_inference_nnunet(self.cta_array, self.cta_affine, self.mode, "3d_lowres")
            else:
                print("Slicing CTA into head and neck...")
                self.slice_cta()
                print("Performing segmentation (head)...")
                _, self.segmentation_head_array =  perform_single_inference_nnunet(self.cta_head_array, self.cta_head_affine, self.mode, "3d_fullres")
                print("Performing segmentation (neck)...")
                _, self.segmentation_neck_array =  perform_single_inference_nnunet(self.cta_neck_array, self.cta_affine, self.mode, "3d_lowres")
                print("Joining segmentations...")
                self.segmentation_nifti, self.segmentation_array = join_head_and_neck_segmentations(self.cta_array, self.cta_affine, self.segmentation_head_array, self.segmentation_neck_array, self.cta_head_affine)
            
        elif self.mode == "intracranial_vessels":   
            print("Slicing CTA for intracranial vessel segmentation...")
            self.slice_cta()
            print("Performing segmentation...")
            self.segmentation_nifti, self.segmentation_array = perform_single_inference_nnunet(self.cta_head_array, self.cta_head_affine, self.mode, "3d_fullres")
        
        print("Saving segmentation...")
        save_nifti(self.segmentation_nifti, os.path.join(self.case_dir, f"{self.mode}_segmentation.nii.gz"))

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
        self.cta_head_array, self.cta_neck_array, self.cta_head_affine = slice_cta_head_and_neck(self.cta_array, self.cta_affine)

# class ThrombusSegmenter():
#     """ 
#     Segmenter class to perform vessel segmentation prediction over CTA. 

#     """
#     def __init__(self, case_dir):
#         """
#         Initializes object of the Segmenter class.

#         Parameters
#         ----------
#         case_dir : string or path-like object
#             Path to case directory. 

#         Returns
#         -------
        
#         """
#         self.case_dir = case_dir

#     def predict(self):
#         """
#         This method calls perform_inference_thrombus to perform inference using a trained nnunet
#         model over a localized bimodal patch (NCCT + CTA) containing a suspected vessel occlusion.
#         The CTA should be in nifti format, within the self.case_dir directory and with case_id being 
#         the basename of the self.case_dir, the name convention used should be:

#         >>> {case_id}_cta_thrombus_patch.nii.gz
#         >>> {case_id}_ncct_thrombus_patch.nii.gz

#         At the end of the segmentation prediction, a nifti file with the format:
        
#         >>> {case_id}_thrombus_segmentation.nii.gz
        
#         should be generated in the self.case_dir. This class acts as a wrapper
#         for the arterial.segmentation.inference.perform_thrombus_inference() function. 

#         Parmeters
#         ---------

#         Returns
#         -------

#         """
#         perform_inference_thrombus(self.case_dir)
    
#     def predict_patch_recentering(self):
#         """
#         This method calls perform_dynamic_iterative_thrombus_inference to perform inference using a trained nnunet
#         model over a localized bimodal patch (NCCT + CTA) containing a suspected vessel occlusion.
#         The CTA should be in nifti format, within the self.case_dir directory and with case_id being 
#         the basename of the self.case_dir, the name convention used should be:

#         >>> {case_id}_cta_thrombus_patch.nii.gz
#         >>> {case_id}_ncct_thrombus_patch.nii.gz

#         At the end of the segmentation prediction, a nifti file with the format:
        
#         >>> {case_id}_thrombus_segmentation.nii.gz
        
#         should be generated in the self.case_dir. This class acts as a wrapper
#         for the arterial.segmentation.inference.perform_dynamic_iterative_thrombus_inference() function. 

#         Parmeters
#         ---------

#         Returns
#         -------

#         """
#         perform_dynamic_iterative_thrombus_inference(self.case_dir)