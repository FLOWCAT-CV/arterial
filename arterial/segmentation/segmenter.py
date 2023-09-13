#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.segmentation.inference import perform_inference_fast, perform_inference_full, perform_inference_intracranial
from arterial.segmentation.inference import perform_inference_thrombus, perform_dynamic_iterative_thrombus_inference

class VesselSegmenter():
    """ 
    Segmenter class to perform vessel segmentation prediction over CTA. 

    """
    def __init__(self, case_dir):
        """
        Initializes object of the Segmenter class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        self.case_dir = case_dir

    def predict_fast(self):
        """
        This method calls perform_inference to perform inference using a trained nnunet
        model over the original CTA. The CTA should be in nifti format, within the 
        self.case_dir directory and with case_id being the basename of the self.case_dir,
        the name convention used should be:

        >>> {case_id}_cta.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_id}_vessel_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.segmentation.inference.perform_inference() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_inference_fast(self.case_dir)

    def predict_full(self):
        """
        This method calls perform_inference_full to perform inference using two trained nnunet
        models over the original CTA. The CTA should be in nifti format, within the 
        self.case_dir directory and with case_id being the basename of the self.case_dir,
        the name convention used should be:

        >>> {case_id}_cta.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_id}_vessel_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.segmentation.inference.perform_inference_full() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_inference_full(self.case_dir)

    def predict_intracranial(self):
        """
        This method calls perform_inference_intracranial to perform inference using a trained nnunet
        model over the CTA cropped to include only the intracranial region. The CTA should be in nifti 
        format, within the self.case_dir directory and with case_id being the basename of the self.case_dir
        followed by "_intracranial" the name convention used should be:

        >>> {case_id}_cta_intracranial.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_id}_intracranial_vessel_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.segmentation.inference.perform_inference_intracranial() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_inference_intracranial(self.case_dir)

class ThrombusSegmenter():
    """ 
    Segmenter class to perform vessel segmentation prediction over CTA. 

    """
    def __init__(self, case_dir):
        """
        Initializes object of the Segmenter class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        self.case_dir = case_dir

    def predict(self):
        """
        This method calls perform_inference_thrombus to perform inference using a trained nnunet
        model over a localized bimodal patch (NCCT + CTA) containing a suspected vessel occlusion.
        The CTA should be in nifti format, within the self.case_dir directory and with case_id being 
        the basename of the self.case_dir, the name convention used should be:

        >>> {case_id}_cta_thrombus_patch.nii.gz
        >>> {case_id}_ncct_thrombus_patch.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_id}_thrombus_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.segmentation.inference.perform_thrombus_inference() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_inference_thrombus(self.case_dir)
    
    def predict_patch_recentering(self):
        """
        This method calls perform_dynamic_iterative_thrombus_inference to perform inference using a trained nnunet
        model over a localized bimodal patch (NCCT + CTA) containing a suspected vessel occlusion.
        The CTA should be in nifti format, within the self.case_dir directory and with case_id being 
        the basename of the self.case_dir, the name convention used should be:

        >>> {case_id}_cta_thrombus_patch.nii.gz
        >>> {case_id}_ncct_thrombus_patch.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_id}_thrombus_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.segmentation.inference.perform_dynamic_iterative_thrombus_inference() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_dynamic_iterative_thrombus_inference(self.case_dir)