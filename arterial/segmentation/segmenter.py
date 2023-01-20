#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.segmentation.inference import perform_inference

class Segmenter():
    """ 
    Segmenter class to perform segmentation prediction over CTA. 

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
        This method calls perform_inference to perform inference using a trained nnunet
        model over the original CTA. The CTA should be in nifti format, within the 
        self.case_dir directory and with case_id being the basename of the self.case_dir,
        the name convention used should be:

        >>> {case_id}.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_id}_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.segmentation.inference.perform_inference() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_inference(self.case_dir)