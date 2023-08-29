#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.centerline_extraction.run_centerline_extraction_slicer import perform_preprocessing_and_centerline_extraction
from arterial.centerline_extraction.postprocessing.branch_and_clipped_model_extraction import perform_centerline_branching, perform_surface_model_clipping
from arterial.centerline_extraction.postprocessing.postprocessing import compute_centerline_segments_array, perform_segmentation_unification

class CenterlineExtractor():
    """
    CenterlineExtractor class to perform centerline extraction over predicted
    segmentation.   
        
    """
    def __init__(self, case_dir, mode = "vessels", no_display = False, fast_segmentation = False):
        """
        Initializes object of the CenterlineExtractor class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory.
        mode: string, default = "vessels"
            Determines whether the centerline is extracted from ```vessels```, ```intracranial_vessels``` 
            or ```thrombus```. 
        no_display : bool, default = False
            Boolean variable to be used when running analysis on a headless server.
            In addition, add ```$xvfb-run --auto-servernum --server-num=1``` at the beggining
            of the command line call when executing the script from the command line.
        fast_segmentation : bool, default = False
            Boolean variable to be used when running analysis derived from fast segmentation (lowres).
            In this case, segmentation of the cerebral arteries is less reliable, so a higher fraction
            of vessels is ignored.

        Returns
        -------
        
        """
        self.case_dir = case_dir
        self.mode = mode
        self.no_display = no_display
        self.fast_segmentation = fast_segmentation
    
    def extract_centerline(self):
        """
        Runs preprocessing and centerline extraction, including analysis for circular
        centerlines, all using Slicer and SlicerVMTK functions. Since PythonSlicer
        functions and Slicer GUI elements are used to run the analysis, there is a need for an 
        intermediate script that runs a command line command. We use os.system() to do that.        

        At the end of the execution, the following files should be generated:

        >>> case_dir/centerlines/centerlines{idx}.vtk
        >>> case_dir/segmentations/segmentation{idx}.vtk
        
        Acts as a wrapper for the arterial.centerline_extraction.run_centerline_extraction_slicer.
            perform_preprocessing_and_centerline_extraction() function.

        Parameters
        ----------

        Returns
        -------

        """
        perform_preprocessing_and_centerline_extraction(self.case_dir, self.mode, self.no_display, self.fast_segmentation)

    def extract_branch_model(self):
        """
        Runs centerline model branching through vmtk command ```vmtkbranchextractor```.

        At the end of the execution, the following files should be generated:

        >>> case_dir/branch_models/branch_model{idx}.vtk
        >>> case_dir/branch_model.vtk
        
        Acts as a wrapper for the arterial.centerline_extraction.branch_extraction.
            branch_and_clipped_model_extraction.perform_centerline_branching() function.
        
        Parameters
        ----------

        Returns
        -------

        """
        perform_centerline_branching(self.case_dir)

    def extract_clipped_model(self):
        """
        Runs centerline model branching through vmtk command ```vmtkbranchclipper```.

        At the end of the execution, the following files should be generated:

        >>> case_dir/clipped_models/clipped_model{idx}.vtk
        >>> case_dir/clipped_model.vtk
        
        Acts as a wrapper for the arterial.centerline_extraction.branch_extraction.
            branch_and_clipped_model_extraction.perform_surface_model_clipping() function.
        
        Parameters
        ----------

        Returns
        -------

        """
        perform_surface_model_clipping(self.case_dir)

    def postprocess_centerline(self):
        """
        Runs postprocessing of the centerlines model to generate a numpy array
        with the coordinates and radii of each centerline point for each non-overlapping
        centerline segment.

        At the end of the execution, the following files should be generated:

        >>> case_dir/centerline_segments_array.npy

        Acts as a wrapper for the arterial.centerline_extraction.postprocessing.postprocessing.
            compute_centerline_segments_array() function.

        Parameters
        ----------
        
        """
        compute_centerline_segments_array(self.case_dir)