#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.segmentation.segmenter import VesselSegmenter
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
# from arterial.vessel_labelling.vessel_labeller import VesselLabeller
from arterial.feature_extraction.feature_extractor import FeatureExtractor

from time import time

class ArterialProcessor():
    """
    ArterialProcessor class to perform the desired analysis specified by the parsed arguments upon
    command line call. Wraps all modules' classes and their methods within one object.
    
    """
    def __init__(self, args):
        """
        Initializes object of the ArterialProcessor class, and creates objects for all module central classes
        as attributes for the ArterialProcessor object.

        Parameters
        ----------
        args : argparse.ArgumentParser object
            Contains all parsed arguments from the perform_analysis.py call as attributes.
            These are:
            - case_dir : string or path-like object
                Path to case directory. 
            - skip_segmentation : bool, default = False
                If True, it will skip the segmentation process. Useful if segmentation is already done,
                as segmentation is time-consuming. False by default.
            - skip_centerline_extraction : bool, default = False
                If True, it it will skip centerline extraction entirely. False by default.
            - skip_branching : bool, default = False
                If True, it it will skip centerline model branching. False by default.
            - skip_clipping : bool, default = False
                If True, it it will skip surface model clipping. False by default.
            - skip_vessel_labelling : bool, default = False
                If True, it it will skip vessel labelling. False by default.
            - skip_feature_extraction : bool, default = False
                If True, it it will skip feature extraction. False by default.

        Returns
        -------
        
        """
        # Parameters from args
        self.case_dir = args.case_dir
        self.cta_nifti_path = args.cta_nifti_path
        self.mode = args.mode
        self.skip_segmentation = args.skip_segmentation
        self.fast_segmentation = args.fast_segmentation
        self.skip_centerline_extraction = args.skip_centerline_extraction
        self.skip_branching = args.skip_branching
        self.skip_clipping = args.skip_clipping
        self.skip_vessel_labelling = args.skip_vessel_labelling
        self.skip_feature_extraction = args.skip_feature_extraction

        # Initialize module classes
        self.vessel_segmenter = VesselSegmenter(self.case_dir,
                                                self.mode,
                                                self.cta_nifti_path,
                                                self.fast_segmentation)
        self.centerline_extractor = CenterlineExtractor(self.case_dir, 
                                                        self.mode, 
                                                        None, 
                                                        self.no_display, 
                                                        self.fast_segmentation)
        # self.vessel_labeller = VesselLabeller(self.case_dir, # Compatibility issue with pytorch
        #                                       self.mode)
        self.vessel_labeller = None
        self.feature_extractor = FeatureExtractor(self.case_dir)

    def perform_analysis(self):
        """
        Calls wrapper method from each of the Arterial modules.

        Binary arguments from args are used to specify 

        """
        print("Performing analysis over case {}. \n".format(os.path.basename(self.case_dir)))
        start = time()
        self.perform_segmentation()
        step0 = time()
        segmentation_time = step0 - start
        print("Segmentation took {:.2f} s".format(step0 - start))
        self.perform_centerline_extraction()
        step1 = time()
        centerline_extraction_time = step1 - step0
        print("Centerline extraction took {:.2f} s".format(step1 - step0))
        self.perform_vessel_labelling()
        step2 = time()
        vessel_labelling_time = step2 - step1
        print("Vessel labelling took {:.2f} s".format(step2 - step1))
        if self.mode == "extracranial_vessels":
            self.perform_feature_extraction()   
        step3 = time()
        feature_extraction_time = step3 - step2
        print("Feature extraction took {:.2f} s".format(step3 - step2))
        print("Total time for analysis: {:.2f} s".format(step3 - start))

        times = [segmentation_time, centerline_extraction_time, vessel_labelling_time, feature_extraction_time]

        return times

    def perform_segmentation(self):
        """
        Wrapper method of the segmentation module. Calls method to perform segmentation.

        At the end of the execution, the following files should be generated:

        >>> case_dir/{os.path.basename(case_dir)}_segmentation.nii.gz
    
        Parmeters
        ---------

        Returns
        -------

        """
        # Predicts segmentation by nnunet inference
        if not self.skip_segmentation:
            self.vessel_segmenter.segment_vessels_from_cta()
        else:
            print("Skipping segmentation \n")

    def perform_centerline_extraction(self):
        """
        Wrapper method of the centerline_extraction module. Calls methods to perform centerline
        extraction, model branching and clipping as well as postprocessing of the centerline models.

        At the end of the execution, the following files should be generated:
        
        >>> case_dir/centerlines/{self.mode}_centerlines_{idx}.vtk
        >>> case_dir/segmentations/{self.mode}_segmentation_{idx}.vtk
        >>> case_dir/branch_models/branch_model_{idx}.vtk
        >>> case_dir/branch_model.vtk
        >>> case_dir/clipped_models/clipped_model_{idx}.vtk
        >>> case_dir/clipped_model.vtk
        >>> case_dir/{self.mode}_centerline_segments_array.npy

        Parmeters
        ---------

        Returns
        -------
        
        """
        if not self.skip_centerline_extraction:
            print("Performing centerline extraction...")
            # Applies centerline preprocessing and extraction using Slicer and VMTK
            self.centerline_extractor.perform_centerline_extraction()
            if self.mode == "extracranial_vessels":
                if not self.skip_branching:
                    print("Performing centerline branching...")
                    # Performs centerline model branching with VMTK
                    self.centerline_extractor.perform_branch_model_extraction()
                    print("done")
                else:
                    print("Skipping centerline branching")
                if not self.skip_clipping:
                    print("Performing surface model clipping...")
                    # Performs surface model clipping with VMTK
                    self.centerline_extractor.perform_clipped_model_extraction()
                    print("done")
                else:
                    print("Skipping surface model clipping")
            # Creates array for easier centerline analysis
            self.centerline_extractor.perform_centerline_postprocessing()
            print("done \n")
        else: 
            print("Skipping centerline extraction \n")

    def perform_vessel_labelling(self):
        """
        Wrapper method of the vessel_labelling module. Calls methods to perform vessel labelling,
        including preprocessing (simple graph generation) and postprocessing.

        At the end of the execution, the following files should be generated:

        >>> case_dir/graph_simple.pickle
        >>> case_dir/graph_simple.png
        >>> case_dir/graph_pred.pickle
        >>> case_dir/graph_pred.png

        Parmeters
        ---------

        Returns
        -------
        
        """
        if not self.skip_vessel_labelling:
            print("Performing vessel labelling...")
            # Makes and featurizes simple graph
            self.vessel_labeller.preprocessing()
            # Performs inference over simple graph for vessel labelling
            self.vessel_labeller.predict()
            print("done \n")
        else:
            print("Skipping vessel labelling \n")

    def perform_feature_extraction(self):
        """
        Wrapper method of the feature_extraction module. Calls methods to perform feature extraction,
        including centerline graph built, feature extraction at different levels and mapping of 
        catheter pathways for patient characterization.

        At the end of the execution, the following files should be generated:

        >>> case_dir/graph.pickle
        >>> case_dir/graph.png
        >>> case_dir/single_segments.png
        >>> case_dir/supersegments/{configuration_name}.pickle
        >>> case_dir/supersegments.png

        Additionally, if a patient_configuration.json file is available, the following files should be
        generated:

        >>> case/dir/thrombectomy_configuration/supersegment.pickle
        >>> case/dir/thrombectomy_configuration/supersegment.png

        Parameters
        ----------

        Returns
        -------

        """
        if not self.skip_feature_extraction:
            print("Performing feature extraction...")
            # Build centerline graph
            self.feature_extractor.build_graph()
            # Perform local feature extraction
            self.feature_extractor.extract_local_features()
            # Perform segment feature extraction
            self.feature_extractor.extract_segment_features()
            # Perform global feature extraction
            self.feature_extractor.extract_global_features()
            # Extract catheter pathways
            self.feature_extractor.map_catheter_pathways()
            print("done \n")
        else:
            print("Skipping feature extraction \n")