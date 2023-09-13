#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.segmentation.segmenter import VesselSegmenter, ThrombusSegmenter
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.vessel_labelling.vessel_labeller import VesselLabeller
from arterial.feature_extraction.feature_extractor import FeatureExtractor

from time import time

class ArterialProcessor():
    """
    ArterialProcessor class to perform the desired analysis specified by the parsed arguments upon
    command line call. Wraps all modules' classes and their methods within one object.
    
    """
    def __init__(self, parser):
        """
        Initializes object of the ArterialProcessor class, and creates objects for all module central classes
        as attributes for the ArterialProcessor object.

        Parameters
        ----------
        parser : argparse.ArgumentParser object
            Contains all parsed arguments from the perform_analysis.py call as attributes.
            These are:
            - case_dir : string or path-like object
                Path to case directory. 
            - no_display : bool, default = False
                Boolean variable to be used when running analysis on a headless server.
                In addition, add ```$xvfb-run --auto-servernum --server-num=1``` at the beggining
                of the command line call when executing the script from the command line.
                E.g.: ```$xvfb-run --auto-servernum --server-num=1 python perform_analysis.py -case_dir {case_dir} -no_display {True}```
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
        # Parameters from parser
        self.case_dir = parser.case_dir
        self.mode = parser.mode
        self.no_display = parser.no_display
        self.skip_segmentation = parser.skip_segmentation
        self.fast_segmentation = parser.fast_segmentation
        self.skip_centerline_extraction = parser.skip_centerline_extraction
        self.skip_branching = parser.skip_branching
        self.skip_clipping = parser.skip_clipping
        self.skip_vessel_labelling = parser.skip_vessel_labelling
        self.skip_feature_extraction = parser.skip_feature_extraction

        # Initialize module classes
        self.vessel_segmenter = VesselSegmenter(self.case_dir)
        self.thrombus_segmenter = ThrombusSegmenter(self.case_dir)
        self.centerline_extractor = CenterlineExtractor(self.case_dir, self.mode, self.no_display, self.fast_segmentation)
        self.vessel_labeller = VesselLabeller(self.case_dir, self.mode)
        self.feature_extractor = FeatureExtractor(self.case_dir)

    def perform_analysis(self):
        """
        Calls wrapper method from each of the Arterial modules.

        Binary arguments from parser are used to specify 

        """
        print("Performing analysis over case {}. \n".format(os.path.basename(self.case_dir)))
        start = time()
        self.perform_segmentation()
        step0 = time()
        print("Segmentation took {:.2f} s".format(step0 - start))
        self.perform_centerline_extraction()
        step1 = time()
        print("Centerline extraction took {:.2f} s".format(step1 - step0))
        self.perform_vessel_labelling()
        step2 = time()
        print("Vessel labelling took {:.2f} s".format(step2 - step1))
        self.perform_feature_extraction()
        step3 = time()
        print("Feature extraction took {:.2f} s".format(step3 - step2))
        print("Total time for analysis: {:.2f} s".format(step3 - start))

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
            if self.mode == "vessels":
                if self.fast_segmentation:
                    print("Predicting segmentation (fast)...")
                    self.vessel_segmenter.predict_fast()
                    print("done \n")
                else:
                    print("Predicting segmentation (full)...")
                    self.vessel_segmenter.predict_full()
                    print("done \n")
            elif self.mode == "intracranial_vessels":
                print("Predicting intracranial segmentation...")
                self.vessel_segmenter.predict_intracranial()
                print("done \n")
            elif self.mode == "thrombus":
                print("Predicting thrombus segmentation...")
                self.thrombus_segmenter.predict_patch_recentering()
                print("done \n")
        else:
            print("Skipping segmentation \n")

    def perform_centerline_extraction(self):
        """
        Wrapper method of the centerline_extraction module. Calls methods to perform centerline
        extraction, model branching and clipping as well as postprocessing of the centerline models.

        At the end of the execution, the following files should be generated:
        
        >>> case_dir/centerlines/centerlines{idx}.vtk
        >>> case_dir/segmentations/segmentation{idx}.vtk
        >>> case_dir/branch_models/branch_model{idx}.vtk
        >>> case_dir/branch_model.vtk
        >>> case_dir/clipped_models/clipped_model{idx}.vtk
        >>> case_dir/clipped_model.vtk
        >>> case_dir/centerline_segments_array.npy

        Parmeters
        ---------

        Returns
        -------
        
        """
        if not self.skip_centerline_extraction:
            print("Performing centerline extraction...")
            # Applies centerline preprocessing and extraction using Slicer and VMTK
            self.centerline_extractor.extract_centerline()
            if self.mode == "vessels":
                if not self.skip_branching:
                    print("Performing centerline branching...")
                    # Performs centerline model branching with VMTK
                    self.centerline_extractor.extract_branch_model()
                    print("done")
                else:
                    print("Skipping centerline branching")
                if not self.skip_clipping:
                    print("Performing surface model clipping...")
                    # Performs surface model clipping with VMTK
                    self.centerline_extractor.extract_clipped_model()
                    print("done")
                else:
                    print("Skipping surface model clipping")
            # Creates array for easier centerline analysis
            self.centerline_extractor.postprocess_centerline()
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