#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.segmentation.segmenter import VesselSegmenter
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.vessel_labelling.vessel_labeller import VesselLabeller
from arterial.feature_extraction.feature_extractor import FeatureExtractor
from arterial.access_prediction.access_predictor import AccessPredictor
from arterial.landmark_detection.landmark_detector import LandmarkDetector

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
            - cta_nifti_path : string or path-like object
                Path to the CTA nifti to be processed.
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
        self.mode = args.mode
        self.cta_nifti_path = args.cta_nifti_path
        if self.cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        self.sampling_distance_mm = args.sampling_distance_mm
        self.skip_segmentation = args.skip_segmentation
        self.fast_segmentation = args.fast_segmentation
        self.skip_centerline_extraction = args.skip_centerline_extraction
        self.skip_branching = args.skip_branching
        self.skip_clipping = args.skip_clipping
        self.skip_vessel_labelling = args.skip_vessel_labelling
        self.skip_feature_extraction = args.skip_feature_extraction
        self.skip_access_prediction = args.skip_access_prediction
        self.skip_landmark_detection = args.skip_landmark_detection
        self.use_vanilla_nnunet = not args.cl_dice_nnunet
        self.no_slicing = args.no_slicing
        self.set_threshold_099 = args.set_threshold_099

        # Initialize module classes
        self.vessel_segmenter = VesselSegmenter(self.case_dir,
                                                self.mode,
                                                self.cta_nifti_path,
                                                self.fast_segmentation,
                                                self.use_vanilla_nnunet,
                                                self.no_slicing,
                                                self.set_threshold_099)
        self.centerline_extractor = CenterlineExtractor(self.case_dir, 
                                                        self.mode, 
                                                        None, 
                                                        self.fast_segmentation)
        self.vessel_labeller = VesselLabeller(self.case_dir,
                                              self.mode)
        self.landmark_detector = LandmarkDetector(self.case_dir,
                                                self.mode,
                                                self.cta_nifti_path)
        self.feature_extractor = FeatureExtractor(self.case_dir,
                                                self.mode,
                                                self.sampling_distance_mm,
                                                self.cta_nifti_path)
        self.access_predictor = AccessPredictor(self.case_dir,
                                                self.cta_nifti_path,
                                                None,
                                                ['femoral'],
                                                ['left', 'right'])

    def perform_analysis(self):
        """
        Calls wrapper method from each of the Arterial modules.

        Binary arguments from args are used to specify 

        """
        print("Performing analysis over {}. \n".format(self.cta_nifti_path))
        start = time()
        self.perform_segmentation()
        step0 = time()
        segmentation_time = step0 - start
        print("Segmentation took {:.2f} s".format(step0 - start))
        self.perform_centerline_extraction()
        step1 = time()
        centerline_extraction_time = step1 - step0
        print("Centerline extraction took {:.2f} s".format(step1 - step0))
        self.perform_landmark_detection()
        step2 = time()
        landmark_detection_time = step2 - step1
        print("Landmark detection took {:.2f} s".format(step2 - step1))
        self.perform_vessel_labelling()
        step3 = time()
        vessel_labelling_time = step3 - step2
        print("Vessel labelling took {:.2f} s".format(step3 - step2))
        feature_extraction_time = 0
        access_prediction_time = 0
        if self.mode == "extracranial_vessels":
            self.perform_feature_extraction()   
            step4 = time()
            feature_extraction_time = step4 - step3
            print("Feature extraction took {:.2f} s".format(step4 - step3))
            step5 = time()
            self.perform_access_prediction()
            access_prediction_time = step5 - step4
            print("Access prediction took {:.2f} s".format(step5 - step4))
        final_time = time()
        print("Total time for analysis: {:.2f} s".format(final_time - start))

        times = {
            "segmentation_time": segmentation_time,
            "centerline_extraction_time": centerline_extraction_time,
            "landmark_detection_time": landmark_detection_time,
            "vessel_labelling_time": vessel_labelling_time,
            "feature_extraction_time": feature_extraction_time,
            "access_prediction_time": access_prediction_time,
            "total_time": final_time - start
        }

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
        
        >>> case_dir/{self.mode}/centerlines/centerlines_{idx}.vtk
        >>> case_dir/{self.mode}/segmentations/segmentation_{idx}.vtk
        >>> case_dir/{self.mode}/branch_models/branch_model_{idx}.vtk
        >>> case_dir/{self.mode}/branch_model.vtk
        >>> case_dir/{self.mode}/clipped_models/clipped_model_{idx}.vtk
        >>> case_dir/{self.mode}/clipped_model.vtk
        >>> case_dir/{self.mode}/centerline_segments_array.npy

        Parmeters
        ---------

        Returns
        -------
        
        """
        if not self.skip_centerline_extraction:
            print("Performing centerline extraction...")
            # Applies centerline preprocessing and extraction using VMTK
            self.centerline_extractor.perform_centerline_extraction()
            # if self.mode == "extracranial_vessels":
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
                # self.centerline_extractor.perform_clipped_model_extraction()
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

        >>> case_dir/{self.mode}/segments_graph.pickle
        >>> case_dir/{self.mode}/segments_graph.png
        >>> case_dir/{self.mode}/segments_graph_pred.pickle
        >>> case_dir/{self.mode}/segments_graph_pred.png

        Parmeters
        ---------

        Returns
        -------
        
        """
        if not self.skip_vessel_labelling:
            print("Performing vessel labelling...")
            # Makes and featurizes simple graph
            self.vessel_labeller.build_segments_graph()
            # Performs inference over simple graph for vessel labelling
            self.vessel_labeller.predict_vessel_types()
            print("done \n")
        else:
            print("Skipping vessel labelling \n")

    def perform_feature_extraction(self):
        """
        Wrapper method of the feature_extraction module. Calls methods to perform feature extraction,
        including centerline graph built, feature extraction at different levels and mapping of 
        catheter pathways for patient characterization.

        At the end of the execution, the following files should be generated:

        >>> case_dir/{self.mode}/graph.pickle
        >>> case_dir/{self.mode}/graph.png
        >>> case_dir/{self.mode}/single_segments.png
        >>> case_dir/{self.mode}/supersegments/{configuration_name}.pickle
        >>> case_dir/{self.mode}/supersegments.png

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
            self.feature_extractor.build_local_graph()
            # Perform local feature extraction
            self.feature_extractor.extract_local_features()
            # Perform segment feature extraction
            self.feature_extractor.extract_segment_features()
            # Perform global feature extraction
            self.feature_extractor.extract_global_features()
            # Extract catheter pathways
            self.feature_extractor.extract_supersegments()
            print("done \n")
        else:
            print("Skipping feature extraction \n")

    def perform_access_prediction(self):
        """
        Wrapper method of the access_prediction module. Calls methods to perform access prediction,
        including preprocessing and inference.

        At the end of the execution, the following directories should be generated:

        >>> case_dir/extracranial_vessels/access_prediction/femoral_left
        >>> case_dir/extracranial_vessels/access_prediction/femoral_right

        Each of the directories will contain the following files:

        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/access_prediction.json
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/attention_map.pickle
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/attention_map.png
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/attention_map.vtk
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/dense_supersegment.pickle
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/segment_supersegment.pickle
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/global_features.json
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/combined_plot.png
        >>> case_dir/extracranial_vessels/access_prediction/{access}_{side}/preprocessed_supersegment_dict.pickle

        Parameters
        ----------

        Returns
        -------

        """
        if not self.skip_access_prediction:
            # Perform access prediction
            self.access_predictor.predict_accessibility()

    def perform_landmark_detection(self, run_centerline_extraction=True):
        """
        Wrapper method of the landmark_detection module. Calls methods to perform landmark detection,
        including preprocessing and inference.

        At the end of the execution, the following files should be generated:

        >>> case_dir/extracranial_vessels/landmarks/landmarks.json
        >>> case_dir/extracranial_vessels/landmarks/landmarks_slicer.json
        >>> case_dir/extracranial_vessels/individual_centerlines/individual_centerline_{centerline_id}.vtk
        >>> case_dir/extracranial_vessels/individual_centerlines/individual_centerline_{centerline_id}.pickle
        >>> case_dir/extracranial_vessels/individual_centerlines/individual_centerline_{centerline_id}.png
        """
        from arterial.io.load_and_save_operations import load_vtkpolydata
        feature_extractor_for_landmark_detection = FeatureExtractor(self.case_dir, self.mode, self.sampling_distance_mm, self.cta_nifti_path)
        if not self.skip_landmark_detection:
            print("Performing landmark detection...")
            self.landmark_detector.detect_landmarks_on_cta()
            print("done \n")
            if run_centerline_extraction:
                # Extract centerlines between detected landmarks/endpoints
                if self.mode == "extracranial_vessels":
                    landmark_pairs = {
                        'l-ica': ('l-eica', 'l-tica'),
                        'r-ica': ('r-eica', 'r-tica'),
                        'l-mca': ('l-tica', 'l-mca'),
                        'r-mca': ('r-tica', 'r-mca'),
                        'l-ica_mca': ('l-eica', 'l-mca'),
                        'r-ica_mca': ('r-eica', 'r-mca')
                    }
                elif self.mode == "intracranial_vessels":
                    landmark_pairs = {
                        'l-mca': ('l-tica', 'l-mca'),
                        'r-mca': ('r-tica', 'r-mca'),
                    }
                for landmark_pair_key in landmark_pairs.keys():
                    try:
                        self.centerline_extractor.extract_centerline_between_endpoints(self.landmark_detector.landmarks_ras_mm_dict[landmark_pairs[landmark_pair_key][0]], self.landmark_detector.landmarks_ras_mm_dict[landmark_pairs[landmark_pair_key][1]], landmark_pair_key, save=True)
                    except Exception as e:
                        print(f"Error extracting centerline between {landmark_pairs[landmark_pair_key][0]} and {landmark_pairs[landmark_pair_key][1]}: {e}")
                        continue

                for centerline_id in list(landmark_pairs.keys()):
                    if os.path.isfile(os.path.join(self.case_dir, f"{self.mode}/individual_centerlines", f"individual_centerline_{centerline_id}.vtk")):
                        print(f"Building and featurizing {centerline_id}")
                        centerline_model = load_vtkpolydata(os.path.join(self.case_dir, f"{self.mode}/individual_centerlines", f"individual_centerline_{centerline_id}.vtk"))
                        if centerline_model.GetNumberOfPoints() < 5:
                            print(f"Skipping graph building and featurization of {centerline_id} because it has less than 5 points")
                            continue
                        feature_extractor_for_landmark_detection.build_and_featurize_individual_centerline_graph(centerline_model, centerline_id=centerline_id, save=True)
        else:
            print("Skipping landmark detection \n")