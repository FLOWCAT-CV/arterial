#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from arterial.segmentation.segmenter import VesselSegmenter
from arterial.segmentation.utils import get_num_threads
from arterial.io.load_and_save_operations import save_json, load_json
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.vessel_labelling.vessel_labeller import VesselLabeller
from arterial.feature_extraction.feature_extractor import FeatureExtractor
from arterial.access_prediction.access_predictor import AccessPredictor
from arterial.landmark_detection.landmark_detector import LandmarkDetector

from time import time

SUPPORTED_MODES = ("extracranial_vessels", "intracranial_vessels")

# Written to case_dir/{mode}/ when the analysis of a case finishes; multi-case runs skip cases that have it
RUN_RECORD_FILENAME = "arterial_run.json"
SKIP_FLAGS = ("skip_segmentation", "skip_centerline_extraction", "skip_branching", "skip_vessel_labelling",
              "skip_feature_extraction", "skip_access_prediction", "skip_landmark_detection")
# Peak memory of one case at default settings, used to limit the number of parallel workers
GPU_MEMORY_PER_WORKER_GB = 5
RAM_PER_WORKER_GB = 6


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
            Contains the parsed command-line arguments (see arterial/cli.py) as attributes.
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
            - resampling : string, default = "scipy"
                Resampling used by nnunet in the segmentation: "scipy" (CPU) or "torch" (GPU).
            - num_threads : int, default = None
                Number of CPU threads. If None, ARTERIAL_NUM_THREADS or the usable cores capped at 8.

        Returns
        -------
        
        """
        # Parameters from args
        self.case_dir = args.case_dir
        self.mode = getattr(args, "mode", "extracranial_vessels")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode should be one of {SUPPORTED_MODES}, got {self.mode!r}")
        self.cta_nifti_path = getattr(args, "cta_nifti_path", None)
        if self.cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        self.sampling_distance_mm = getattr(args, "sampling_distance_mm", 2)
        self.skip_segmentation = getattr(args, "skip_segmentation", False)
        self.fast_segmentation = getattr(args, "fast_segmentation", False)
        self.skip_centerline_extraction = getattr(args, "skip_centerline_extraction", False)
        self.skip_branching = getattr(args, "skip_branching", False)
        self.skip_clipping = getattr(args, "skip_clipping", False)  # kept for CLI compatibility; clipping is not part of the pipeline
        self.skip_vessel_labelling = getattr(args, "skip_vessel_labelling", False)
        self.skip_feature_extraction = getattr(args, "skip_feature_extraction", False)
        self.skip_access_prediction = getattr(args, "skip_access_prediction", False)
        self.skip_landmark_detection = getattr(args, "skip_landmark_detection", False)
        self.use_vanilla_nnunet = not getattr(args, "cl_dice_nnunet", False)
        self.no_slicing = getattr(args, "no_slicing", False)
        self.set_threshold_099 = getattr(args, "set_threshold_099", False)
        self.resampling = getattr(args, "resampling", "scipy")
        self.num_threads = getattr(args, "num_threads", None)

        # Initialize module classes
        self.vessel_segmenter = VesselSegmenter(self.case_dir,
                                                self.mode,
                                                self.cta_nifti_path,
                                                self.fast_segmentation,
                                                self.use_vanilla_nnunet,
                                                self.no_slicing,
                                                self.set_threshold_099,
                                                resampling=self.resampling,
                                                num_threads=self.num_threads)
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

        The skip_* flags given at construction decide which stages run; feature extraction and
        access prediction only run in extracranial_vessels mode.

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
            self.perform_access_prediction()
            step5 = time()
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

        >>> case_dir/{mode}/segmentation.nii.gz
    
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
            # Surface model clipping (perform_clipped_model_extraction) is not part of the
            # pipeline; --skip_clipping is accepted for compatibility and has no effect.
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

        >>> case_dir/{mode}/landmarks.json
        >>> case_dir/{mode}/landmarks_slicer.json
        >>> case_dir/extracranial_vessels/individual_centerlines/individual_centerline_{centerline_id}.vtk
        >>> case_dir/extracranial_vessels/individual_centerlines/individual_centerline_{centerline_id}.pickle
        >>> case_dir/extracranial_vessels/individual_centerlines/individual_centerline_{centerline_id}.png
        """
        from arterial.io.load_and_save_operations import load_vtkpolydata
        if not self.skip_landmark_detection:
            feature_extractor_for_landmark_detection = FeatureExtractor(self.case_dir, self.mode, self.sampling_distance_mm, self.cta_nifti_path)
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
                else:
                    raise ValueError(f"No landmark pairs defined for mode {self.mode!r}")
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
                        try:
                            feature_extractor_for_landmark_detection.build_and_featurize_individual_centerline_graph(centerline_model, centerline_id=centerline_id, save=True)
                        except Exception as e:
                            print(f"Error building and featurizing {centerline_id}: {e}")
                            continue
        else:
            print("Skipping landmark detection \n")


def save_run_record(processor, times, workers=1):
    """
    Saves the run record of a case (stage times, skip flags and run settings) to case_dir/{mode}/arterial_run.json.

    Parameters
    ----------
    processor : ArterialProcessor
        Processor that ran the analysis of the case.
    times : dict
        Stage times, as returned by ArterialProcessor.perform_analysis.
    workers : int, default = 1
        Number of cases processed in parallel.

    """
    record = {
        "times": times,
        "skip_flags": {flag: getattr(processor, flag) for flag in SKIP_FLAGS},
        "resampling": processor.resampling,
        "num_threads": processor.vessel_segmenter.num_threads,
        "workers": workers,
    }
    os.makedirs(os.path.join(processor.case_dir, processor.mode), exist_ok=True)
    save_json(record, os.path.join(processor.case_dir, processor.mode, RUN_RECORD_FILENAME))

def is_case_processed(args):
    """
    Checks whether a previous run already completed the same analysis of a case, i.e. whether
    case_dir/{mode}/arterial_run.json exists and was written with the same skip flags.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, with a single case_dir.

    Returns
    -------
    processed : bool
        True if the case can be skipped.

    """
    run_record_path = os.path.join(args.case_dir, getattr(args, "mode", "extracranial_vessels"), RUN_RECORD_FILENAME)
    if not os.path.isfile(run_record_path):
        return False
    skip_flags = load_json(run_record_path).get("skip_flags", {})
    return all(skip_flags.get(flag) == getattr(args, flag, False) for flag in SKIP_FLAGS)

def get_available_ram_gb():
    """
    Reads the available RAM from /proc/meminfo.

    Returns
    -------
    available_ram_gb : float or None
        Available RAM in GB, or None if /proc/meminfo cannot be read (not Linux).

    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None

def get_max_workers(workers, n_cases):
    """
    Limits the number of parallel workers to the number of cases, the free GPU memory and the available RAM.

    Parameters
    ----------
    workers : int
        Number of workers requested.
    n_cases : int
        Number of cases to process.

    Returns
    -------
    workers : int
        Number of workers to use (at least 1).

    """
    workers = max(1, min(workers, n_cases))
    if torch.cuda.is_available():
        free_gpu_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if workers * GPU_MEMORY_PER_WORKER_GB > free_gpu_gb:
            fitting = max(1, int(free_gpu_gb // GPU_MEMORY_PER_WORKER_GB))
            print(f"Warning: {free_gpu_gb:.1f} GB of free GPU memory fit {fitting} worker(s) of ~{GPU_MEMORY_PER_WORKER_GB} GB, using {fitting} instead of {workers}")
            workers = fitting
    available_ram_gb = get_available_ram_gb()
    if available_ram_gb is not None and workers * RAM_PER_WORKER_GB > available_ram_gb:
        fitting = max(1, int(available_ram_gb // RAM_PER_WORKER_GB))
        print(f"Warning: {available_ram_gb:.1f} GB of available RAM fit {fitting} worker(s) of ~{RAM_PER_WORKER_GB} GB, using {fitting} instead of {workers}")
        workers = fitting
    return workers

def _process_single_case(args, workers=1):
    """
    Runs the analysis of one case and saves its run record. Top-level function so that worker processes can run it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, with a single case_dir.
    workers : int, default = 1
        Number of cases processed in parallel, stored in the run record.

    Returns
    -------
    times : dict
        Stage times, as returned by ArterialProcessor.perform_analysis.

    """
    processor = ArterialProcessor(args)
    times = processor.perform_analysis()
    save_run_record(processor, times, workers)
    return times

def process_cases(args):
    """
    Runs the analysis of several cases (args.case_dir is a list), args.workers of them in parallel on the same GPU.
    Cases that a previous run already completed with the same skip flags are skipped, so an interrupted run can be
    resumed. The CPU threads (args.num_threads) are divided between the workers.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, with a list of case directories as case_dir.

    Returns
    -------
    results : dict
        Stage times of every case that finished in this run, and the error message of every case that failed, by case_dir.

    """
    case_args = [argparse.Namespace(**{**vars(args), "case_dir": case_dir}) for case_dir in args.case_dir]
    case_args = [single_case_args for single_case_args in case_args if not is_case_processed(single_case_args)]
    print(f"{len(args.case_dir)} cases, {len(args.case_dir) - len(case_args)} already processed, {len(case_args)} to process")
    if not case_args:
        return {}

    workers = get_max_workers(getattr(args, "workers", 1), len(case_args))
    total_threads, usable_cores, source = get_num_threads(getattr(args, "num_threads", None))
    threads_per_worker = max(1, total_threads // workers)
    print(f"Using {workers} worker(s) with {threads_per_worker} CPU thread(s) each (usable cores {usable_cores}, source: {source})")
    for single_case_args in case_args:
        single_case_args.num_threads = threads_per_worker
    results = {}
    if workers == 1:
        for single_case_args in case_args:
            try:
                results[single_case_args.case_dir] = _process_single_case(single_case_args)
            except Exception as e:
                print(f"Error processing {single_case_args.case_dir}: {e}")
                results[single_case_args.case_dir] = f"error: {e}"
    else:
        # "spawn" so that every worker initializes CUDA on its own
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_process_single_case, single_case_args, workers): single_case_args.case_dir for single_case_args in case_args}
            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
                    results[futures[future]] = f"error: {e}"

    n_failed = sum(isinstance(result, str) for result in results.values())
    print(f"Finished {len(results) - n_failed} of {len(results)} cases ({n_failed} failed)")
    return results
