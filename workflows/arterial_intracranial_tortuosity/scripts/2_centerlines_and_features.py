#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
Stage 2: centerline extraction + per-centerline feature extraction.

Runs on the outputs of stage 1 in paths.output_dir. For every case directory it:

1. Detects the processing mode from the case directory contents (which of
   extracranial_vessels/ or intracranial_vessels/ holds landmarks).
2. Resolves segmentation and landmarks through a priority chain, so manual
   revisions are used automatically when present:
       segmentation_mod_minimal.nii.gz > segmentation_mod.nii.gz > segmentation.nii.gz
       landmarks_slicer_mod.json       > landmarks.json
   ("mod" = manual revision in 3D Slicer; "mod_minimal" is the contingency for
   cases whose fully revised segmentation still breaks centerline extraction.)
3. Extracts centerlines between the landmark pairs relevant to the mode
   -> <mode>/individual_centerlines/individual_centerline_<id>.vtk
4. Builds and featurizes each centerline graph
   -> <mode>/individual_centerlines/individual_centerline_<id>.pickle (+ .png)

Failed centerlines within a case do not abort the case: the remaining centerlines
are still featurized and the case is recorded in the error log with the list of
failed centerline ids. Fix the segmentation (add a _mod / _mod_minimal file) or
the landmarks (landmarks_slicer_mod.json) and re-run with --cases <id> --force.

Usage:
    python 2_centerlines_and_features.py [--cases ID ...] [--skip ID ...] [--force]
"""

import os

from arterial.io.load_and_save_operations import load_vtkpolydata
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.feature_extraction.feature_extractor import FeatureExtractor

from pipeline_utils import (
    LANDMARKS_PRIORITY,
    MODE_EXTRACRANIAL,
    MODE_INTRACRANIAL,
    SEGMENTATION_PRIORITY,
    ErrorLog,
    detect_mode_from_outputs,
    load_config,
    parse_batch_args,
    read_landmarks,
    redirect_output_to,
    resolve_first_existing,
    select_case_ids,
    setup_case_logger,
    setup_general_logger,
)

STAGE = "centerlines_and_features"

# Centerlines extracted per mode, as {centerline_id: (start_landmark, end_landmark)}.
# A pair is only extracted when both landmarks were detected for the case.
# occ = occlusion point, tica = terminal ICA, eica = extracranial (proximal) ICA.
LANDMARK_PAIRS = {
    MODE_EXTRACRANIAL: {
        "l-ica": ("l-eica", "l-tica"),
        "r-ica": ("r-eica", "r-tica"),
        "l-tica-occ": ("l-tica", "occ"),
        "r-tica-occ": ("r-tica", "occ"),
        "l-ica-occ": ("l-eica", "occ"),
        "r-ica-occ": ("r-eica", "occ"),
    },
    # Head-only scans have no cervical ICA: only the intracranial segments are extracted.
    MODE_INTRACRANIAL: {
        "l-tica-occ": ("l-tica", "occ"),
        "r-tica-occ": ("r-tica", "occ"),
    },
}

OCCLUSION_CENTERLINE_IDS = ("l-tica-occ", "r-tica-occ")


def stage_is_done(case_dir, mode):
    """Done when at least one terminal-ICA-to-occlusion centerline has been featurized."""
    centerlines_dir = os.path.join(case_dir, mode, "individual_centerlines")
    return any(os.path.isfile(os.path.join(centerlines_dir, f"individual_centerline_{cid}.pickle"))
               for cid in OCCLUSION_CENTERLINE_IDS)


def run_centerline_extraction(case_dir, mode, segmentation_nifti_path, landmarks_dict):
    centerline_extractor = CenterlineExtractor(
        case_dir=case_dir,
        mode=mode,
        segmentation_nifti_path=segmentation_nifti_path,
    )
    for centerline_id, (start, end) in LANDMARK_PAIRS[mode].items():
        if start in landmarks_dict and end in landmarks_dict:
            centerline_extractor.extract_centerline_between_endpoints(
                landmarks_dict[start], landmarks_dict[end], centerline_id, save=True)
    return list(LANDMARK_PAIRS[mode].keys())


def run_feature_extraction(case_dir, mode, cta_nifti_path, centerline_ids, sampling_distance_mm):
    feature_extractor = FeatureExtractor(
        case_dir, mode, cta_nifti_path=cta_nifti_path, sampling_distance_mm=sampling_distance_mm)
    centerlines_dir = os.path.join(case_dir, mode, "individual_centerlines")
    failed = []
    for centerline_id in centerline_ids:
        centerline_vtk_path = os.path.join(centerlines_dir, f"individual_centerline_{centerline_id}.vtk")
        if not os.path.isfile(centerline_vtk_path):
            continue
        print(f"Building and featurizing {centerline_id}")
        try:
            centerline_model = load_vtkpolydata(centerline_vtk_path)
            feature_extractor.build_and_featurize_individual_centerline_graph(
                centerline_model, centerline_id=centerline_id, save=True)
        except Exception as e:
            print(f"Error building and featurizing {centerline_id}: {e}")
            failed.append(centerline_id)
    if failed:
        raise ValueError(f"Errors building and featurizing centerlines: {failed}")


def process_single_case(case_dir, mode, segmentation_nifti_path, landmarks_dict,
                        sampling_distance_mm, general_logger):
    case_logger = setup_case_logger(case_dir, log_prefix="centerline_and_feature_extraction")
    with redirect_output_to(case_logger, also_print=True, additional_logger=general_logger):
        case_logger.info(f"Starting processing for case: {case_dir} (mode: {mode})")

        cta_nifti_path = os.path.join(case_dir, "cta.nii.gz")
        centerline_ids = run_centerline_extraction(case_dir, mode, segmentation_nifti_path, landmarks_dict)
        run_feature_extraction(case_dir, mode, cta_nifti_path, centerline_ids, sampling_distance_mm)

        case_logger.info(f"Completed processing for case: {case_dir}")


if __name__ == "__main__":
    args = parse_batch_args(__doc__)
    config = load_config(args.config)
    output_dir = config["paths"]["output_dir"]
    logs_dir = config["paths"]["logs_dir"]
    sampling_distance_mm = config["params"].get("sampling_distance_mm", 0.5)

    os.makedirs(logs_dir, exist_ok=True)
    general_logger = setup_general_logger(logs_dir, log_prefix=f"batch_{STAGE}")

    with redirect_output_to(general_logger, also_print=False):
        general_logger.info("Beginning centerline and feature extraction")
        general_logger.info("================================================")
        general_logger.info(f"  Output directory: {output_dir}")
        general_logger.info(f"  Sampling distance (mm): {sampling_distance_mm}")
        general_logger.info(f"  Segmentation priority: {SEGMENTATION_PRIORITY}")
        general_logger.info(f"  Landmarks priority: {LANDMARKS_PRIORITY}")
        general_logger.info("================================================")

        candidate_ids = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        case_ids = select_case_ids(candidate_ids, args, general_logger)
        general_logger.info(f"Processing {len(case_ids)} cases")

        error_log = ErrorLog(logs_dir, STAGE)
        for case_id in case_ids:
            try:
                case_dir = os.path.join(output_dir, case_id)

                mode = detect_mode_from_outputs(case_dir, LANDMARKS_PRIORITY)
                if mode is None:
                    general_logger.error(f"Skipping case {case_id}: no landmarks found (run stage 1 first)")
                    continue

                if stage_is_done(case_dir, mode) and not args.force:
                    general_logger.info(f"Skipping case {case_id}: already processed (use --force to redo)")
                    continue

                mode_dir = os.path.join(case_dir, mode)
                segmentation_nifti_path = resolve_first_existing(mode_dir, SEGMENTATION_PRIORITY)
                landmarks_json_path = resolve_first_existing(mode_dir, LANDMARKS_PRIORITY)
                general_logger.info(f"Processing case: {case_id} (mode: {mode})")
                general_logger.info(f"  Using segmentation: {os.path.basename(segmentation_nifti_path)}")
                general_logger.info(f"  Using landmarks: {os.path.basename(landmarks_json_path)}")

                landmarks_dict = read_landmarks(landmarks_json_path)
                process_single_case(case_dir, mode, segmentation_nifti_path, landmarks_dict,
                                    sampling_distance_mm, general_logger)
                general_logger.info(f"Completed processing for case: {case_id}")
            except Exception as e:
                general_logger.error(f"Error processing case {case_id}: {e}", exc_info=True)
                error_log.record(case_id, e)
                continue

        error_log.summarize(general_logger)
