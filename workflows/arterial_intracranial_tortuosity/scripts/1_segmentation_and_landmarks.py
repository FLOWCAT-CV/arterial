#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
Stage 1: vessel segmentation + landmark detection (GPU required).

For every case directory in paths.input_dir containing a CTA NIfTI (cta.nii.gz /
cta.nii, or cta_intracranial.nii.gz for head-only scans), this script:

1. Detects the processing mode (extracranial_vessels for cervico-cerebral CTAs,
   intracranial_vessels for head-only CTAs) — see pipeline_utils.detect_mode.
2. Reorients the CTA to LPS and saves it as <output_dir>/<case_id>/cta.nii.gz.
3. Runs nnU-Net vessel segmentation  -> <case_id>/<mode>/segmentation.nii.gz
4. Runs landmark detection           -> <case_id>/<mode>/landmarks.json
                                        <case_id>/<mode>/landmarks_slicer.json

Cases whose segmentation and landmarks already exist are skipped unless --force.
Per-case failures are logged to the error log and do not stop the batch.

Usage:
    python 1_segmentation_and_landmarks.py [--cases ID ...] [--skip ID ...] [--force]
"""

import os
import shutil

from arterial.io.dicom_and_nifti import convert_orientation
from arterial.io.load_and_save_operations import save_nifti
from arterial.segmentation.segmenter import VesselSegmenter
from arterial.landmark_detection.landmark_detector import LandmarkDetector

from pipeline_utils import (
    LANDMARKS_PRIORITY,
    MODE_INTRACRANIAL,
    ErrorLog,
    detect_mode,
    find_input_cta,
    load_config,
    parse_batch_args,
    redirect_output_to,
    resolve_first_existing,
    select_case_ids,
    setup_case_logger,
    setup_general_logger,
)

STAGE = "segmentation_and_landmarks"


def stage_is_done(case_dir, mode):
    # Any landmarks file counts: manually revised cases may only keep landmarks_slicer_mod.json.
    mode_dir = os.path.join(case_dir, mode)
    return (os.path.isfile(os.path.join(mode_dir, "segmentation.nii.gz"))
            and resolve_first_existing(mode_dir, LANDMARKS_PRIORITY) is not None)


def process_single_case(case_id, case_dir, mode, src_cta_path, general_logger):
    general_logger.info(f"Converting CTA orientation from {src_cta_path} to LPS")
    cta_nifti_path = os.path.join(case_dir, "cta.nii.gz")
    save_nifti(convert_orientation(src_cta_path, target_orientation="LPS"), cta_nifti_path)

    case_logger = setup_case_logger(case_dir, log_prefix="vessel_segmentation_landmark")
    with redirect_output_to(case_logger, also_print=True, additional_logger=general_logger):
        case_logger.info(f"Starting processing for case: {case_dir} (mode: {mode})")

        # Head-only scans skip the head/neck slicing step and are segmented whole.
        segmenter = VesselSegmenter(
            case_dir=case_dir,
            mode=mode,
            cta_nifti_path=cta_nifti_path,
            no_slicing=(mode == MODE_INTRACRANIAL),
        )
        segmenter.segment_vessels_from_cta(save=True)

        if mode == MODE_INTRACRANIAL:
            # Keep a copy of the head CTA under the name downstream analyses expect.
            head_cta_path = os.path.join(case_dir, "cta_intracranial.nii.gz")
            segmenter.save_head_cta_nifti(head_cta_path)
            shutil.copy(head_cta_path, os.path.join(case_dir, f"{case_id}_cta_intracranial.nii.gz"))

        landmark_detector = LandmarkDetector(
            case_dir=case_dir,
            mode=mode,
            cta_nifti_path=cta_nifti_path,
        )
        landmark_detector._load_cta_nifti_from_nib(segmenter.cta_nifti)
        landmark_detector._load_segmentation_nifti_from_nib(segmenter.segmentation_nifti)
        landmark_detector.detect_landmarks_on_cta(return_mask=False, save=True, use_segmentation_model=True)

        case_logger.info(f"Completed processing for case: {case_dir}")


if __name__ == "__main__":
    args = parse_batch_args(__doc__)
    config = load_config(args.config)
    input_dir = config["paths"]["input_dir"]
    output_dir = config["paths"]["output_dir"]
    logs_dir = config["paths"]["logs_dir"]
    configured_mode = config["params"].get("mode", "auto")
    min_coverage_mm = config["params"].get("extracranial_min_si_coverage_mm", 250)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    general_logger = setup_general_logger(logs_dir, log_prefix=f"batch_{STAGE}")

    with redirect_output_to(general_logger, also_print=False):
        general_logger.info("Beginning vessel segmentation and landmark detection")
        general_logger.info("================================================")
        general_logger.info(f"  Input directory: {input_dir}")
        general_logger.info(f"  Output directory: {output_dir}")
        general_logger.info(f"  Mode: {configured_mode}")
        general_logger.info("================================================")

        candidate_ids = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        case_ids = select_case_ids(candidate_ids, args, general_logger)
        general_logger.info(f"Processing {len(case_ids)} cases")

        error_log = ErrorLog(logs_dir, STAGE)
        for case_id in case_ids:
            try:
                src_cta_path = find_input_cta(os.path.join(input_dir, case_id))
                if src_cta_path is None:
                    raise FileNotFoundError(
                        f"No input CTA found for case {case_id} in {os.path.join(input_dir, case_id)}")

                case_dir = os.path.join(output_dir, case_id)
                os.makedirs(case_dir, exist_ok=True)
                mode = detect_mode(case_dir, src_cta_path, configured_mode, min_coverage_mm, general_logger)

                if stage_is_done(case_dir, mode) and not args.force:
                    general_logger.info(f"Skipping case {case_id}: already processed (use --force to redo)")
                    continue

                general_logger.info(f"Processing case: {case_id}")
                process_single_case(case_id, case_dir, mode, src_cta_path, general_logger)
                general_logger.info(f"Completed processing for case: {case_id}")
            except Exception as e:
                general_logger.error(f"Error processing case {case_id}: {e}", exc_info=True)
                error_log.record(case_id, e)
                continue

        error_log.summarize(general_logger)
