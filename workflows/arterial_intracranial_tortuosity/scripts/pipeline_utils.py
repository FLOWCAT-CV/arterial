#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
Shared helpers for the distal-occlusion arterial pipeline.

Covers four concerns used by all stage scripts:
- config loading and the common command-line interface (--cases / --skip / --force)
- automatic extracranial vs intracranial mode detection
- file-resolution priority chains (manually revised files take precedence over automatic ones)
- logging: every case gets its own log file inside its case directory, plus a batch-level
  log and a JSON error log in logs_dir. Errors never stop the batch.

Do not replace the StreamToLogger machinery with logging.basicConfig: it deliberately
fans out stdout/stderr to both the per-case file and the batch logger so that prints
from inside third-party libraries (nnU-Net, VMTK, ...) are captured as well.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime

import nibabel as nib
import pandas as pd
import yaml

MODE_EXTRACRANIAL = "extracranial_vessels"
MODE_INTRACRANIAL = "intracranial_vessels"
MODES = (MODE_EXTRACRANIAL, MODE_INTRACRANIAL)

# Input CTA naming conventions accepted inside each case's input directory.
# "cta_intracranial" in the filename forces intracranial mode.
INPUT_CTA_CANDIDATES = [
    "cta.nii.gz",
    "cta.nii",
    "cta_intracranial.nii.gz",
    "cta_intracranial.nii",
]

# Priority chains for stage 2 inputs (first existing file wins). The "_mod" files are
# manual revisions made in 3D Slicer; "_mod_minimal" is the contingency for cases where
# the full manual revision itself breaks centerline extraction — a minimal edit of the
# automatic segmentation that only opens/closes what is strictly needed.
SEGMENTATION_PRIORITY = [
    "segmentation_mod_minimal.nii.gz",
    "segmentation_mod.nii.gz",
    "segmentation.nii.gz",
]
LANDMARKS_PRIORITY = [
    "landmarks_slicer_mod.json",
    "landmarks.json",
]


# ---------------------------------------------------------------------------
# Config and CLI
# ---------------------------------------------------------------------------

def load_config(config_path=None):
    """Load config.yaml (defaults to the one next to the scripts)."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_batch_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml next to the scripts)")
    parser.add_argument("--cases", nargs="*", default=[],
                        help="Process only these case ids (default: every case directory found)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Case ids to skip")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess cases even when their outputs already exist")
    return parser.parse_args()


def select_case_ids(candidate_ids, args, logger):
    """Apply --cases / --skip to the discovered case ids. --cases wins over discovery."""
    case_ids = sorted(candidate_ids)
    if args.cases:
        logger.info(f"Restricting to cases given on the command line: {args.cases}")
        case_ids = list(args.cases)
    if args.skip:
        logger.info(f"Skipping cases given on the command line: {args.skip}")
        case_ids = [case_id for case_id in case_ids if case_id not in args.skip]
    return case_ids


# ---------------------------------------------------------------------------
# Mode detection and file resolution
# ---------------------------------------------------------------------------

def find_input_cta(case_input_dir):
    """Return the input CTA path for a case, or None if no accepted filename exists."""
    for name in INPUT_CTA_CANDIDATES:
        path = os.path.join(case_input_dir, name)
        if os.path.isfile(path):
            return path
    return None


def si_coverage_mm(nifti_path):
    """Physical superior-inferior coverage of a NIfTI volume in mm, read from the header only."""
    img = nib.load(nifti_path)
    shape = img.shape[:3]
    # Row 2 of the affine maps voxel indices to the world z (superior-inferior) axis,
    # regardless of the on-disk voxel axis order.
    spans = abs(img.affine[:3, :3]).dot(shape)
    return float(spans[2])


def detect_mode(case_dir, input_cta_path, configured_mode, min_extracranial_coverage_mm, logger):
    """
    Decide extracranial_vessels vs intracranial_vessels for one case.

    Order of precedence:
    1. An explicit mode in config.yaml (params.mode other than "auto").
    2. Existing stage-1 outputs in case_dir (keeps re-runs consistent with the first run).
    3. The input filename: "cta_intracranial*" means a head-only scan was provided.
    4. Superior-inferior coverage of the input CTA: a cervico-cerebral (arch-to-vertex)
       CTA covers well over min_extracranial_coverage_mm; a head-only CTA does not.
    """
    if configured_mode in MODES:
        return configured_mode
    if configured_mode not in (None, "auto"):
        raise ValueError(f"params.mode must be 'auto' or one of {MODES}, got: {configured_mode}")

    existing_output_markers = LANDMARKS_PRIORITY + ["segmentation.nii.gz"]
    for mode in MODES:
        if any(os.path.isfile(os.path.join(case_dir, mode, name)) for name in existing_output_markers):
            logger.info(f"Mode {mode} detected from existing outputs in {case_dir}")
            return mode

    if input_cta_path is None:
        raise FileNotFoundError(f"No input CTA and no existing outputs for case dir: {case_dir}")

    if "cta_intracranial" in os.path.basename(input_cta_path):
        logger.info(f"Mode {MODE_INTRACRANIAL} detected from input filename: {input_cta_path}")
        return MODE_INTRACRANIAL

    coverage = si_coverage_mm(input_cta_path)
    mode = MODE_EXTRACRANIAL if coverage >= min_extracranial_coverage_mm else MODE_INTRACRANIAL
    logger.info(f"Mode {mode} detected from scan coverage: {coverage:.0f} mm superior-inferior "
                f"(threshold {min_extracranial_coverage_mm} mm)")
    return mode


def detect_mode_from_outputs(case_dir, marker_filenames):
    """
    Detect the mode of an already-processed case from its directory contents:
    the mode whose subdirectory contains one of marker_filenames (extracranial preferred,
    since extracranial runs also produce the intracranial segment features).
    Returns None when neither subdirectory qualifies.
    """
    for mode in MODES:
        for name in marker_filenames:
            if os.path.isfile(os.path.join(case_dir, mode, name)):
                return mode
    return None


def resolve_first_existing(directory, filenames):
    """Return the first existing file among directory/filenames, or None."""
    for name in filenames:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def read_landmarks(landmarks_json_path):
    """
    Read a landmarks file into {label: [x, y, z] in RAS mm}.
    Handles both the pipeline's plain-dict format and 3D Slicer markup files
    (detected by content, converting LPS coordinates to RAS when needed).
    """
    with open(landmarks_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "markups" in data:  # 3D Slicer markups file
        markup = data["markups"][0]
        flip = markup.get("coordinateSystem") == "LPS"
        landmarks = {}
        for point in markup["controlPoints"]:
            x, y, z = point["position"]
            landmarks[point["label"]] = [-x, -y, z] if flip else [x, y, z]
        return landmarks

    return data


# ---------------------------------------------------------------------------
# Cases table
# ---------------------------------------------------------------------------

# Accepted column names in the cases table, and side spellings (Catalan included
# for compatibility with the original VHIR registry exports).
CASE_ID_COLUMNS = ("case_id", "proces_id", "pat_id")
SIDE_COLUMNS = ("occlusion_side", "lateralitat_ictus")
SIDE_ALIASES = {
    "left": "l", "l": "l", "esquerra": "l",
    "right": "r", "r": "r", "dreta": "r",
}


def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def write_table(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)


def find_column(df, accepted_names, what):
    for name in accepted_names:
        if name in df.columns:
            return name
    raise KeyError(f"Cases table is missing a {what} column (accepted names: {accepted_names})")


def normalize_case_id(value):
    """Case ids may be read as floats from Excel (e.g. 1234567.0)."""
    return str(value).removesuffix(".0")


def normalize_side(value):
    """Return 'l'/'r', or None when the value is not a recognized side."""
    return SIDE_ALIASES.get(str(value).strip().lower())


# ---------------------------------------------------------------------------
# Error log
# ---------------------------------------------------------------------------

class ErrorLog:
    """Accumulates per-case errors into a timestamped JSON file, rewritten after every error."""

    def __init__(self, logs_dir, stage_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(logs_dir, f"error_log_{stage_name}_{timestamp}.json")
        self.errors = {}

    def record(self, case_id, exception):
        self.errors[case_id] = {
            "error": str(exception),
            "traceback": traceback.format_exc().splitlines(),
        }
        with open(self.path, "w") as f:
            json.dump(self.errors, f, indent=4)

    def summarize(self, logger):
        logger.info("================================================")
        logger.info("Batch processing completed")
        if self.errors:
            logger.info(f"Cases with errors (see {self.path}): {list(self.errors.keys())}")
        else:
            logger.info("All cases processed successfully")
        logger.info("================================================")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class StreamToLogger:
    """Redirect stdout/stderr to a logger while preserving terminal output."""

    def __init__(self, logger, log_level=logging.INFO, also_print=True,
                 additional_logger=None, additional_logger_file_only=True):
        self.logger = logger
        self.log_level = log_level
        self.also_print = also_print
        self.additional_logger = additional_logger
        self.additional_logger_file_only = additional_logger_file_only
        self._original_stream = sys.__stdout__ if log_level == logging.INFO else sys.__stderr__

        # If file_only, extract file handlers from additional_logger to avoid console duplication
        self._additional_file_handlers = []
        if additional_logger and additional_logger_file_only:
            for handler in additional_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    self._additional_file_handlers.append(handler)

    def write(self, buf):
        # Always print to original terminal so output is visible
        if self.also_print and buf:
            self._original_stream.write(buf)
            self._original_stream.flush()

        # Log to primary logger and optional additional logger
        for line in buf.rstrip().split('\n'):
            if line:
                self.logger.log(self.log_level, line)
                if self.additional_logger:
                    if self.additional_logger_file_only:
                        # Write directly to file handlers only (skip console)
                        record = self.additional_logger.makeRecord(
                            self.additional_logger.name, self.log_level, '', 0, line, None, None
                        )
                        for handler in self._additional_file_handlers:
                            handler.emit(record)
                    else:
                        self.additional_logger.log(self.log_level, line)

    def flush(self):
        if self.also_print:
            self._original_stream.flush()

    def isatty(self):
        # Return True to prevent libraries from suppressing output
        return True

    def fileno(self):
        # Return the original stream's file descriptor for compatibility
        return self._original_stream.fileno()


def setup_general_logger(logs_dir, log_prefix="general"):
    """Batch-level logger: logs to a timestamped file in logs_dir and to the console."""
    log_filepath = os.path.join(logs_dir, f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(f"general_{logs_dir}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # clear any existing handlers to avoid duplication

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.__stdout__)  # use the original stdout
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def setup_case_logger(case_dir, log_prefix="case"):
    """Case-level logger: logs to a timestamped file inside case_dir only (no console)."""
    log_filepath = os.path.join(case_dir, f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(f"case_{case_dir}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # clear any existing handlers to avoid duplication

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger


def cleanup_logger(logger, orig_stdout=None, orig_stderr=None):
    """Close logging handlers and optionally restore stdout/stderr."""
    if orig_stdout is not None:
        sys.stdout = orig_stdout
    if orig_stderr is not None:
        sys.stderr = orig_stderr

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


@contextmanager
def redirect_output_to(logger, also_print, additional_logger=None):
    """
    Route sys.stdout/sys.stderr through `logger` for the duration of the block, then
    restore the previous streams and close the logger's handlers.

    also_print=True when `logger` writes to file only (case logger) so output still
    reaches the terminal; also_print=False when `logger` already has a console handler
    (general logger).
    """
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamToLogger(logger, logging.INFO, also_print=also_print,
                                additional_logger=additional_logger)
    sys.stderr = StreamToLogger(logger, logging.ERROR, also_print=also_print,
                                additional_logger=additional_logger)
    try:
        yield
    finally:
        cleanup_logger(logger, orig_stdout, orig_stderr)
