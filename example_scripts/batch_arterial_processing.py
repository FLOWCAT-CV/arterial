#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import shutil
import sys
import json
import logging
from datetime import datetime
from test_arterial_processing import main
import pandas as pd
import traceback

SRC_DIR = ""
DST_DIR = ""
XLSX_PATH = ""
TEMPLATE_PARAMETERS_FILE_PATH = os.path.join(os.environ["arterial_dir"], "../example_scripts", "arterial_processing_params.json")
USE_SYMLINKS = True

# Connect to the database. Use any logic here, as long as you have an iterator with identifiers and existing images it'll be fine
# Also, you will have to define a logic for the filename
df = pd.read_excel(XLSX_PATH)

# Custom class to redirect stdout/stderr to logger
class StreamToLogger:
    """Redirect print statements to logger"""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().split('\n'):
            if line:
                self.logger.log(self.log_level, line)

    def flush(self):
        pass

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_filename = f"batch_arterial_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()  # Also log to console
        ]
    )
    return log_filepath

# Initialize logging
log_filepath = setup_logging()
logger = logging.getLogger(__name__)

# Redirect stdout and stderr to logger
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

logger.info("Starting batch arterial processing")
logger.info(f"Log file: {log_filepath}")

logger.info(f"Source directory: {SRC_DIR}")
logger.info(f"Results directory: {DST_DIR}")
logger.info(f"Template parameters file: {TEMPLATE_PARAMETERS_FILE_PATH}")

try:
    with open(TEMPLATE_PARAMETERS_FILE_PATH, 'r') as f:
        template_params = json.load(f)
    logger.info("Successfully loaded template parameters")
except Exception as e:
    logger.error(f"Failed to load template parameters from {TEMPLATE_PARAMETERS_FILE_PATH}: {e}")
    raise

error_log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
if os.path.exists(error_log_file_path):
    with open(error_log_file_path, 'r') as f:
        error_log = json.load(f)
else:
    error_log = {}

total_cases = 0
processed_cases = 0
skipped_cases = 0
failed_cases = 0

try:
    total_cases = len(df)
    logger.info(f"Found {total_cases} cases to process")
    
    for idx, row in df.iterrows():
        case_id = row.proces_id
        series_uid = row.series_uid
        logger.info(f"Processing case {case_id}")
        
        try:
            case_dir = os.path.join(DST_DIR, str(case_id))

            # Edit this to match your logic for the filename
            cta_src_nifti_path = os.path.join(SRC_DIR, f"{series_uid}.nii.gz")
            cta_dst_nifti_path = os.path.join(case_dir, "cta.nii.gz")

            # Create symlink or copy the file
            os.makedirs(case_dir, exist_ok=True)
            if USE_SYMLINKS:
                if not os.path.exists(cta_dst_nifti_path):
                    os.symlink(cta_src_nifti_path, cta_dst_nifti_path)
                else:
                    logger.info(f"  Symlink already exists for case {case_id}: {cta_dst_nifti_path}")
            else:
                if not os.path.exists(cta_dst_nifti_path):
                    shutil.copy(cta_src_nifti_path, cta_dst_nifti_path)
                else:
                    logger.info(f"  File already exists for case {case_id}: {cta_dst_nifti_path}")
                
            # Check if already processed
            params_file = os.path.join(case_dir, "arterial_processing_params.json")
            if os.path.exists(params_file):
                logger.info(f"  Skipping case {case_id} - already processed")
                skipped_cases += 1
                continue
                
            # Verify DICOM path exists
            if not os.path.exists(cta_src_nifti_path):
                logger.warning(f"  SRC Nifti path not found for case {case_id}: {cta_src_nifti_path}")
                continue
            
            try:
                # Prepare parameters
                params = template_params.copy()
                params["case_dir"] = case_dir
                params["cta_nifti_path"] = cta_dst_nifti_path
                
                # Create output directory
                os.makedirs(case_dir, exist_ok=True)
                logger.debug(f"  Created/verified case directory: {case_dir}")
                
                # Save parameters
                with open(params_file, "w") as f:
                    json.dump(params, f, indent=4)
                logger.debug(f"  Saved parameters to: {params_file}")
                
                # Process the case
                logger.info(f"  Starting arterial processing for case {case_id}")
                main(params_file)
                logger.info(f"  Successfully completed processing case {case_id}")
                processed_cases += 1
                    
            except Exception as e:
                logger.error(f"  Failed to process case {case_id}: {e}")
                error_log[str(case_id)] = {
                    "error_1": str(e),
                    "traceback_1": traceback.format_exc().splitlines()
                }
                with open(error_log_file_path, 'w') as f:
                    json.dump(error_log, f, indent=4)
                failed_cases += 1
            
            logger.info(f"Finished processing case {case_id}")
            
        except Exception as e:
            logger.error(f"Failed to process case {case_id}: {e}")
            error_log[str(case_id)] = {
                "error_2": str(e),
                "traceback_2": traceback.format_exc().splitlines()
            }
            with open(error_log_file_path, 'w') as f:
                json.dump(error_log, f, indent=4)
            failed_cases += 1

except Exception as e:
    logger.error(f"Failed to list cases in source directory {SRC_DIR}: {e}")
    raise

# Log final summary
logger.info("="*60)
logger.info("BATCH PROCESSING SUMMARY")
logger.info(f"Total cases found: {total_cases}")
logger.info(f"Successfully processed: {processed_cases}")
logger.info(f"Skipped (already processed): {skipped_cases}")
logger.info(f"Failed: {failed_cases}")
logger.info("="*60)
logger.info("Batch arterial processing completed")