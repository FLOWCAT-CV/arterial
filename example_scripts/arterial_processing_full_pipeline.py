from arterial.run.processor import ArterialProcessor
from arterial.io.dicom_and_nifti import convert_dicom_to_nifti_d2n
import os, shutil
import json
import pydicom
import argparse
import logging
import sys
from datetime import datetime

def load_args_from_params_json(params):
    # Load params into args
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", type=str, default=params.get('case_dir', None))
    parser.add_argument("--cta_nifti_path", type=str, default=params.get('cta_nifti_path', os.path.join(params.get('case_dir', None), "cta.nii.gz")))
    parser.add_argument("--mode", type=str, default=params.get('mode', "extracranial_vessels"))
    parser.add_argument("--sampling_distance_mm", type=float, default=params.get('sampling_distance_mm', 2.0))
    parser.add_argument("--fast_segmentation", type=bool, default=params.get('fast_segmentation', False))
    parser.add_argument("--skip_segmentation", type=bool, default=params.get('skip_segmentation', False))
    parser.add_argument("--skip_centerline_extraction", type=bool, default=params.get('skip_centerline_extraction', False))
    parser.add_argument("--skip_branching", type=bool, default=params.get('skip_branch_model_extraction', False))
    parser.add_argument("--skip_clipping", type=bool, default=params.get('skip_centerline_postprocessing', True))
    parser.add_argument("--skip_vessel_labelling", type=bool, default=params.get('skip_vessel_labelling', False))
    parser.add_argument("--skip_feature_extraction", type=bool, default=params.get('skip_feature_extraction', False))
    parser.add_argument("--skip_access_prediction", type=bool, default=params.get('skip_access_prediction', False))
    parser.add_argument("--skip_landmark_detection", type=bool, default=params.get('skip_landmark_detection', False))
    parser.add_argument("--cl_dice_nnunet", type=bool, default=params.get('cl_dice_nnunet', False))  
    parser.add_argument("--no_slicing", type=bool, default=params.get('no_slicing', False))
    parser.add_argument("--set_threshold_099", type=bool, default=params.get('set_threshold_099', False))
    args = parser.parse_args()
    return args

def print_dcm_info(cta_dicom_path):
    dcm = pydicom.dcmread(os.path.join(cta_dicom_path, sorted(os.listdir(cta_dicom_path))[0]))
    print("Study description: ", getattr(dcm, 'StudyDescription', None))
    print("Series description: ", getattr(dcm, 'SeriesDescription', None))
    print("Patient ID: ", getattr(dcm, 'PatientID', None))
    print("Slice thickness: ", getattr(dcm, 'SliceThickness', None))
    print("Number of slices: ", len(os.listdir(cta_dicom_path)))
    print("Pixel spacing: ", getattr(dcm, 'PixelSpacing', None))
    print("Manufacturer: ", getattr(dcm, 'Manufacturer', None))
    print("Body part examined: ", getattr(dcm, 'BodyPartExamined', None))

    return dcm

def main(parameters_file_path):
    with open(parameters_file_path, 'r') as f:
        params = json.load(f)

    case_dir = params.get('case_dir', None)
    assert case_dir is not None and case_dir != "", "case_dir is required"
    cta_nifti_path = params.get('cta_nifti_path', os.path.join(case_dir, "cta.nii.gz"))
    assert cta_nifti_path is not None and cta_nifti_path != "", "cta_nifti_path is required"

    os.makedirs(case_dir, exist_ok=True)
    
    # Setup case-specific logging
    log_filename = f"arterial_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(case_dir, log_filename)
    
    logger = logging.getLogger(case_dir)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to avoid duplication
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Redirect stdout and stderr to logger
    class StreamToLogger:
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
        
        def isatty(self):
            return False
    
    # Save original stdout/stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    
    try:
        # Copy params file to case_dir. Overwrite if it already exists.
        if not os.path.exists(os.path.join(case_dir, "arterial_processing_params.json")):
            shutil.copy(parameters_file_path, os.path.join(case_dir, "arterial_processing_params.json"))

        # dcm = print_dcm_info(cta_dicom_path)

        # if dcm.SliceThickness > 1.5:
        #     print("Skipping case because slice thickness is greater than 1.5 mm")
        #     return
        
        # if not os.path.exists(cta_nifti_path):
        #     convert_dicom_to_nifti_d2n(cta_dicom_path, cta_nifti_path)

        # if not os.path.exists(cta_nifti_path):
        #     print("Skipping case because cta_nifti_path does not exist")
        #     shutil.rmtree(case_dir)
        #     return

        # Load params into args
        args = load_args_from_params_json(params)

        processor = ArterialProcessor(args)
        processor.perform_analysis()
    
    except Exception as e:
        logger.error(f"Error processing case: {e}", exc_info=True)
        raise
    
    finally:
        # Restore original stdout and stderr
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
        # Close logging handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
    parameters_file_path = os.path.join(os.environ["arterial_dir"], "../example_scripts", "arterial_processing_params.json")
    main(parameters_file_path)