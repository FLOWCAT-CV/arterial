#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
This script shows how to use landmark detection and centerline extraction to extract centerlines between well-defined endpoints, 
within a vascular segmentation derived from a CTA. This script will run the following steps:

1. Segmentation
2. Landmark detection
3. Centerline extraction between detected landmarks/endpoints

To demonstrate the usage of the python API without the need to save intermediate files, we will
load data directly from the py objects, not from the files.

"""

from arterial.segmentation.segmenter import VesselSegmenter
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.landmark_detection.landmark_detector import LandmarkDetector

def run_segmentation(case_dir, mode="extracranial_vessels", cta_nifti_path=None, fast_segmentation=False, no_slicing=False):
    # Initialize the segmentation model
    segmentation_model = VesselSegmenter(
        case_dir=case_dir,
        mode=mode,
        cta_nifti_path=cta_nifti_path,
        fast_segmentation=fast_segmentation,
        no_slicing=no_slicing
    )
    # Segment the CTA
    segmentation_model.segment_vessels_from_cta(save=True)

    return segmentation_model.segmentation_nifti

def run_landmark_detection(case_dir, mode="extracranial_vessels", cta_nifti_path=None):
    # Initialize the landmark detector
    landmark_detector = LandmarkDetector(
        case_dir=case_dir,
        mode=mode,
        cta_nifti_path=cta_nifti_path
    )
    # Detect landmarks on the CTA
    landmark_detector.detect_landmarks_on_cta(return_mask=False, save=True)

    return landmark_detector.landmarks_ras_mm_dict

def run_centerline_extraction(case_dir, segmentation_nifti, landmarks_dict, mode="extracranial_vessels", fast_segmentation=False):
    # Initialize the centerline extractor
    centerline_extractor = CenterlineExtractor(
        case_dir=case_dir,
        mode=mode,
        segmentation_nifti_path=None,
        fast_segmentation=fast_segmentation
    )
    # Load segmentation nifti
    centerline_extractor._load_segmentation_nifti_from_nib(segmentation_nifti)
    # Extract centerlines between detected landmarks/endpoints
    landmark_pairs = {
        'l-ica': ('l-eica', 'l-tica'),
        'r-ica': ('r-eica', 'r-tica'),
        'l-mca': ('l-tica', 'l-mca'),
        'r-mca': ('r-tica', 'r-mca'),
        'l-ica_mca': ('l-eica', 'l-mca'),
        'r-ica_mca': ('r-eica', 'r-mca')
    }
    for landmark_pair_key in landmark_pairs.keys():
        centerline_extractor.extract_centerline_between_endpoints(landmarks_dict[landmark_pairs[landmark_pair_key][0]], landmarks_dict[landmark_pairs[landmark_pair_key][1]], landmark_pair_key, save=True)

def main(case_dir, mode, cta_nifti_path):
    segmentation_nifti = run_segmentation(case_dir, mode, cta_nifti_path)
    landmarks_dict = run_landmark_detection(case_dir, mode, cta_nifti_path)
    run_centerline_extraction(case_dir, segmentation_nifti, landmarks_dict, mode)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cd", "--case_dir", type=str, required=True,
        help="Path to directory containing the CTA nifti file. Assumes that the CTA nifti file has the basename of the dir.")
    parser.add_argument("-cnp", "--cta_nifti_path", type=str, default=None,
        help="Path to the CTA nifti file. Assumes case_dir/cta.nii.gz if not provided.")
    parser.add_argument("-m", "--mode", type=str, default="extracranial_vessels",
        help="Mode is defined coherently with the rest of Arterial, although CarotiCAT only supports extracranial_vessels (i.e. head and neck CTA) for now.")
    args = parser.parse_args()

    case_dir = args.case_dir 
    cta_nifti_path = args.cta_nifti_path 
    mode = args.mode 
    
    main(case_dir, mode, cta_nifti_path)

""" 
Usage: $ python landmark_detection_and_centerline_extraction.py -cd /path/to/case_dir -cnp /path/to/cta.nii.gz
"""