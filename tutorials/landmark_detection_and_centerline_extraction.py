#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
This script shows how to use landmark detection and centerline extraction to extract centerlines between well-defined endpoints, 
within a vascular segmentation derived from a CTA. This script will run the following steps:

1. Segmentation
2. Landmark detection
3. Centerline extraction between detected landmarks/endpoints

To demonstrate the usage of the python API without the need to save intermediate files, we will
load data directly from the py objects, not from the files.

"""

import os

from arterial.segmentation.segmenter import VesselSegmenter
from arterial.landmark_detection.landmark_detector import LandmarkDetector
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.feature_extraction.feature_extractor import FeatureExtractor
from arterial.io.load_and_save_operations import load_vtkpolydata

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

def run_landmark_detection(case_dir, mode="extracranial_vessels", cta_nifti_path=None, use_segmentation_model=True):
    # Initialize the landmark detector
    landmark_detector = LandmarkDetector(
        case_dir=case_dir,
        mode=mode,
        cta_nifti_path=cta_nifti_path
    )
    # Detect landmarks on the CTA
    landmark_detector.detect_landmarks_on_cta(return_mask=False, save=True, use_segmentation_model=use_segmentation_model)

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

    return list(landmark_pairs.keys())

def run_feature_extraction(case_dir, cta_nifti_path, centerline_ids, mode="extracranial_vessels", sampling_distance_mm=0.5):
    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(case_dir, mode, cta_nifti_path=cta_nifti_path, sampling_distance_mm=sampling_distance_mm)
    for centerline_id in centerline_ids:
        if os.path.isfile(os.path.join(case_dir, f"{mode}/individual_centerlines", f"individual_centerline_{centerline_id}.vtk")):
            print(f"Building and featurizing {centerline_id}")
            centerline_model = load_vtkpolydata(os.path.join(case_dir, f"{mode}/individual_centerlines", f"individual_centerline_{centerline_id}.vtk"))
            feature_extractor.build_and_featurize_individual_centerline_graph(centerline_model, centerline_id=centerline_id, save=True)

def main(case_dir, mode, cta_nifti_path):
    segmentation_nifti = run_segmentation(case_dir, mode, cta_nifti_path)
    landmarks_dict = run_landmark_detection(case_dir, mode, cta_nifti_path)
    centerline_ids = run_centerline_extraction(case_dir, segmentation_nifti, landmarks_dict, mode)
    run_feature_extraction(case_dir, cta_nifti_path, centerline_ids, mode)

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