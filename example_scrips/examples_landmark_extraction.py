"""
Simple usage examples for CarotiCAT landmark extraction.

This file shows the most common use cases for the LandmarkExtractor class.

TODO: needs to be updated to use the new LandmarkExtractor class.
"""

from arterial.landmark_extraction.landmark_extractor import LandmarkExtractor

def run_landmark_extraction(case_dir, mode, cta_nifti_path):
    # Initialize the landmark extractor
    landmark_extractor = LandmarkExtractor(case_dir, mode, cta_nifti_path)
    # Detect landmarks on the CTA
    landmark_extractor.detect_landmarks_on_cta(return_mask=True, save=True)

    return landmark_extractor.landmarks_ras_mm_dict, landmark_extractor.landmarks_slicer_json_path, landmark_extractor.predicted_mask_nifti_path


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
    
    landmarks_ras_mm_dict, landmarks_slicer_json_path, predicted_mask_nifti_path = run_landmark_extraction(case_dir, mode, cta_nifti_path)

    print("Landmarks:", landmarks_ras_mm_dict)
    print("Landmarks in Slicer format saved in:", landmarks_slicer_json_path)
    print("Predicted mask saved in:", predicted_mask_nifti_path)