#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
Single case landmark detection. This script shows the most common use case for the LandmarkDetector class.
The LandmarkDetector class will run a segmentation model of the following structures:

>>> l-tica: left internal carotid artery bifurcation
>>> r-tica: right internal carotid artery bifurcation
>>> l-eica: left external carotid artery bifurcation
>>> r-eica: right external carotid artery bifurcation
>>> r-mca: right middle cerebral artery M1 bifurcation
>>> l-mca: left middle cerebral artery M1 bifurcation

If save is enabled, the following files will be saved:
>>> case_dir/extracranial_vessels/landmarks.json : landmarks in RAS coordinates, saved as a dictionary {label: [x, y, z]}
>>> case_dir/extracranial_vessels/landmarks_slicer.json : landmarks in json format for Slicer to read
>>> case_dir/extracranial_vessels/predicted_mask.nii.gz : predicted mask of the landmarks, with each structure being encoded by a different value

"""

from arterial.landmark_detection.landmark_detector import LandmarkDetector

def run_landmark_detection(case_dir, mode="extracranial_vessels", cta_nifti_path=None):
    # Initialize the landmark detector
    landmark_detector = LandmarkDetector(case_dir, mode, cta_nifti_path)
    # Detect landmarks on the CTA
    landmark_detector.detect_landmarks_on_cta(return_mask=True, save=True)

    return landmark_detector.landmarks_ras_mm_dict, landmark_detector.landmarks_slicer_json_path, landmark_detector.predicted_mask_nifti_path


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
    
    landmarks_ras_mm_dict, landmarks_slicer_json_path, predicted_mask_nifti_path = run_landmark_detection(case_dir, mode, cta_nifti_path)

    print("Landmarks:", landmarks_ras_mm_dict)
    print("Landmarks in Slicer format saved in:", landmarks_slicer_json_path)
    print("Predicted mask saved in:", predicted_mask_nifti_path)