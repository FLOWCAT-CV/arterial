#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Batch processing of landmark detection. This script shows the most common use case for the LandmarkDetector class.
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

def run_landmark_detection_single_case(case_dir, mode="extracranial_vessels", cta_nifti_path=None):
    # Initialize the landmark detector
    landmark_detector = LandmarkDetector(case_dir, mode, cta_nifti_path)
    # Detect landmarks on the CTA
    landmark_detector.detect_landmarks_on_cta(return_mask=True, save=True)

    return landmark_detector.landmarks_ras_mm_dict, landmark_detector.landmarks_slicer_json_path, landmark_detector.predicted_mask_nifti_path

def run_landmark_detection_batch_processing(src_dir, mode, cta_nifti_path_convention=None):
    """
    Run landmark detection for all cases in the source directory.

    Parameters
    ----------
    src_dir: str
        Path to the source directory.
    mode: str
        Mode of the landmark detector.
    cta_nifti_path_convention: str, optional
        Convention to use for the CTA nifti path.

    Returns
    -------

    """
    landmarks_ras_mm_dict = {}
    landmarks_slicer_json_path_dict = {}
    predicted_mask_nifti_path_dict = {}

    for case_dir in os.listdir(src_dir):
        if cta_nifti_path_convention is not None:
            cta_nifti_path = os.path.join(case_dir, cta_nifti_path_convention)
        else:
            cta_nifti_path = None # Will default to case_dir/cta.nii.gz
        # Run landmark detection for the current case
        landmarks_ras_mm_dict_, landmarks_slicer_json_path_, predicted_mask_nifti_path_ = run_landmark_detection_single_case(case_dir, mode, cta_nifti_path)

        # Output objects are stored as attributes of the landmark_detector object
        landmarks_ras_mm_dict[case_dir] = landmarks_ras_mm_dict_
        landmarks_slicer_json_path_dict[case_dir] = landmarks_slicer_json_path_
        predicted_mask_nifti_path_dict[case_dir] = predicted_mask_nifti_path_

    return landmarks_ras_mm_dict, landmarks_slicer_json_path_dict, predicted_mask_nifti_path_dict


if __name__ == "__main__":
    input_dir = "/path/to/input_dir"
    mode = "extracranial_vessels"
    cta_nifti_path_convention = "cta.nii.gz"
    
    run_landmark_detection_batch_processing(input_dir, mode, cta_nifti_path_convention)