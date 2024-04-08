import os
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

case_dir = os.path.join(os.path.dirname(__file__), "test_data")
mode = "extracranial_vessels"
segmentation_nifti_path = os.path.join(case_dir, f"input_test_data/{mode}_segmentation.nii.gz")
fast_segmentation = True
centerline_extractor = CenterlineExtractor(case_dir, mode, segmentation_nifti_path, fast_segmentation)

centerline_extractor.perform_centerline_extraction()