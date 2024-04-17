#   Copyright 2022 Stroke Research at Vall d"Hebron Research Institute (VHIR), Barcelona, Spain.
 
import argparse

from arterial.run.processor import ArterialProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cd", "--case_dir", type=str, required=True,
        help="Path to directory containing the nifti image (assumes that the nifti file has the basename of the dir). Required.")
    parser.add_argument("-cnp", "--cta_nifti_path", type=str, required=False, default=None,
        help="Path to the nifti image. Not required (assumes is case_dir/cta.nii.gz).")
    # parser.add_argument("-no_display", "--no_display", type=bool, required=False, default=False, 
    #     help="If using a remote Linux, this should be used following correct Slicer installation, and should be coulpled "
    #     "with the use of `xvfb-run --auto-servernum --server-num=1` upon use before calling this script (prior to the python command). "
    #     "Not required, default = False.")
    parser.add_argument('-m', '--mode', type=str, required=False, default='extracranial_vessels',
        help='Determines whether the analysis is performed for extracranial_vessels, intracranial_vessels or thrombus. Not required.')
    parser.add_argument("-fast", "--fast_segmentation", action="store_true",
        help="Boolean argument to determine if fast segmentation is used or not. Use (True) if fast segmentation is wanted. "
        "Otherwise, full segmentation will be performed. Not required, default = False.")
    parser.add_argument("-ss", "--skip_segmentation", action="store_true",
        help="Boolean argument to determine if segmentation is predicted or not. Use (True) if segmentation is already "
        "predicted, to skip nnUNet inference and save time. If True, there should exist a nifti file with the binary map "
        "with the following naming convention: {mode}_segmentation.nii.gz. Not required, default = False.")
    parser.add_argument("-sce", "--skip_centerline_extraction", action="store_true",
        help="Boolean argument to determine if centerline extraction should be skipped or not. Not required, default = False.")
    parser.add_argument("-sb", "--skip_branching", action="store_true",
        help="Boolean argument to determine if centerline branching should be performed or not. Not required, default = False.")
    parser.add_argument("-sc", "--skip_clipping", action="store_true",
        help="Boolean argument to determine if model clipping should be performed or not. Not required, default = False.")
    parser.add_argument("-svl", "--skip_vessel_labelling", action="store_true",
        help="Boolean argument to determine if vessel labelling should be skipped or not. Not required, default = False.")
    parser.add_argument("-sfe", "--skip_feature_extraction", action="store_true",
        help="Boolean argument to determine if feature extraction should be skipped or not. Not required, default = False.")
    
    parser = parser.parse_args()

    processor = ArterialProcessor(parser)
    processor.perform_analysis()

if __name__ == "__main__":
    main()