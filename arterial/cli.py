#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Command-line entry point of the Arterial pipeline.

Installed as the ``arterial`` command (see setup.py); ``python -m arterial.cli``
works as well.
"""

import os
import argparse

from arterial.run.processor import ArterialProcessor, SUPPORTED_MODES, save_run_record, process_cases, get_missing_outputs


def build_parser():
    """
    Builds the command-line parser of the Arterial pipeline.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser whose namespace is accepted by ArterialProcessor.

    """
    parser = argparse.ArgumentParser(prog="arterial", description="Arterial: automated vascular analysis of a head-and-neck CTA.")
    parser.add_argument("-cd", "--case_dir", type=str, nargs="+", required=True,
        help="Path to directory containing the nifti image (assumes that the nifti file has the basename of the dir). Required. "
        "Several directories can be given to process several cases; cases already completed with the same settings "
        "({mode}/arterial_run.json) are skipped.")
    parser.add_argument("-cnp", "--cta_nifti_path", type=str, required=False, default=None,
        help="Path to the nifti image. Not required (assumes is case_dir/cta.nii.gz).")
    parser.add_argument('-m', '--mode', type=str, required=False, default='extracranial_vessels', choices=SUPPORTED_MODES,
        help='Determines whether the analysis is performed for extracranial_vessels or intracranial_vessels. Not required.')
    parser.add_argument("-fast", "--fast_segmentation", action="store_true",
        help="Boolean argument to determine if fast segmentation is used or not. Use (True) if fast segmentation is wanted. "
        "Otherwise, full segmentation will be performed. Not required, default = False.")
    parser.add_argument('-sd', '--sampling_distance_mm', type=float, required=False, default=2,
        help='Sampling density (in mm) for the dense centerline graph sampling from the centerline models. Not required. Default = 2.')
    parser.add_argument("-ss", "--skip_segmentation", action="store_true",
        help="Boolean argument to determine if segmentation is predicted or not. Use (True) if segmentation is already "
        "predicted, to skip nnUNet inference and save time. If True, there should exist a nifti file with the binary map "
        "at case_dir/{mode}/segmentation.nii.gz. Not required, default = False.")
    parser.add_argument("-sce", "--skip_centerline_extraction", action="store_true",
        help="Boolean argument to determine if centerline extraction should be skipped or not. Not required, default = False.")
    parser.add_argument("-sb", "--skip_branching", action="store_true",
        help="Boolean argument to determine if centerline branching should be performed or not. Not required, default = False.")
    parser.add_argument("-sc", "--skip_clipping", action="store_true",
        help="Accepted for backwards compatibility; surface model clipping is not part of the pipeline and this flag has no effect.")
    parser.add_argument("-svl", "--skip_vessel_labelling", action="store_true",
        help="Boolean argument to determine if vessel labelling should be skipped or not. Not required, default = False.")
    parser.add_argument("-sfe", "--skip_feature_extraction", action="store_true",
        help="Boolean argument to determine if feature extraction should be skipped or not. Not required, default = False.")
    parser.add_argument("-sap", "--skip_access_prediction", action="store_true",
        help="Boolean argument to determine if access prediction should be skipped or not. Not required, default = False.")
    parser.add_argument("-sld", "--skip_landmark_detection", action="store_true",
        help="Boolean argument to determine if landmark detection should be skipped or not. Not required, default = False.")
    parser.add_argument("-clnn", "--cl_dice_nnunet", action="store_true",
        help="Boolean argument to determine if nnunet trained with centerline dice should be used or not. Not required, default = False.")
    parser.add_argument("-ns", "--no_slicing", action="store_true",
        help="Boolean argument to determine if the image should be sliced or not. Meant to be used in case of intracranial_vessels segmentation for head CTA. "
        "Not required, default = False.")
    parser.add_argument("-s99", "--set_threshold_099", action="store_true",
        help="Boolean argument to determine if the segmentation logit threshold should be set to 0.99 or not. Not required, default = False.")
    parser.add_argument("-rs", "--resampling", type=str, required=False, default="torch", choices=["scipy", "torch"],
        help="Resampling used by nnunet before and after the network: scipy (CPU) or torch (GPU, falls back to scipy without a GPU). "
        "Not required, default = torch.")
    parser.add_argument("-nt", "--num_threads", type=int, required=False, default=None,
        help="Number of CPU threads (divided between workers). Not required, default = ARTERIAL_NUM_THREADS if set, "
        "otherwise the usable cores capped at 8.")
    parser.add_argument("-w", "--workers", type=int, required=False, default=1,
        help="Number of cases processed in parallel on the same GPU when several case directories are given. "
        "Reduced if there is not enough free GPU memory or RAM. Not required, default = 1.")
    return parser


def main(argv=None):
    """
    Runs the pipeline from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse instead of sys.argv[1:].

    Returns
    -------
    times : dict
        Wall-clock time of every stage, as returned by ArterialProcessor.perform_analysis. With several
        case directories, the times (or error message) of every case processed, by case directory.

    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(os.path.realpath(case_dir) for case_dir in args.case_dir)) > 1:
        if args.cta_nifti_path is not None:
            parser.error("-cnp/--cta_nifti_path can only be used with a single case directory")
        return process_cases(args)

    args.case_dir = args.case_dir[0]
    processor = ArterialProcessor(args)
    times = processor.perform_analysis()
    missing_outputs = get_missing_outputs(processor)
    if missing_outputs:
        print(f"Warning: the analysis is incomplete, missing {', '.join(missing_outputs)}")
    else:
        save_run_record(args, processor, times)
    return times

if __name__ == "__main__":
    main()