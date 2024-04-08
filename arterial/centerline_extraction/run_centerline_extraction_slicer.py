#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

def perform_preprocessing_and_centerline_extraction(case_dir, segmentation_nifti_path, mode = "extracranial_vessels", fast_segmentation = False):
    """
    Wrapper function to make terminal command calls for centerline preprocessing and extraction,
    using Slicer [1] and VMTK [2-4]. Since PythonSlicer is needed, we need to call terminal commands to execute
    the corresponding script. When this script is executed as __main__ (upon terminal command call),
    it performs preprocessing and terminal extraction. This function basically calls itself from the 
    terminal line using Slicer's Python interpreter.

    Assumes that enviroment variables slicer_path and arterial_dir are correctly set according to your
    own system. This should be set in the ~/.bashrc (or ~/.zshrc) and should look something like:

    >>> export slicer_path="/path/to/Slicer"
    >>> arterial_dir="/path/to/arterial/arterial"

    References:
    [1]     Fedorov, Andriy, Reinhard Beichel, Jayashree Kalpathy-Cramer, Julien Finet, Jean-Christophe Fillion-Robin, 
    Sonia Pujol, Christian Bauer, et al. 2012. "3D Slicer as an Image Computing Platform for the Quantitative Imaging 
    Network." Magnetic Resonance Imaging 30 (9): 1323-41. https://doi.org/10.1016/j.mri.2012.05.001.
    [2]     Antiga, Luca, Bogdan Ene-Iordache, and Andrea Remuzzi. 2003. "Centerline Computation and Geometric Analysis 
    of Branching Tubular Surfaces with Application to Blood Vessel Modeling." Wscg. 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.671&rep=rep1&type=pdf.
    [3]     Antiga, Luca, Marina Piccinelli, Lorenzo Botti, Bogdan Ene-Iordache, Andrea Remuzzi, and David A. Steinman. 
    2008. "An Image-Based Modeling Framework for Patient-Specific Computational Hemodynamics." Medical and Biological 
    Engineering and Computing 46 (11): 1097-1112. https://doi.org/10.1007/s11517-008-0420-1.
    [4]     Antiga, Luca, and David A. Steinman. 2004. "Robust and Objective Decomposition and Mapping of Bifurcating 
    Vessels." IEEE Transactions on Medical Imaging 23 (6): 704-13. https://doi.org/10.1109/TMI.2004.826946.

    Note: most of the times we will be running this processing on a headless server. In order to do that in a linux server,
    we have to install a dummy X server and we have to call the python scripts preceeded by:

    >>> $xvfb-run --auto-servernum --server-num=1

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    segmentation_nifti_path : string or path-like object
        Path to the binary nifti to be processed.
    mode: string, default = "extracranial_vessels"
        Determines whether the centerline is extracted from `extracranial_vessels`, `intracranial_vessels`
        or `thrombus`. 
    fast_segmentation : bool, default = False
            Boolean variable to be used when running analysis derived from fast segmentation (lowres).
            In this case, segmentation of the cerebral arteries is less reliable, so a higher fraction
            of vessels is ignored.

    Return
    ------
    
    """
    # Define paths from enviornment variables. These should be set prior to execution in ~/.bashrc or equivalent
    SLICER_PATH = os.environ["slicer_path"]
    RUN_CENTERLINE_EXTRACTION_SCRIPT = os.path.join(os.environ["arterial_dir"], "centerline_extraction/run_centerline_extraction_slicer.py")

    # Perform vessel segmentation and centerline extraction. This generates segmentations and centerlines in case_dir/centerlines and case_dir/segmentations
    if fast_segmentation:
        os.system("{} --python-script {} --disable-terminal-outputs -case_dir {} -segmentation_nifti_path {} -m {} -fast t --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir, segmentation_nifti_path, mode))
    else:
        os.system("{} --python-script {} --disable-terminal-outputs -case_dir {} -segmentation_nifti_path {} -m {} --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir, segmentation_nifti_path, mode))

if __name__ == "__main__":
    # Script to be executed by PythonSlicer interpreter
    import slicer

    import argparse

    from preprocessing.preprocessing import preprocessing_extracranial_vessels, preprocessing_intracranial_vessels
    from centerline_extraction import centerline_extraction

    import nibabel as nib

    parser = argparse.ArgumentParser()

    parser.add_argument('-case_dir', '--case_dir', type=str, required=True, 
        help='path binary nifti to be processed. Required.')
    parser.add_argument('-segmentation_nifti_path', '--segmentation_nifti_path', type=str, required=True,
        help='path to the binary nifti to be processed. Required.')
    parser.add_argument('-mode', '--mode', type=str, required=False, default='extracranial_vessels',
        help='Determines whether the centerline is extracted from extracranial vessels, intracranial vessels or thrombus. Not required.')
    parser.add_argument("-fast", "--fast_segmentation", type=str, default=False, required=False,
        help='flag to indicate if segmentation was acquired in fast or full mode. It will change '
             'the preprocessing of the centerline extraction process. Defaults to False. Not required.')

    args = parser.parse_args()

    case_dir = args.case_dir
    segmentation_nifti_path = args.segmentation_nifti_path
    mode = args.mode
    fast_segmentation = args.fast_segmentation

    segmentation_nifti = nib.load(segmentation_nifti_path)
    
    # Load volume
    slicer.util.loadLabelVolume(segmentation_nifti_path)
    # Associate to volume node
    master_volume_node = getNode(os.path.basename(segmentation_nifti_path)[:-7])

    if mode == "extracranial_vessels":
        # Perform segmentation from binary mask
        segmentation_node, masked_volume_array = preprocessing_extracranial_vessels(case_dir, segmentation_nifti, master_volume_node, fast_segmentation)
        # Perform centerline extraction
    elif mode == "intracranial_vessels":
        # Perform segmentation from binary mask
        segmentation_node, masked_volume_array = preprocessing_intracranial_vessels(case_dir, segmentation_nifti, master_volume_node)
        # Perform centerline extraction
    centerline_extraction(case_dir, segmentation_nifti, mode, segmentation_node, masked_volume_array)