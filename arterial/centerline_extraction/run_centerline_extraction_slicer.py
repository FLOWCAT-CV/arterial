#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

def perform_preprocessing_and_centerline_extraction(case_dir, mode = "vessels", no_display = False, fast_segmentation = False):
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

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    mode: string, default = "vessels"
        Determines whether the centerline is extracted from ```vessels```, ```intracranial_vessels``` 
        or ```thrombus```. 
    no_display : bool, default = False
        Boolean variable to be used when running analysis on a headless server.
        In addition, add ```$xvfb-run --auto-servernum --server-num=1``` at the beggining
        of the command line call when executing the script from the command line.
        E.g.: ```$xvfb-run --auto-servernum --server-num=1 python perform_analysis.py -case_dir {case_dir} -no_display {True}```
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
    if mode == "vessels":
        # Perform vessel segmentation and centerline extraction. This generates segmentations and centerlines in case_dir/centerlines and case_dir/segmentations
        if no_display: # Use if remote server is used, in combination with xvfb-run --auto-servernum --server-num=1
            if fast_segmentation:
                os.system("{} --disable-terminal-outputs --python-script {} -case_dir {} -fast t -m vessels --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
            else:
                os.system("{} --disable-terminal-outputs --python-script {} -case_dir {} -m vessels --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
        else:
            if fast_segmentation:
                os.system("{} --no-main-window --no-splash --python-script {} -case_dir {} -m vessels -fast t --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
            else:
                os.system("{} --no-main-window --no-splash --python-script {} -case_dir {} -m vessels --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
    if mode == "intracranial_vessels":
        # Perform intracranial vessel segmentation and centerline extraction. This generates intracranial_segmentations and intracranial_centerlines in case_dir/centerlines and case_dir/segmentations
        if no_display: # Use if remote server is used, in combination with xvfb-run --auto-servernum --server-num=1
            os.system("{} --disable-terminal-outputs --python-script {} -case_dir {} -m intracranial_vessels --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
        else:
            os.system("{} --no-main-window --no-splash --python-script {} -case_dir {} -m intracranial_vessels --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
    elif mode == "thrombus":
        # Perform thrombus segmentation and centerline extraction. This generates thrombus_segmentations and thrombus_centerlines in case_dir/centerlines and case_dir/segmentations
        if no_display: # Use if remote server is used, in combination with xvfb-run --auto-servernum --server-num=1
            os.system("{} --disable-terminal-outputs --python-script {} -case_dir {} -m thrombus --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
        else:
            os.system("{} --no-main-window --no-splash --python-script {} -case_dir {} -m thrombus --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))

if __name__ == "__main__":
    # Script to be executed by PythonSlicer interpreter
    # Will only be executed when this script is called from a direct terminal command
    import slicer

    import argparse

    from preprocessing.preprocessing import preprocessing_vessels, preprocessing_intracranial_vessels, preprocessing_thrombus
    from centerline_extraction import centerline_extraction, intracranial_centerline_extraction, thrombus_centerline_extraction

    parser = argparse.ArgumentParser()

    parser.add_argument('-case_dir', '--case_dir', type=str, required=True, 
        help='path binary nifti to be processed. Required.')
    parser.add_argument('-mode', '--mode', type=str, required=False, default='vessels',
        help='Determines whether the centerline is extracted from vessels or thrombus. Not required.')
    parser.add_argument("-fast", "--fast_segmentation", type=str, default=False, required=False,
        help='flag to indicate if segmentation was acquired in fast or full mode. It will change '
             'the preprocessing of the centerline extraction process. Defaults to False. Not required.')

    args = parser.parse_args()

    case_dir = args.case_dir
    mode = args.mode
    fast_segmentation = args.fast_segmentation
    
    if mode == "vessels":
        # Load volume and associate to node
        slicer.util.loadLabelVolume(os.path.join(case_dir, "{}_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))
        master_volume_node = getNode("{}_vessel_segmentation".format(os.path.basename(case_dir)))

        # Perform segmentation from binary mask
        segmentation_node, masked_volume_array = preprocessing_vessels(case_dir, master_volume_node, fast_segmentation)
        # Perform centerline extraction. Creates centerlines.vtk
        centerline_extraction(case_dir, segmentation_node, masked_volume_array)
    elif mode == "intracranial_vessels":
        # Load volume and associate to node
        slicer.util.loadLabelVolume(os.path.join(case_dir, "{}_intracranial_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))
        master_volume_node = getNode("{}_intracranial_vessel_segmentation".format(os.path.basename(case_dir)))

        # Perform segmentation from binary mask
        segmentation_node, masked_volume_array = preprocessing_intracranial_vessels(case_dir, master_volume_node)
        # Perform centerline extraction. Creates centerlines.vtk
        intracranial_centerline_extraction(case_dir, segmentation_node, masked_volume_array)
    elif mode == "thrombus":
        # Load volume and associate to node
        slicer.util.loadLabelVolume(os.path.join(case_dir, "{}_thrombus_segmentation.nii.gz".format(os.path.basename(case_dir))))
        master_volume_node = getNode("{}_thrombus_segmentation".format(os.path.basename(case_dir)))

        # Perform segmentation from binary mask
        segmentation_node, masked_volume_array = preprocessing_thrombus(case_dir, master_volume_node)
        # Perform centerline extraction. Creates centerlines.vtk
        thrombus_centerline_extraction(case_dir, segmentation_node)