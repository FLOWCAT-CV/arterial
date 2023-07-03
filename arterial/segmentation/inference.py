#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import shutil

from arterial.segmentation.utils import slice_cta_head_and_neck, join_head_and_neck_segmentations

def perform_inference_fast(case_dir):
    """
    Performs fast inference of CTA in case_dir (nifti file) with a trained nnunet [1]
    model to output a binary mask in a nifti format. The resulting nifti will be named 
    after the CTA with the same identifier adding a "_segmentation" at the end, and it
    will be stored in the case_dir.

    It is considered a fast approach since it applies a single segmentation algorithm
    for the full image.

    Assumes that it contains the image that we want to segment in nifti format, and 
    shares name with containing directory (os.path.basename(case_dir)).

    It also assumes that nnunet paths are set correctly, according to the guidelines 
    provided in the nnunet documentation (see documentation in
    <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
    The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
    set in the ~/.bashrc (or ~/.zshrc) file as:

    >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

    Saves binary map as:

    >>> case_dir/{os.path.basename(case_dir)}_segmentation.nii.gz

    References:
    [1]     Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method "
    "for deep learning-based biomedical image segmentation. Nat Methods (2020). "
    "https://doi.org/10.1038/s41592-020-01008-z"

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    # Define path to CTA
    case_path = os.path.join(case_dir, "{}.nii.gz".format(os.path.basename(case_dir)))

    # Create input directory and copy input CTA to it
    input_dir = os.path.join(case_dir, "input")
    if not os.path.isdir(input_dir): os.mkdir(input_dir)
    # Create output directory
    output_dir = os.path.join(case_dir, "output")
    if not os.path.isdir(output_dir): os.mkdir(output_dir)

    # Copy original CTA to input directory with nnunet format (_0000.nii.gz)
    shutil.copyfile(case_path, os.path.join(input_dir, "{}_0000.nii.gz".format(os.path.basename(case_dir))))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    input_dir = input_dir.replace("\ ", "")
    input_dir = input_dir.replace(" ", "\ ")
    output_dir = output_dir.replace("\ ", "")
    output_dir = output_dir.replace(" ", "\ ")

    # Perform inference using nnUNet_predict command in the command line
    os.system("nnUNet_predict -i " + input_dir + " -o " + output_dir + " -t Task001_Arterial -m 3d_lowres -f all")

    # Set path back to normal
    input_dir = input_dir.replace("\\", "")
    output_dir = output_dir.replace("\\", "")

    # Copy segmentation nifti to case_dir
    shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_segmentation.nii.gz".format(os.path.basename(case_dir))))

    # Remove input and output directories
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)

def perform_inference_full(case_dir):
    """
    Performs full inference of CTA in case_dir (nifti file) with a trained nnunet [1]
    model to output a binary mask in a nifti format. The resulting nifti will be named 
    after the CTA with the same identifier adding a "_segmentation" at the end, and it
    will be stored in the case_dir.

    It is considered a full approach since it applies two segmentation algorithms
    to obtain enhanced quality for cerebral arteries. To achieve that, first it performs
    a slicing of the original head-and-neck CTA into two separate nifti files (head and 
    neck), then it performs separate inference on both using trained models specifically
    for each of the two regions, and joins them into one.

    Assumes that it contains the image that we want to segment in nifti format, and 
    shares name with containing directory (os.path.basename(case_dir)).

    It also assumes that nnunet paths are set correctly, according to the guidelines 
    provided in the nnunet documentation (see documentation in
    <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
    The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
    set in the ~/.bashrc (or ~/.zshrc) file as:

    >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

    Saves binary map as:

    >>> case_dir/{os.path.basename(case_dir)}_segmentation.nii.gz

    References:
    [1]     Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method "
    "for deep learning-based biomedical image segmentation. Nat Methods (2020). "
    "https://doi.org/10.1038/s41592-020-01008-z"

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    def perform_inference_head(case_dir):
        """
        Performs inference of the head part of the CTA.

        It uses a network trained for inference in a full resolution version
        of the image, trained on a dataset including only cerebral arteries.

        It also assumes that nnunet paths are set correctly, according to the guidelines 
        provided in the nnunet documentation (see documentation in
        <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
        The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
        set in the ~/.bashrc (or ~/.zshrc) file as:

        >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

        Saves binary map as:

        >>> case_dir/{os.path.basename(case_dir)}_head_segmentation.nii.gz

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        # Define path to CTA
        case_path = os.path.join(case_dir, "{}_head.nii.gz".format(os.path.basename(case_dir)))

        # Create input directory and copy input CTA to it
        input_dir = os.path.join(case_dir, "input")
        if not os.path.isdir(input_dir): os.mkdir(input_dir)
        # Create output directory
        output_dir = os.path.join(case_dir, "output")
        if not os.path.isdir(output_dir): os.mkdir(output_dir)

        # Copy original CTA to input directory with nnunet format (_0000.nii.gz)
        shutil.copyfile(case_path, os.path.join(input_dir, "{}_0000.nii.gz".format(os.path.basename(case_dir))))

        # We have to get rid of potential spaces in paths for the terminal commands to work 
        input_dir = input_dir.replace("\ ", "")
        input_dir = input_dir.replace(" ", "\ ")
        output_dir = output_dir.replace("\ ", "")
        output_dir = output_dir.replace(" ", "\ ")

        # Perform inference using nnUNet_predict command in the command line
        os.system("nnUNet_predict -i " + input_dir + " -o " + output_dir + " -t Task002_Cerebral -m 3d_fullres -f all")

        # Set path back to normal
        input_dir = input_dir.replace("\\", "")
        output_dir = output_dir.replace("\\", "")

        # Copy segmentation nifti to case_dir
        shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_head_segmentation.nii.gz".format(os.path.basename(case_dir))))

        # Remove input and output directories
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)

    def perform_inference_neck(case_dir):
        """
        Performs inference of the neck part of the CTA.

        It uses the same network as for the full CTA processing,
        but is faster since only looks at part of the image.

        It also assumes that nnunet paths are set correctly, according to the guidelines 
        provided in the nnunet documentation (see documentation in
        <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
        The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
        set in the ~/.bashrc (or ~/.zshrc) file as:

        >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

        Saves binary map as:

        >>> case_dir/{os.path.basename(case_dir)}_neck_segmentation.nii.gz

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        # Define path to CTA
        case_path = os.path.join(case_dir, "{}_neck.nii.gz".format(os.path.basename(case_dir)))

        # Create input directory and copy input CTA to it
        input_dir = os.path.join(case_dir, "input")
        if not os.path.isdir(input_dir): os.mkdir(input_dir)
        # Create output directory
        output_dir = os.path.join(case_dir, "output")
        if not os.path.isdir(output_dir): os.mkdir(output_dir)

        # Copy original CTA to input directory with nnunet format (_0000.nii.gz)
        shutil.copyfile(case_path, os.path.join(input_dir, "{}_0000.nii.gz".format(os.path.basename(case_dir))))

        # We have to get rid of potential spaces in paths for the terminal commands to work 
        input_dir = input_dir.replace("\ ", "")
        input_dir = input_dir.replace(" ", "\ ")
        output_dir = output_dir.replace("\ ", "")
        output_dir = output_dir.replace(" ", "\ ")

        # Perform inference using nnUNet_predict command in the command line
        os.system("nnUNet_predict -i " + input_dir + " -o " + output_dir + " -t Task001_Arterial -m 3d_lowres -f all")

        # Set path back to normal
        input_dir = input_dir.replace("\\", "")
        output_dir = output_dir.replace("\\", "")

        # Copy segmentation nifti to case_dir
        shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_neck_segmentation.nii.gz".format(os.path.basename(case_dir))))

        # Remove input and output directories
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    # Slice full CTA into two nifties for head and neck
    print("Running segmentation preprocessing...")
    slice_cta_head_and_neck(case_dir)
    print("done")
    # Perform inference of the head CTA
    print("Performing inference (head)...")
    perform_inference_head(case_dir)
    print("done")
    # Perform inference of the neck CTA
    print("Performing inference (neck)...")
    perform_inference_neck(case_dir)
    print("done")
    # Join segmentations
    print("Joining segmentation")
    join_head_and_neck_segmentations(case_dir)
    print("done")

    # Remove useless files
    os.remove(os.path.join(case_dir, "{}_head.nii.gz".format(os.path.basename(case_dir))))
    os.remove(os.path.join(case_dir, "{}_head_segmentation.nii.gz".format(os.path.basename(case_dir))))
    os.remove(os.path.join(case_dir, "{}_neck.nii.gz".format(os.path.basename(case_dir))))
    os.remove(os.path.join(case_dir, "{}_neck_segmentation.nii.gz".format(os.path.basename(case_dir))))