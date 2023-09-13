#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import shutil

from arterial.segmentation.utils import slice_cta_head_and_neck, join_head_and_neck_segmentations, recenter_thrombus_patch, crop_intracranial_cta

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

    >>> case_dir/{os.path.basename(case_dir)}_vessel_segmentation.nii.gz

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
    case_path = os.path.join(case_dir, "{}_cta.nii.gz".format(os.path.basename(case_dir)))

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
    shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))

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

    >>> case_dir/{os.path.basename(case_dir)}_vessel_segmentation.nii.gz

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

        >>> case_dir/{os.path.basename(case_dir)}_head_vessel_segmentation.nii.gz

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        # Define path to CTA
        case_path = os.path.join(case_dir, "{}_cta_head.nii.gz".format(os.path.basename(case_dir)))

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
        os.system("nnUNet_predict -i " + input_dir + " -o " + output_dir + " -t Task002_Intracranial -m 3d_fullres -f all")

        # Set path back to normal
        input_dir = input_dir.replace("\\", "")
        output_dir = output_dir.replace("\\", "")

        # Copy segmentation nifti to case_dir
        shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_head_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))

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

        >>> case_dir/{os.path.basename(case_dir)}_neck_vessel_segmentation.nii.gz

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        # Define path to CTA
        case_path = os.path.join(case_dir, "{}_cta_neck.nii.gz".format(os.path.basename(case_dir)))

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
        shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_neck_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))

        # Remove input and output directories
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    # Slice full CTA into two nifties for head and neck
    slice_cta_head_and_neck(case_dir)
    # Perform inference of the head CTA
    perform_inference_head(case_dir)
    # Perform inference of the neck CTA
    perform_inference_neck(case_dir)
    # Join segmentations
    join_head_and_neck_segmentations(case_dir)

    # Remove useless files
    os.remove(os.path.join(case_dir, "{}_cta_head.nii.gz".format(os.path.basename(case_dir))))
    os.remove(os.path.join(case_dir, "{}_head_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))
    os.remove(os.path.join(case_dir, "{}_cta_neck.nii.gz".format(os.path.basename(case_dir))))
    os.remove(os.path.join(case_dir, "{}_neck_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))

def perform_inference_intracranial(case_dir):
    """
    Performs inference of the intracranial part of the CTA.

    It uses a network trained for inference of the intracranial part of the image, 
    trained on a dataset including only cerebral arteries.

    It also assumes that nnunet paths are set correctly, according to the guidelines 
    provided in the nnunet documentation (see documentation in
    <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
    The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
    set in the ~/.bashrc (or ~/.zshrc) file as:

    >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

    Saves binary map as:

    >>> case_dir/{os.path.basename(case_dir)}_intracranial_vessel_segmentation.nii.gz

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------
    
    """
    # Compute intracranial CTA
    if not os.path.isfile(os.path.join(case_dir, "{}_cta_intracranial.nii.gz".format(os.path.basename(case_dir)))):
        print("Cropping intracranial CTA patch...")
        crop_intracranial_cta(case_dir)
        print("done")
    # Define path to CTA
    case_path = os.path.join(case_dir, "{}_cta_intracranial.nii.gz".format(os.path.basename(case_dir)))

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
    os.system("nnUNet_predict -i " + input_dir + " -o " + output_dir + " -t Task002_Intracranial -m 3d_fullres -f all")

    # Set path back to normal
    input_dir = input_dir.replace("\\", "")
    output_dir = output_dir.replace("\\", "")

    # Copy segmentation nifti to case_dir
    shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_intracranial_vessel_segmentation.nii.gz".format(os.path.basename(case_dir))))

    # Remove input and output directories
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)

def perform_inference_thrombus(case_dir):
    """
    Performs inference of the bimodal CTA + NCCT localized patch for thrombus segmentation.

    It uses a network trained for inference of the bimodal CT + NCCT patch suspect of containing 
    an occlusion. It assumes that each of these patches are stored in the case_dir directory, following
    the naming convention:

    >>> {case_id}_cta_thrombus_patch.nii.gz
    >>> {case_id}_ncct_thrombus_patch.nii.gz

    It also assumes that nnunet paths are set correctly, according to the guidelines 
    provided in the nnunet documentation (see documentation in
    <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
    The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
    set in the ~/.bashrc (or ~/.zshrc) file as:

    >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

    Saves binary map as:

    >>> case_dir/{os.path.basename(case_dir)}_thrombus_segmentation.nii.gz

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------
    
    """
    # Define path to CTA and NCCT patches
    case_path_cta = os.path.join(case_dir, "{}_cta_thrombus_patch.nii.gz".format(os.path.basename(case_dir)))
    case_path_ncct = os.path.join(case_dir, "{}_ncct_thrombus_patch.nii.gz".format(os.path.basename(case_dir)))

    # Create input directory and copy input CTA to it
    input_dir = os.path.join(case_dir, "input")
    if not os.path.isdir(input_dir): os.mkdir(input_dir)
    # Create output directory
    output_dir = os.path.join(case_dir, "output")
    if not os.path.isdir(output_dir): os.mkdir(output_dir)

    # Copy cropped CTA and NCCT to input directory with nnunet format (_0000.nii.gz, _0001.nii.gz)
    shutil.copyfile(case_path_cta, os.path.join(input_dir, "{}_0000.nii.gz".format(os.path.basename(case_dir))))
    shutil.copyfile(case_path_ncct, os.path.join(input_dir, "{}_0001.nii.gz".format(os.path.basename(case_dir))))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    input_dir = input_dir.replace("\ ", "")
    input_dir = input_dir.replace(" ", "\ ")
    output_dir = output_dir.replace("\ ", "")
    output_dir = output_dir.replace(" ", "\ ")

    # Perform inference using nnUNet_predict command in the command line
    os.system("nnUNet_predict -i " + input_dir + " -o " + output_dir + " -t Task003_Thrombus -m 3d_fullres -f all")

    # Set path back to normal
    input_dir = input_dir.replace("\\", "")
    output_dir = output_dir.replace("\\", "")

    # Copy segmentation nifti to case_dir
    shutil.copyfile(os.path.join(output_dir, "{}.nii.gz".format(os.path.basename(case_dir))), os.path.join(case_dir, "{}_thrombus_segmentation.nii.gz".format(os.path.basename(case_dir))))

    # Remove input and output directories
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)

def perform_dynamic_iterative_thrombus_inference(case_dir):
    """
    Performs dynamic iterative inference of the bimodal CTA + NCCT localized patch for thrombus segmentation.

    This method performs a first inference, then re-centers the patch based on the first inference result,
    and performs a second inference. This iterative approach can potentially improve the accuracy of thrombus detection.

    It uses a network trained for inference of the bimodal CT + NCCT patch suspect of containing 
    an occlusion. It assumes that each of these patches are stored in the case_dir directory, following
    the naming convention:

    >>> {case_id}_cta_thrombus_patch.nii.gz
    >>> {case_id}_ncct_thrombus_patch.nii.gz

    It also assumes that nnunet paths are set correctly, according to the guidelines 
    provided in the nnunet documentation (see documentation in
    <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md>).
    The only path that is needed for correct inference is RESULTS_FOLDER. This should be 
    set in the ~/.bashrc (or ~/.zshrc) file as:

    >>> export RESULTS_FOLDER="/path/to/arterial/arterial/segmentation/models"

    Saves binary map as:

    >>> case_dir/{os.path.basename(case_dir)}_thrombus_segmentation.nii.gz

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------
    
    """
    
    # Perform a first inference
    perform_inference_thrombus(case_dir)
    # Recenter patch
    recenter_thrombus_patch(case_dir)
    # Perform a second inference
    perform_inference_thrombus(case_dir)