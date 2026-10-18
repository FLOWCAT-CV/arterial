#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os, shutil, glob
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation, io_orientation, inv_ornt_aff

def convert_dicom_to_nifti_sitk(input_path, output_path):
    """
    Passes dicom series to nifti using the sitk package.

    Parameters
    ----------
    input_path : string
        Path to the dicom series.
    output_path : string
        Path where final nifti will be stored.

    Returns
    -------

    """
    import SimpleITK as sitk  # optional dependency: pip install arterial[dicom]

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)

def convert_dicom_to_nifti_d2n(input_path, output_path):
    """
    Passes dicom series to nifti using the dicom2nifti package.

    Parameters
    ----------
    input_path : string
        Path to the dicom series.
    output_path : string
        Path where final nifti will be stored.

    Returns
    -------

    """
    import dicom2nifti  # optional dependency: pip install arterial[dicom]
    import dicom2nifti.settings as settings

    settings.disable_validate_orthogonal()
    settings.enable_resampling()
    settings.set_resample_spline_interpolation_order(1)
    settings.set_resample_padding(-1000)
    
    # Create a new dummy directry for temporary storage of the output nifti file
    output_dir = os.path.dirname(output_path)
    dummy_output_dir = os.path.join(output_dir, "temporary_dir_dicom2nifti")
    # Start from an empty directory, so that a nifti left by an earlier run cannot be picked up
    shutil.rmtree(dummy_output_dir, ignore_errors=True)
    os.makedirs(dummy_output_dir)
    # Make conversion, it will create a nifti file in the dummy directory
    try:
        dicom2nifti.convert_directory(input_path, dummy_output_dir)
        converted = glob.glob(os.path.join(dummy_output_dir, "*.nii.gz"))
        if len(converted) != 1:
            raise RuntimeError(f"dicom2nifti produced {len(converted)} nifti files from {input_path}, expected 1")
        # Move the new nifti to its output path with a proper, chosen name
        shutil.move(converted[0], output_path)
    finally:
        # Remove the temporary directory, also when the conversion fails (the error is raised to the caller)
        shutil.rmtree(dummy_output_dir, ignore_errors=True)

def convert_to_las(nifti_file):
    """
    Loads a NIfTI file and reorients it to LAS.

    Parameters
    ----------
    nifti_file : string or path-like object
        Path to the NIfTI file.

    Returns
    -------
    img : nibabel.Nifti1Image
        Image reoriented to LAS, from any source orientation.

    """
    return convert_orientation(nib.load(nifti_file), "LAS")

def convert_to_lps(nifti_file):
    """
    Loads a NIfTI file and reorients it to LPS.

    Parameters
    ----------
    nifti_file : string or path-like object
        Path to the NIfTI file.

    Returns
    -------
    img : nibabel.Nifti1Image
        Image reoriented to LPS, from any source orientation.

    """
    return convert_orientation(nib.load(nifti_file), "LPS")

def convert_orientation(nifti_input, target_orientation="RAS"):
    """
    Convert a NIfTI image to any desired orientation.

    Parameters
    ----------
    nifti_input : string or nibabel.Nifti1Image
        Path to the NIfTI file or a nibabel image object.
    target_orientation : string
        Three-letter orientation code (e.g., "RAS", "LAS", "LPS", "PIR").
        Each letter specifies the direction of the corresponding axis:
        - First letter: Left (L) or Right (R)
        - Second letter: Anterior (A) or Posterior (P)
        - Third letter: Inferior (I) or Superior (S)
        Default is "RAS".

    Returns
    -------
    nibabel.Nifti1Image
        NIfTI image reoriented to the target orientation.

    """
    # Load the image if a path is provided
    if isinstance(nifti_input, str):
        img = nib.load(nifti_input)
    else:
        img = nifti_input
    
    # Validate target orientation string
    target_orientation = target_orientation.upper()
    if len(target_orientation) != 3:
        raise ValueError(f"Target orientation must be a 3-letter string, got: {target_orientation}")
    
    valid_codes = {'L', 'R', 'A', 'P', 'I', 'S'}
    for code in target_orientation:
        if code not in valid_codes:
            raise ValueError(f"Invalid orientation code '{code}'. Valid codes are: {valid_codes}")
    
    # Get current orientation from affine
    current_axcodes = nib.aff2axcodes(img.affine)
    
    # Check if already in target orientation
    if tuple(current_axcodes) == tuple(target_orientation):
        return img
    
    # Get current orientation array
    current_ornt = io_orientation(img.affine)
    
    # Parse target orientation string to orientation array
    target_ornt = axcodes2ornt(tuple(target_orientation))
    
    # Compute transformation from current to target orientation
    transform = ornt_transform(current_ornt, target_ornt)
    
    # Apply transformation to data
    data = img.get_fdata()
    new_data = apply_orientation(data, transform)
    
    # Compute new affine matrix
    new_affine = img.affine @ inv_ornt_aff(transform, data.shape)
    
    # Create new NIfTI image preserving header information
    new_img = nib.Nifti1Image(new_data, new_affine, img.header)
    
    return new_img
