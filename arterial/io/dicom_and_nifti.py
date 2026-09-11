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
    os.makedirs(dummy_output_dir, exist_ok=True)
    # Make conversion, it will create a nifti file in the dummy directory
    try:
        dicom2nifti.convert_directory(input_path, dummy_output_dir)
        # Move the new nifti to its output path with a proper, chosen name
        shutil.move(glob.glob(os.path.join(dummy_output_dir, "*.nii.gz"))[0], output_path)
        # Remove the final directory
        shutil.rmtree(dummy_output_dir)
    except Exception as e:
        print(e)
        print("Conversion failed.")
        shutil.rmtree(dummy_output_dir)

def convert_to_las(nifti_file):
    # Load the NIfTI file
    img = nib.load(nifti_file)
    # Get the header affine matrix
    affine = img.affine
    # Get orientation
    orientation = nib.aff2axcodes(affine)
    
    if orientation == ("L", "A", "S"):
        # img.set_data_dtype(dtype)
        return img
    elif orientation == ("L", "P", "S"):
        # Change the sign of the first and second columns to flip the Anterior-Posterior axis
        affine[:, 1] = -affine[:, 1]

        # Also need to flip the data in the x and y axes
        data = img.get_fdata()
        data = data[:, ::-1, :]

        # Update the translation part of the affine matrix to compensate for the flipping of the data
        affine[1, 3] = affine[1, 3] - data.shape[1] * affine[1, 1]

        # Create a new NIfTI image with the new affine matrix and data
        img_las = nib.Nifti1Image(data, affine)
        
        # Return the new image
        return img_las
    
def convert_to_lps(nifti_file):
    # Load the NIfTI file
    img = nib.load(nifti_file)
    # Get the header affine matrix
    affine = img.affine
    # Get orientation
    orientation = nib.aff2axcodes(affine)
    
    if orientation == ("L", "P", "S"):
        # img.set_data_dtype(dtype)
        return img
    elif orientation == ("L", "A", "S"):
        # Change the sign of the first and second columns to flip the Anterior-Posterior axis
        affine[:, 1] = -affine[:, 1]

        # Also need to flip the data in the x and y axes
        data = img.get_fdata()
        data = data[:, ::-1, :]

        # Update the translation part of the affine matrix to compensate for the flipping of the data
        affine[1, 3] = affine[1, 3] - data.shape[1] * affine[1, 1]

        # Create a new NIfTI image with the new affine matrix and data
        img_lps = nib.Nifti1Image(data, affine)
        
        # Return the new image
        return img_lps

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
