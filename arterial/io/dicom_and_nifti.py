#   Copyright 2023 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os, shutil, glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import dicom2nifti
import dicom2nifti.settings as settings

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
    settings.disable_validate_orthogonal()
    settings.enable_resampling()
    settings.set_resample_spline_interpolation_order(1)
    settings.set_resample_padding(-1000)
    
    # Create a new dummy directry for temporary storage of the output nifti file
    output_dir = os.path.dirname(output_path)
    dummy_output_dir = os.path.join(output_dir, "temporary_dir_dicom2nifti")
    os.mkdir(dummy_output_dir)
    # Make conversion, it will create a nifti file in the dummy directory
    dicom2nifti.convert_directory(input_path, dummy_output_dir)
    # Move the new nifti to its output path with a proper, chosen name
    shutil.move(glob.glob(os.path.join(dummy_output_dir, "*.nii.gz"))[0], output_path)
    # Remove the final directory
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
