#   Copyright 2023 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import ants

import numpy as np
import nibabel as nib

import scipy.io as sio

from scipy.ndimage import zoom

def registration(fixed_image_path, moving_image_path, output_image_path, transformation_type = "Affine", save_transform = False, label = False):
    """
    Performs registration of the moving image to the fixed image. Possible types of transform: "AffineFast", "Affine", "SyN"

    Parameters
    ----------
    fixed_image_path : str
        Path to the fixed image.
    moving_image_path : str
        Path to the moving image.
    output_image_path : str
        Path to the output image.
    transformation_type : str, optional
        Type of transformation. The default is "Affine".
    save_transform : bool, optional
        If True, the transformation matrix is saved. The default is False.
    label : bool, optional
        If True, the transformation is applied to the moving image using genericLabel interpolator. The default is False.

    Raises
    ------
    TypeError
        If the transformation type is not one of the following: "AffineFast", "Affine", "SyN".
        
    """
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)

    if transformation_type == "AffineFast":
        transform = ants.registration(fixed_image, moving_image, type_of_transform = 'AffineFast')
    elif transformation_type == "Affine":
        transform = ants.registration(fixed_image, moving_image, type_of_transform = 'Affine')
    elif transformation_type == "SyN":
        transform = ants.registration(fixed_image, moving_image, type_of_transform = 'SyN')
    
    # Apply the transformation to the moving image
    if label:
        warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transform['fwdtransforms'], interpolator='genericLabel')
    else:
        warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transform['fwdtransforms'])
    
    # Save the transformed image
    warped_image.to_file(output_image_path)

    # Pass dtype of final nifti to int16
    nifti = nib.load(output_image_path)
    nib.save(nifti, output_image_path, dtype = np.int32)
    
    if save_transform:
        # Save the transformation matrix
        np.savetxt(os.path.join(os.path.dirname(output_image_path), 'transfomation_matrix.txt'), transform['fwdtransforms'], fmt='%s')
        
def registration_mask(fixed_image_path, moving_mask_path, output_mask_path, transformation_file = None, remove_transformation = True):
    """
    Applies the transformation matrix to the mask.

    Parameters
    ----------
    fixed_image_path : str
        Path to the fixed image.
    moving_mask_path : str
        Path to the moving mask.
    output_mask_path : str
        Path to the output mask.
    transformation_file : str, optional
        Path to the transformation matrix. The default is None.
    remove_transformation : bool, optional
        If True, the transformation matrix and all the transformation files are removed. The default is True.

    Raises
    ------
    TypeError
        If the transformation file is not provided and the default path does not exist.

    """
    if transformation_file is None:
        if not os.path.exists(os.path.join(os.path.dirname(output_mask_path), 'transfomation_matrix.txt')):
            raise TypeError("Text file with transformation path does not exist.")       
        else:
            transformation_file = os.path.join(os.path.dirname(output_mask_path), 'transfomation_matrix.txt')

    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_mask_path)
    
    # Load the transformation matrix
    with open(transformation_file, 'r') as f:
        transformations = [line.strip() for line in f]
      
    # Apply the transformation to the mask
    warped_mask = ants.apply_transforms(fixed = fixed_image, 
                                        moving = moving_image, 
                                        transformlist = transformations, 
                                        interpolator='genericLabel')
    
    # Save the transformed mask
    warped_mask.to_file(output_mask_path)
    
    if remove_transformation:
        for transformation_path in transformations:
            os.remove(transformation_path)
        os.remove(transformation_file)

def registration_brainsfit(fixed_image_path, moving_image_path, output_path):
    """
    Performs registration wiht the BRAINSFit cli-module from Slicer.
    Slicer has to be set up for this to work.

    """
    SLICER_PATH = os.environ["slicer_path"]
    os.system(f"{SLICER_PATH} --launch BRAINSFit --fixedVolume {fixed_image_path} --movingVolume {moving_image_path} " +
              f"--samplingPercentage 0.002 --splineGridSize 14,10,12 --outputVolume {output_path} --initializeTransformMode Off " +
              "--useAffine --maskProcessingMode NOMASK --medianFilterSize 0,0,0 --removeIntensityOutliers 0 --outputVolumePixelType float " + 
              "--backgroundFillValue 0 --interpolationMode Linear --numberOfIterations 1500 --maximumStepLength 0.05 --minimumStepLength 0.001 " +
              "--relaxationFactor 0.5 --translationScale 1000 --reproportionScale 1 --skewScale 1 --maxBSplineDisplacement 0 " + 
              "--fixedVolumeTimeIndex 0 --movingVolumeTimeIndex 0 --numberOfHistogramBins 50 --numberOfMatchPoints 10 --costMetric MMI " + 
              "--maskInferiorCutOffFromCenter 1000 --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --numberOfSamples 0 --failureExitCode -1 " + 
              "--numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 2e+13 --projectedGradientTolerance 1e-05 --maximumNumberOfEvaluations 900 " +
              "--maximumNumberOfCorrections 25 --metricSamplingStrategy Random")
    
def convert_to_orientation(nifti, reference_ornt = ("R", "A", "S")):
    """
    Converts nifti orientation to that of reference.

    Parameters
    ----------
    nifti : nibabel.nifti1.Nifti1Image
        Nifti image to be converted.
    reference_ornt : tuple, optional
        Orientation to convert to. The default is ("R", "A", "S").

    Returns
    -------
    reoriented_nifti : nibabel.nifti1.Nifti1Image
        Reoriented nifti image.
    
    """
    # Get orientation from affine matrices
    nifti_ornt = nib.aff2axcodes(nifti.affine)

    if nifti_ornt == reference_ornt:
        print("Nifti orientation already in {}".format(reference_ornt))
        return nifti
    else:
        print("Converting orientation from {} to {}".format(nifti_ornt, reference_ornt))
        data = nifti.get_fdata()
        affine = nifti.affine
        # Get orientation transform from axcodes
        ornt = nib.orientations.axcodes2ornt(reference_ornt)
        # Apply orientation transform to array
        data = nib.orientations.apply_orientation(data, ornt)
        # Update affine matrix
        for idx in range(len(nifti_ornt)):
            if nifti_ornt[idx] != reference_ornt[idx]:
                affine[idx, 3] = affine[idx, 3] + affine[idx, idx] * (data.shape[idx] - 1)
                affine[idx, idx] = -affine[idx, idx]

        reoriented_nifti = nib.Nifti1Image(data, affine=affine)

        return reoriented_nifti

def mat_to_affine(mat_path):
    """
    Reads the affine transformation from a .mat file and returns it as a numpy array.

    Parameters
    ----------
    mat_path : str
        Path to .mat file.
        
    Returns
    -------
    affine : numpy.ndarray
        Affine matrix.
    """
    mat = sio.loadmat(mat_path)
    rotation_scaling = np.array(mat["AffineTransform_float_3_3"]).reshape(3, 4)
    translation = np.array(mat["fixed"]).reshape(3, 1)
    affine = np.eye(4)
    affine[:3, :4] = rotation_scaling
    affine[:3, 3] = translation.ravel()
    return affine

def resample_nifti(nifti, target_spacing):
    """
    Resamples the nifti image to the target spacing.

    Parameters
    ----------
    nifti : nibabel.nifti1.Nifti1Image
        Nifti image to be resampled.
    target_spacing : float or array_like
        Target spacing of the resampled image.  
        If float, isotropic spacing is used.
        If array_like, spacing is used for each axis.
    
    Returns
    -------
    new_img : nibabel.nifti1.Nifti1Image
        Resampled nifti image.

    """
    data = nifti.get_fdata()
    header = nifti.header
    affine = nifti.affine

    # Original spacing (from the affine matrix)
    original_spacing = nifti.header.get_zooms()
    # If target_spacing is a single value, make it isotropic
    if isinstance(target_spacing, float):
        target_spacing = np.array([target_spacing, target_spacing, target_spacing])

    # Calculate zoom factors
    zoom_factors = original_spacing / target_spacing

    # Resample the image using 3rd order spline interpolation
    resampled_data = zoom(data, zoom_factors, order=5)

    # Update the header and affine matrix
    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.append(target_spacing, [1]))
    for idx in range(3):
        new_affine[idx, idx] *= np.sign(affine[idx, idx])
    new_img = nib.Nifti1Image(resampled_data, new_affine, header)

    return new_img