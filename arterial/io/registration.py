#   Copyright 2023 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import ants

import numpy as np
import nibabel as nib

def registration_ant(fixed_image_path, moving_image_path, output_path, tranformation_type = "Affine"):
    """
    Performs registration wiht the ants library. Modes include "AffineFast", "Affine" or SyN".

    
    """
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)

    if tranformation_type == "AffineFast":
        mytx = ants.registration(fixed_image, moving_image, type_of_transform = 'AffineFast')
                            #reg_iterations=(1, 1),
                            #aff_iterations=(1, 1), 
                            #aff_shrink_factors=(4, 4),     
                            #aff_smoothing_sigmas=(5, 5), verbose = True)
    elif tranformation_type == "Affine":
        mytx = ants.registration(fixed_image, moving_image, type_of_transform = 'Affine')
    elif tranformation_type == "SyN":
        # ANTsPy registration command
        mytx = ants.registration(fixed_image, 
                                 moving_image, 
                                 type_of_transform='SyN')
    else:
        raise RuntimeError("Please input a valid ransformation type ['Affinefast', 'Affine', 'SyN']")
    
    # Apply the transformation to the moving image
    warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=mytx['fwdtransforms'])
    
    # Save the transformed image
    warped_image.to_file(output_path)

    # Pass dtype of final nifti to int16
    nifti = nib.load(output_path)
    nib.save(nifti, output_path, dtype = np.int16)

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