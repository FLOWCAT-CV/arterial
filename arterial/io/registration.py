#   Copyright 2023 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import ants

import numpy as np
import nibabel as nib

def registration_ant(fixed_image_path, moving_image_path, output_path, tranformation_type = "Affine"):
    """
    Possible types of transform: "AffineFast", "Affine", "SyN"
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