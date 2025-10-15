#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import torch
import numpy as np
import nibabel as nib

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import warnings
warnings.filterwarnings("ignore") # There is a warning that is wrong with nnunet 
                                  # (something about the version of trained model)
    
def perform_single_inference_nnunet(img_array, img_affine, mode="extracranial_vessels", nnunet_mode="3d_lowres", use_vanilla_nnunet=True, return_probabilities=False, set_threshold_099=False):
    """
    Performs inference of the CTA in img_array with a trained nnunet models. The combination of
    mode and nnunet_mode should be consistent with the trained model that is going to be used. 

    Parameters
    ----------
    img_array : numpy array
        3D numpy array representing the CTA image.
    img_affine : numpy array
        4x4 numpy array representing the affine transformation of the CTA image.
    mode : string
        Mode of the segmentation. It can be either "extracranial_vessels" or "intracranial_vessels".
    nnunet_mode : string
        Mode of the nnunet model. It can be either "3d_lowres" or "3d_fullres".

    Returns
    -------
    segmentation_nifti : nibabel.nifti1.Nifti1Image
        Nifti image with the segmentation mask.
    segmentation_array : numpy array
        3D numpy array representing the segmentation mask.
    probabilities_array : numpy array
        3D numpy array representing the probabilities of the segmentation mask.
    """
    # We are not using the nnunet imageio readers as we do not want to unnecessarily load the
    # same image several times, and most of our pipeline is implemented using the nibabel library.
    # Thus, even if this is not the recommended approach, we apply these transformations to accomodate
    # the nnunet inference to our pipeline.
    img = np.expand_dims(img_array.transpose([2, 1, 0]), axis=0).astype(np.float32)  # reverse axis order to match SITK (from nnunetv2 repo)
    props = {"spacing": tuple(np.abs(np.diag(img_affine, k=0)[:3][::-1]))}

    # Read device
    device = torch.device("cpu") # This will take forever on 'cpu', only to be used for debugging
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.75,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # Initializes the network architecture, loads the checkpoint
    print("Using vanilla nnunet:", use_vanilla_nnunet)
    predictor.initialize_from_trained_model_folder(
        os.path.join(os.environ["arterial_dir"], f'segmentation/models/{mode}/nnUNetTrainer__nnUNetPlans__{nnunet_mode}'),
        use_folds=("all" if use_vanilla_nnunet else "0",), # Models set by use_vanilla_nnunet. "0" is model trained with nnUNetClDiceLossTrainer, "all" is trained with vanilla nnUNetTrainer
        # use_folds=("0",), # Models trained with nnUNetClDiceLossTrainer (these have an error on inference, sometimes return tensors with nan values)
        # use_folds=("all",), # Models trained with vanilla nnUNetTrainer
        checkpoint_name='checkpoint_final.pth',
    )

    # Perform inference
    if return_probabilities or set_threshold_099:
        segmentation_array, probabilities_array = predictor.predict_single_npy_array(img, props, None, None, save_or_return_probabilities=True)
        # Reverse axis order to match nibabel
        segmentation_array = segmentation_array.transpose([2, 1, 0])
        segmentation_nifti = nib.Nifti1Image(segmentation_array, img_affine)
        probabilities_array = probabilities_array[1, :, :, :].transpose([2, 1, 0])
        probabilities_nifti = nib.Nifti1Image(probabilities_array, img_affine)
        
        if set_threshold_099:
            segmentation_array = np.zeros_like(probabilities_array)
            segmentation_array[probabilities_array >= 0.99] = 1
            segmentation_nifti = nib.Nifti1Image(segmentation_array, img_affine)
            if return_probabilities:
                return segmentation_nifti, segmentation_array, probabilities_nifti
            else:
                return segmentation_nifti, segmentation_array
        else:
            return segmentation_nifti, segmentation_array, probabilities_nifti
    else:
        segmentation_array = predictor.predict_single_npy_array(img, props, None, None, False)
        # Reverse axis order to match nibabel
        segmentation_array = segmentation_array.transpose([2, 1, 0])
        segmentation_nifti = nib.Nifti1Image(segmentation_array, img_affine)

        return segmentation_nifti, segmentation_array
        