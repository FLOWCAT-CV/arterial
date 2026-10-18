#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from functools import partial

import torch
import numpy as np
import nibabel as nib

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
import warnings
warnings.filterwarnings("ignore") # There is a warning that is wrong with nnunet 
                                  # (something about the version of trained model)

from arterial import model_registry

# nnUNetPredictor objects already loaded in this process, keyed by (model folder, folds, device)
_PREDICTORS = {}

def get_nnunet_predictor(model_folder, use_folds, device):
    """
    Returns the nnUNetPredictor for a trained model folder, building it and loading its checkpoint
    only the first time it is requested in this process.

    Parameters
    ----------
    model_folder : string or path-like object
        Path to the trained nnunet model folder (the one containing plans.json and the fold directories).
    use_folds : tuple
        Folds of the model to be used, e.g. ("all",) or ("0",).
    device : torch.device
        Device where inference is run.

    Returns
    -------
    predictor : nnunetv2.inference.predict_from_raw_data.nnUNetPredictor
        Initialized predictor, shared by every caller with the same model folder, folds and device.

    """
    key = (str(model_folder), tuple(use_folds), str(device))
    if key not in _PREDICTORS:
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
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=use_folds,
            checkpoint_name='checkpoint_final.pth',
        )
        _PREDICTORS[key] = predictor

    return _PREDICTORS[key]

class ResamplingConfigurationManager(ConfigurationManager):
    """
    ConfigurationManager of a trained model with the resampling functions given directly, instead of read from the
    model plans. Everything else is shared with the original ConfigurationManager.

    """
    def __init__(self, configuration_manager, resampling_fns):
        """
        Initializes object of the ResamplingConfigurationManager class.

        Parameters
        ----------
        configuration_manager : nnunetv2.utilities.plans_handling.plans_handler.ConfigurationManager
            ConfigurationManager built from the model plans.
        resampling_fns : dict
            Resampling functions with the keys "data", "seg" and "probabilities".

        """
        self.__dict__.update(configuration_manager.__dict__)
        self.resampling_fns = resampling_fns

    @property
    def resampling_fn_data(self):
        return self.resampling_fns["data"]

    @property
    def resampling_fn_seg(self):
        return self.resampling_fns["seg"]

    @property
    def resampling_fn_probabilities(self):
        return self.resampling_fns["probabilities"]

def resample_with_fallback(torch_fn, plans_fn, *args, **kwargs):
    """
    Resamples with torch_fn and, if the GPU runs out of memory, repeats only this resampling with plans_fn.

    Parameters
    ----------
    torch_fn : callable
        nnunetv2 torch resampling function.
    plans_fn : callable
        Resampling function of the model plans (scipy/skimage, CPU).
    *args, **kwargs
        Arguments of the resampling function (data, new_shape, current_spacing, new_spacing).

    Returns
    -------
    resampled : numpy array or torch.Tensor
        Resampled data.

    """
    try:
        return torch_fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        print("Not enough GPU memory for torch resampling, using scipy resampling for this step")
        torch.cuda.empty_cache()
        return plans_fn(*args, **kwargs)

def set_nnunet_resampling(predictor, resampling="scipy", device=torch.device("cpu"), num_threads=None):
    """
    Sets the resampling functions used by the predictor before and after the network. "scipy" restores the
    ones in the model plans (scipy/skimage, CPU). "torch" uses the nnunetv2 torch resampler on the given device,
    falling back to the plans resampling for any step that runs out of GPU memory. The model plans file is not
    modified.

    Parameters
    ----------
    predictor : nnunetv2.inference.predict_from_raw_data.nnUNetPredictor
        Initialized predictor.
    resampling : string
        Either "scipy" or "torch".
    device : torch.device
        Device used by the torch resampler.
    num_threads : int, optional
        CPU threads used by the torch resampler. If None, the current torch setting is used.

    """
    assert resampling in ["scipy", "torch"], "resampling should be either 'scipy' or 'torch'."

    # Keep the ConfigurationManager built from the plans, to be able to switch back to it
    if not hasattr(predictor, "plans_configuration_manager"):
        predictor.plans_configuration_manager = predictor.configuration_manager
    plans_configuration_manager = predictor.plans_configuration_manager

    if resampling == "scipy":
        predictor.configuration_manager = plans_configuration_manager
    else:
        resampling_fns = {}
        for name in ["data", "seg", "probabilities"]:
            plans_fn = getattr(plans_configuration_manager, f"resampling_fn_{name}")
            torch_fn = partial(resample_torch_fornnunet,
                               is_seg=name == "seg",
                               num_threads=num_threads if num_threads is not None else torch.get_num_threads(),
                               device=device,
                               force_separate_z=plans_configuration_manager.configuration[f"resampling_fn_{name}_kwargs"].get("force_separate_z"))
            resampling_fns[name] = partial(resample_with_fallback, torch_fn, plans_fn)
        predictor.configuration_manager = ResamplingConfigurationManager(plans_configuration_manager, resampling_fns)

def run_nnunet_prediction(predictor, img, props, save_or_return_probabilities=False, resampling="torch", num_threads=None):
    """
    Runs predictor.predict_single_npy_array with the requested resampling. Torch resampling is only used on a
    CUDA device.

    Parameters
    ----------
    predictor : nnunetv2.inference.predict_from_raw_data.nnUNetPredictor
        Initialized predictor.
    img : numpy array
        4D numpy array (channel, z, y, x) with the image, in the nnunet axis order.
    props : dict
        Image properties, with the "spacing" key in the nnunet axis order.
    save_or_return_probabilities : bool
        Whether to also return the class probabilities.
    resampling : string
        Either "scipy" or "torch".
    num_threads : int, optional
        CPU threads used by the torch resampler.

    Returns
    -------
    prediction : numpy array or tuple
        Output of predictor.predict_single_npy_array.

    """
    if resampling == "torch" and predictor.device.type != "cuda":
        print("Torch resampling needs a CUDA device, using scipy resampling instead")
        resampling = "scipy"

    set_nnunet_resampling(predictor, resampling, predictor.device, num_threads)
    return predictor.predict_single_npy_array(img, props, None, None, save_or_return_probabilities)

def perform_single_inference_nnunet(img_array, img_affine, mode="extracranial_vessels", nnunet_mode="3d_lowres", use_vanilla_nnunet=True, return_probabilities=False, set_threshold_099=False, resampling="torch", num_threads=None):
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
    resampling : string
        Resampling used by nnunet before and after the network: "scipy" (CPU, as in the model plans)
        or "torch" (GPU). Torch resampling falls back to scipy without a CUDA device.
    num_threads : int, optional
        CPU threads used by the torch resampler. If None, the current torch setting is used.

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

    # Get the nnUNetPredictor (built and loaded only once per process)
    print("Using vanilla nnunet:", use_vanilla_nnunet)
    predictor = get_nnunet_predictor(
        model_registry.model_path("segmentation", mode, f"nnUNetTrainer__nnUNetPlans__{nnunet_mode}"),
        use_folds=("all" if use_vanilla_nnunet else "0",), # Models set by use_vanilla_nnunet. "0" is model trained with nnUNetClDiceLossTrainer, "all" is trained with vanilla nnUNetTrainer
        # use_folds=("0",), # Models trained with nnUNetClDiceLossTrainer (these have an error on inference, sometimes return tensors with nan values)
        # use_folds=("all",), # Models trained with vanilla nnUNetTrainer
        device=device,
    )

    # Perform inference
    if return_probabilities or set_threshold_099:
        segmentation_array, probabilities_array = run_nnunet_prediction(predictor, img, props, True, resampling, num_threads)
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
                return segmentation_nifti, segmentation_array, None
        else:
            return segmentation_nifti, segmentation_array, probabilities_nifti
    else:
        segmentation_array = run_nnunet_prediction(predictor, img, props, False, resampling, num_threads)
        # Reverse axis order to match nibabel
        segmentation_array = segmentation_array.transpose([2, 1, 0])
        segmentation_nifti = nib.Nifti1Image(segmentation_array, img_affine)

        return segmentation_nifti, segmentation_array, None
        