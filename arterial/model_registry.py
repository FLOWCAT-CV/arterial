#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os


def models_dir():
    """
    Resolves the directory holding the trained model weights.

    Mirrors the resolution order of download_models.sh: the
    ARTERIAL_MODELS_DIR environment variable takes precedence; otherwise the
    weights are expected under $arterial_dir/models, where the download
    script extracts them by default.

    Returns
    -------
    models_dir : str
        Absolute path to the models directory.

    """
    env_models_dir = os.environ.get("ARTERIAL_MODELS_DIR")
    if env_models_dir:
        return os.path.abspath(os.path.expanduser(env_models_dir))
    arterial_dir = os.environ.get("arterial_dir")
    if arterial_dir:
        return os.path.join(os.path.abspath(os.path.expanduser(arterial_dir)), "models")
    raise EnvironmentError(
        "Cannot locate the Arterial model weights: neither ARTERIAL_MODELS_DIR nor "
        "arterial_dir is set. Run download_models.sh, or export "
        "ARTERIAL_MODELS_DIR=/path/to/models (see README, 'Model Weights')."
    )


def model_path(*parts):
    """
    Builds the path to a model file or directory inside the models directory.

    Parameters
    ----------
    *parts : str
        Path components relative to the models directory, for example
        ("landmark_detection", "six_landmarks_2ch.pth").

    Returns
    -------
    path : str
        Absolute path to the requested model file or directory.

    """
    return os.path.join(models_dir(), *parts)
