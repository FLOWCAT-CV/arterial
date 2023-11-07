#   Copyright 2023 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import pickle, json

def load_pickle(path):
    """
    Loads pickle from path.

    Parameters
    ----------
    path : string or path-like object
        Path to pickle file.

    Returns
    -------
    pickle_object : object
        Object loaded from pickle.

    """
    with open(path, "rb") as f:
        pickle_object = pickle.load(f)

    return pickle_object

def save_pickle(pickle_object, path):
    """
    Saves pickle to path.

    Parameters
    ----------
    pickle_object : object
        Object to be saved as pickle.
    path : string or path-like object
        Path to save pickle file.

    Returns
    -------
    None.

    """
    with open(path, "wb") as f:
        pickle.dump(pickle_object, f)

def load_json(path):
    """
    Loads json from path.

    Parameters
    ----------
    path : string or path-like object
        Path to json file.

    Returns
    -------
    json_object : object
        Object loaded from json.

    """
    with open(path, "r") as f:
        json_object = json.load(f)

    return json_object

def save_json(json_object, path):
    """
    Saves json to path.

    Parameters
    ----------
    json_object : object
        Object to be saved as json.
    path : string or path-like object
        Path to save json file.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        json.dump(json_object, f, indent = 4)