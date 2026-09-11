#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os, pickle, json, re, vtk

import numpy as np
import nibabel as nib

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
        json.dump(json_object, f, indent=4, default=_json_default)


def _json_default(value):
    """
    Converts numpy scalars and arrays so that json.dump accepts them.

    Parameters
    ----------
    value : object
        Object json could not serialize.

    Returns
    -------
    converted : object
        A builtin equivalent of value.

    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def load_vtkpolydata(path):
    """
    Loads a vtkPolyData object from a legacy .vtk file.

    Parameters
    ----------
    path : str
        Path to the file where the vtkPolyData object is saved.

    Returns
    -------
    vtk_poly_data : vtk.vtkPolyData
        vtkPolyData object loaded from file.

    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"vtkPolyData file not found: {path}")
    vtk_poly_data_reader = vtk.vtkPolyDataReader()
    vtk_poly_data_reader.SetFileName(path)
    vtk_poly_data_reader.Update()

    return vtk_poly_data_reader.GetOutput()

def save_vtkpolydata(vtk_poly_data, path):
    """
    Saves a vtkPolyData object to a legacy .vtk file.

    Parameters
    ----------
    vtk_poly_data : vtk.vtkPolyData
        vtkPolyData object to be saved.
    path : str
        Path to the file where the vtkPolyData object will be saved.

    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)
    writer.SetInputData(vtk_poly_data)
    writer.SetFileName(path)
    if not writer.Write():
        raise IOError(f"Could not write vtkPolyData to {path}")

def save_vtkpolydata_as_stl(vtk_poly_data, path):
    """
    Saves a vtkPolyData object to a file in .stl format.

    Parameters
    ----------
    vtk_poly_data : vtk.vtkPolyData
        vtkPolyData object to be saved.
    path : str
        Path to the file where the vtkPolyData object will be saved.

    """
    writer = vtk.vtkSTLWriter()
    writer.SetFileTypeToBinary()
    writer.SetInputData(vtk_poly_data)
    writer.SetFileName(path)
    if not writer.Write():
        raise IOError(f"Could not write STL to {path}")

def load_vtk_list_from_dir(dir_path):
    """
    Load a list of vtkPolyData objects from a directory.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the vtkPolyData files.

    Returns
    -------
    vtk_list : list
        List of vtkPolyData objects, in natural filename order so that
        ``model_10.vtk`` comes after ``model_2.vtk``.

    """
    filenames = sorted((f for f in os.listdir(dir_path) if f.endswith(".vtk")), key=natural_sort_key)
    vtk_list = [load_vtkpolydata(os.path.join(dir_path, filename)) for filename in filenames]
    
    return vtk_list

def natural_sort_key(text):
    """
    Builds a sort key that orders embedded integers numerically.

    Parameters
    ----------
    text : str
        Filename or any string.

    Returns
    -------
    key : list
        Alternating string and int chunks.

    """
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def save_numpy(array, path):
    """
    Saves a numpy array to a file in .npy format.

    Parameters
    ----------
    array : numpy.ndarray
        Numpy array to be saved.
    path : str
        Path to the file where the numpy array will be saved.

    """
    np.save(path, array)

def load_numpy(path):
    """
    Loads a numpy array from a file in .npy format.

    Parameters
    ----------
    path : str
        Path to the file where the numpy array is saved.

    Returns
    -------
    array : numpy.ndarray
        Numpy array loaded from file.

    """
    return np.load(path, allow_pickle=True)

def save_nifti(nifti, path):
    """
    Saves a nibabel image to a file in .nii or .nii.gz format.

    Parameters
    ----------
    nifti : nibabel.Nifti1Image
        Image to be saved.
    path : str
        Path to the file where the image will be saved.

    """
    nib.save(nifti, path)

def load_nifti(path):
    """
    Loads a nibabel image from a file in .nii or .nii.gz format.

    Parameters
    ----------
    path : str
        Path to the file where the numpy array is saved.

    Returns
    -------
    array : numpy.ndarray
        Numpy array loaded from file.

    """
    return nib.load(path)

def serialize_vtk_polydata(polydata):
    """
    Serializes vtkPolyData to binary string. This
    is used for example to input a vtlPolyData object
    to a remote function.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        vtkPolyData object to be serialized.

    Returns
    -------
    str
        Binary string representation of the vtkPolyData object.

    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.WriteToOutputStringOn()
    writer.Update()
    return writer.GetOutputString()

def deserialize_vtk_polydata(data):
    """
    Deserializes binary string to vtkPolyData. This is used
    to retrieve a vtkPolyData object from a remote function.

    Parameters
    ----------
    data : str
        Binary string representation of a vtkPolyData object.

    Returns
    -------
    vtk.vtkPolyData
        vtkPolyData object deserialized from the binary string.

    """
    reader = vtk.vtkPolyDataReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(data)
    reader.Update()
    return reader.GetOutput()