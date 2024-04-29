#   Copyright 2023 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os, pickle, json, vtk

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
        json.dump(json_object, f, indent = 4)

def load_vtkpolydata(path):
    """
    Loads a vtkPolyData object from a file in .vtp format.

    Parameters
    ----------
    path : str
        Path to the file where the vtkPolyData object is saved.

    Returns
    -------
    vtk_poly_data : vtk.vtkPolyData
        vtkPolyData object loaded from file.

    """
    vtk_poly_data_reader = vtk.vtkPolyDataReader()
    vtk_poly_data_reader.SetFileName(path)
    vtk_poly_data_reader.Update()

    return vtk_poly_data_reader.GetOutput()

def save_vtkpolydata(vtk_poly_data, path):
    """
    Saves a vtkPolyData object to a file in .vtp format.

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
    writer.Write()

def load_vtk_list_from_dir(dir_path):
    """
    Load a list of vtkPolyData objects from a directory.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the vtkPolyData files.
    mode : str, optional
        Mode of the arterial feature extraction pipeline. The default is "extracranial_vessels".

    Returns
    -------
    vtk_list : list
        List of vtkPolyData objects.

    """
    vtk_list = [load_vtkpolydata(os.path.join(dir_path, filename)) for filename in sorted(os.listdir(dir_path)) if filename.endswith(".vtk")]
    
    return vtk_list

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
    Saves a numpy array to a file in .nii.gz format.

    Parameters
    ----------
    array : numpy.ndarray
        Numpy array to be saved.
    path : str
        Path to the file where the numpy array will be saved.

    """
    nib.save(nifti, path)

def load_nifti(path):
    """
    Loads a numpy array from a file in .nii.gz format.

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