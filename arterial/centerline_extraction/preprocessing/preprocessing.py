#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk

import numpy as np

from arterial.centerline_extraction.preprocessing.utils import split_segmentation, extract_surface, numpy_array_to_vtk_image_data, add_affine_information

def compute_segmentation_model(segmentation_array, segmentation_affine, reduction_factor=0.8, **surface_extraction_parameters):
    """
    Computes the segmentation surface of a binary array. The binary array is first converted to
    a VTK ImageData object. The VTK ImageData object is then resampled to reduce its resolution,
    and the segmentation surface is extracted using the Marching Cubes algorithm.

    The array is processed in chunks to avoid memory issues and crashes.

    Parameters
    ----------
    segmentation_array : numpy.array
        Binary array to be segmented.
    segmentation_affine : numpy.array
        Affine transformation of the binary array.
    reduction_factor : float, optional
        The factor by which to reduce the resolution of the segmentation surfaces. The default is 0.5.
    surface_extraction_parameters : dict
        Parameters to be passed to the extract_segmentation_surface function.

    Returns
    -------
    segmentation_model : vtkPolyData
        Segmentation surface.

    """
    # Create VTK image
    vtk_image = numpy_array_to_vtk_image_data(segmentation_array, segmentation_affine)
    # Add affine information
    vtk_image = add_affine_information(vtk_image, segmentation_affine)

    print("Extracting segmentation surface...")
    segmentation_model = extract_surface(vtk_image, reduction_factor=reduction_factor, **surface_extraction_parameters)

    return segmentation_model

def preprocess_segmentation_for_centerline_extraction(segmentation_array, segmentation_affine, reduction_factor=0.8, minimum_island_voxel_size=8000, **surface_extraction_parameters):
    """
    Computes the segmentation surfaces of a binary array. The binary array is first split into
    different islands, and then each island is converted to a VTK ImageData object. The VTK ImageData
    object is then resampled to reduce its resolution, and the segmentation surfaces are extracted
    using the Marching Cubes algorithm.

    Parameters
    ----------
    segmentation_array : numpy.array
        Binary array to be segmented.
    segmentation_affine : numpy.array
        Affine transformation of the binary array.
    reduction_factor : float, optional
        The factor by which to reduce the resolution of the segmentation surfaces. The default is 0.5.
    surface_extraction_parameters : dict
        Parameters to be passed to the extract_segmentation_surface function.

    Returns
    -------
    segmentation_surfaces : list
        List of segmentation surfaces.
    
    """
    segmentation_array = np.transpose(segmentation_array, (2, 1, 0))

    print("Processing complete array...")
    segmentation_model = compute_segmentation_model(segmentation_array, segmentation_affine, reduction_factor, **surface_extraction_parameters)

    print("\nSplitting segmentation array into islands...")
    segmentation_array_list = split_segmentation(segmentation_array, segmentation_affine, minimum_island_voxel_size=minimum_island_voxel_size)
    print("Number of islands found:", len(segmentation_array_list))
    segmentation_model_list = []
    for idx, segmentation_array_ in enumerate(segmentation_array_list):
        print(f"Processing island {idx + 1}/{len(segmentation_array_list)}...")
        segmentation_model_list.append(compute_segmentation_model(segmentation_array_, segmentation_affine, reduction_factor, **surface_extraction_parameters))

    return segmentation_model, segmentation_model_list