#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import numpy as np

from arterial.centerline_extraction.preprocessing.utils import get_bounding_box_limits_3d, numpy_array_to_vtk_image_data, add_affine_information, split_segmentation, extract_surface

def compute_segmentation_model(segmentation_array, segmentation_affine, reduction_factor, apply_bottom_cutting=False, bottom_height_mm=5.0, **surface_extraction_parameters):
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
    apply_bottom_cutting : bool, optional
        Whether to cut the bottom region. The default is False.
    bottom_height_mm : float, optional
        Height of the bottom region to remove in mm. The default is 5.0 mm.
    surface_extraction_parameters : dict
        Parameters to be passed to the extract_segmentation_surface function.

    Returns
    -------
    segmentation_model : vtkPolyData
        Segmentation surface.

    """
    # Create VTK image
    vtk_image = numpy_array_to_vtk_image_data(segmentation_array)
    # Add affine information
    vtk_image = add_affine_information(vtk_image, segmentation_affine)

    print("Extracting segmentation surface...")
    surface_extraction_parameters["extract_largest_only"] = False
    segmentation_model = extract_surface(vtk_image, reduction_factor=reduction_factor, apply_bottom_cutting=apply_bottom_cutting, bottom_height_mm=bottom_height_mm, **surface_extraction_parameters)

    return segmentation_model

def preprocess_segmentation_for_centerline_extraction(segmentation_array, segmentation_affine, fast_segmentation=False, minimum_island_voxel_size=8000, apply_bottom_cutting_to_first_model=False, bottom_height_mm=5.0, **surface_extraction_parameters):
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
    fast_segmentation : bool, optional
        Flag to indicate if fast processing pipeline should be chosen. All voxels in the
        upper 10% of the segmentation's bounding box will be set to 0. This reduces the burden for the 
        branch model extraction and only affects centerlines in the intracranial region. This can be desired if
        the analysis is focused on the extracranial region. The default is False.
    minimum_island_voxel_size : int, optional
        Minimum number of voxels for an island to be considered. The default is 8000 in native resolution (found empirically).
    apply_bottom_cutting_to_first_model : bool, optional
        Whether to apply bottom cutting to the first model. The default is False.
    bottom_height_mm : float, optional
        Height of the bottom region to remove in mm. The default is 5.0 mm.
    surface_extraction_parameters : dict
        Parameters to be passed to the extract_segmentation_surface function.

    Returns
    -------
    segmentation_surfaces : list
        List of segmentation surfaces.
    
    """
    segmentation_array = np.transpose(segmentation_array, (2, 1, 0))

    print("Splitting segmentation array into islands...")
    segmentation_array_list = split_segmentation(segmentation_array, segmentation_affine, minimum_island_voxel_size=minimum_island_voxel_size)

    # We generate a "clean" segmentation array, with small islands removed
    clean_segmentation_array = np.zeros_like(segmentation_array)
    for segmentation_array_ in segmentation_array_list:
        clean_segmentation_array += segmentation_array_

    print("\nProcessing complete array...")
    segmentation_model = compute_segmentation_model(clean_segmentation_array, segmentation_affine, reduction_factor=0.8, apply_bottom_smoothing=False, **surface_extraction_parameters)

    # For fast segmentation processing, we remove the upper 10% of the segmentation bonding box voxels. This typically simplifies endpoint extraction 
    # and centerline tracing 
    if fast_segmentation:
        _, max_is, _, _, _, _ = get_bounding_box_limits_3d(segmentation_array)
        for idx, segmentation_array_ in enumerate(segmentation_array_list):
            segmentation_array_list[idx][int(np.round(max_is * 0.9)):, :, :] = 0
    
    print("Number of islands found:", len(segmentation_array_list))
    segmentation_model_list = []
    for idx, segmentation_array_ in enumerate(segmentation_array_list):
        print(f"Processing island {idx + 1}/{len(segmentation_array_list)}...")
        surface_extraction_parameters["pass_band"] = 0.1
        if idx == 0 and apply_bottom_cutting_to_first_model:
            apply_bottom_cutting = True
        else:
            apply_bottom_cutting = False
        segmentation_model_list.append(compute_segmentation_model(segmentation_array_, segmentation_affine, reduction_factor=0.9, apply_bottom_cutting=apply_bottom_cutting, bottom_height_mm=bottom_height_mm, **surface_extraction_parameters))

    return segmentation_model, segmentation_model_list