#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from utils import split_segmentation, numpy_array_to_vtk_image_data, add_affine_information, resample_vtk_image_data, extract_segmentation_surface
# from arterial.centerline_extraction.preprocessing.utils import split_segmentation, numpy_array_to_vtk_image_data, add_affine_information, resample_vtk_image_data, extract_segmentation_surface

def compute_segmentation_model(segmentation_array, segmentation_affine, reduction_factor=0.5, **surface_extraction_parameters):
    print("    Converting island to VTK ImageData...")
    vtk_image_data = numpy_array_to_vtk_image_data(segmentation_array)
    print("    Adding affine information to VTK ImageData...")
    vtk_image_data = add_affine_information(vtk_image_data, segmentation_affine)
    print("    Resampling VTK ImageData...")
    vtk_image_data = resample_vtk_image_data(vtk_image_data, reduction_factor)
    print("    Extracting segmentation surface...")
    segmentation_model = extract_segmentation_surface(vtk_image_data,  **surface_extraction_parameters)

    return segmentation_model

def preprocess_segmentation_for_centerline_extraction(segmentation_array, segmentation_affine, reduction_factor=0.5, minimum_island_voxel_size=8000, **surface_extraction_parameters):
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
    print("Processing complete array...")
    segmentation_model = compute_segmentation_model(segmentation_array_, segmentation_affine, reduction_factor, **surface_extraction_parameters)

    print("Splitting segmentation array into islands...")
    segmentation_array_list = split_segmentation(segmentation_array, segmentation_affine, minimum_island_voxel_size=minimum_island_voxel_size)
    print("Number of islands found:", len(segmentation_array_list))
    segmentation_model_list = []
    for idx, segmentation_array_ in enumerate(segmentation_array_list):
        print(f"Processing island {idx + 1}/{len(segmentation_array_list)}...")
        segmentation_model_list.append(compute_segmentation_model(segmentation_array_, segmentation_affine, reduction_factor, **surface_extraction_parameters))

    return segmentation_model, segmentation_model_list