#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk

import numpy as np

from skimage import measure

# Split segmentation_array in a list of arrays with the different islands
def split_segmentation(segmentation_array, segmentation_affine, minimum_island_voxel_size):
    """
    Splits binary array into a list of binary arrays with the different islands.

    Parameters
    ----------
    segmentation_array : numpy.array
        Binary array to be split into islands.

    Returns
    -------
    segmentation_array_list : list
        List of binary arrays with the different islands.
    
    """
    label_mask = measure.label(segmentation_array)
    values, counts = np.unique(label_mask, return_counts=True)

    # Compute minimum island voxel size, taking into account reference voxel size
    # voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    # Get voxel size from image
    voxel_size = np.abs(np.prod([segmentation_affine[idx, idx] for idx in range(3)]))
    # Compute approximate number of voxels
    minimum_island_voxel_size_ = round(minimum_island_voxel_size * (reference_voxel_size / voxel_size))

    segmentation_array_list = []
    for idx in range(len(values)):
        if values[idx] == 0:
            continue
        if counts[idx] >= minimum_island_voxel_size_:
            segmentation_array_list.append(np.where(label_mask == values[idx], 1., 0.))

    return segmentation_array_list

# def get_bounding_box_and_adjust_affine(segmentation_array, segmentation_affine):
#     """
#     Get the bounding box of the segmentation array and adjust the affine transformation accordingly.

#     Parameters
#     ----------
#     segmentation_array : numpy.array
#         Binary array to be segmented.
#     segmentation_affine : numpy.array
#         Affine transformation of the binary array.

#     Returns
#     -------
#     bounding_box : numpy.array
#         Bounding box of the segmentation array.
#     adjusted_affine : numpy.array
#         Adjusted affine transformation.

#     """
#     bounding_box_args = np.argwhere(segmentation_array)
#     min_z, min_y, min_x = bounding_box_args.min(axis=0)
#     max_z, max_y, max_x = bounding_box_args.max(axis=0)

#     adjusted_affine = np.copy(segmentation_affine)
#     print("IJK origin displacement:", min_x, min_y, min_z)
#     print("RAS origin displacement:", segmentation_affine[:3, :3] @ [min_x, min_y, min_z])
#     adjusted_affine[:3, 3] = segmentation_affine[:3, 3] + segmentation_affine[:3, :3] @ [min_x, min_y, min_z]

#     return np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]]), adjusted_affine

def numpy_array_to_vtk_image_data(numpy_array):
    """
    Convert a numpy array to a VTK ImageData object.

    Parameters
    ----------
    numpy_array : numpy.array
        The numpy array to be converted.

    Returns
    -------
    vtkImageData : vtkImageData
        The converted vtkImageData.
        

    """
    # Convert a numpy array to a VTK ImageData object
    importer = vtk.vtkImageImport()
    importer.SetDataScalarTypeToDouble()
    importer.SetNumberOfScalarComponents(1)
    importer.SetDataExtent(0, numpy_array.shape[2] - 1, 0, numpy_array.shape[1] - 1, 0, numpy_array.shape[0] - 1)
    importer.SetWholeExtent(0, numpy_array.shape[2] - 1, 0, numpy_array.shape[1] - 1, 0, numpy_array.shape[0] - 1)
    importer.SetImportVoidPointer(numpy_array)
    importer.Update()
    return importer.GetOutput()

def add_affine_information(vtk_image_data, affine=None):
    """
    Add affine information to a vtkImageData object.
    
    Parameters
    ----------
    vtk_image_data : vtkImageData
        The vtkImageData object to which to add the affine information.
    affine : numpy.array
        The affine information to be added.

    Returns
    -------
    vtkImageData : vtkImageData
        The vtkImageData object with the added affine information.
        
    """
    vtk_image_data.SetOrigin(affine[:3, 3])
    vtk_image_data.SetSpacing(affine[0, 0], affine[1, 1], affine[2, 2])
    vtk_image_data.SetDirectionMatrix(1., 0., 0., 0., 1., 0., 0., 0., 1.)
    return vtk_image_data

# def resample_vtk_image_data(vtk_image_data, reduction_factor=0.5):
#     """
#     Resample the vtkImageData to reduce its resolution. Otherwise we would find that
    
#     for images larger than approximately [512, 512, 600] the process crashes.

#     Parameters
#     ----------
#     vtk_image_data : vtkImageData
#         The original image data.
#     reduction_factor : float
#         The factor to which to reduce the resolution. For example, 0.5 will reduce the 
#         number of points to about half along each dimension, while 0.8 will reduce the
#         number of points to about 80% along each dimension.

#     Returns
#     -------
#     vtkImageData : vtkImageData
#         The resampled image data.

#     """
#     resample = vtk.vtkImageResample()
#     resample.SetInputData(vtk_image_data)
#     resample.SetAxisMagnificationFactor(0, reduction_factor)
#     resample.SetAxisMagnificationFactor(1, reduction_factor)
#     resample.SetAxisMagnificationFactor(2, reduction_factor)
#     resample.SetInterpolationMode(vtk.VTK_RESLICE_NEAREST)
#     resample.Update()
#     return resample.GetOutput()

def extract_surface(vtk_image_data, reduction_factor=0.8, **surface_extraction_parameters):
    """
    Extract the surface vtkPolyData from a vtkImageData object.
    First caps the holes in the surface, then smooths the surface, 
    and finally computes the normals.

    Parameters
    ----------
    vtk_image_data_reader : vtkNIFTIImageReader
        Nifti image reader class. Assumes loaded image.
    n_iteration_smoothing : int
        The number of iterations for the vtkWindowedSincPolyDataFilter 
        smoothing algorithm.
    feature_angle : float
        The feature angle for the smoothing algorithm.
    pass_band : float
        The pass band for the smoothing algorithm.

    Returns
    -------
    segmentation_model : vtkPolyData
        The extracted surface.

    """
    n_iteration_smoothing = surface_extraction_parameters.get('n_iteration_smoothing', 50)
    feature_angle = surface_extraction_parameters.get('feature_angle', 120.)
    pass_band = surface_extraction_parameters.get('pass_band', 0.1)

    # Extract surface using the marching cubes algorithm
    print("    Applying marching cubes...")
    surface_extractor = vtk.vtkDiscreteMarchingCubes()
    surface_extractor.SetInputData(vtk_image_data)
    surface_extractor.SetValue(0, 1) 
    surface_extractor.Update()

    # Ensure that the mesh is formed by triangular cells
    print("    Triangulating surface mesh...")
    surface_triangulator = vtk.vtkTriangleFilter()
    surface_triangulator.SetInputConnection(surface_extractor.GetOutputPort())
    surface_triangulator.PassLinesOff()
    surface_triangulator.PassVertsOff()
    surface_triangulator.Update()

    # Smooth the extracted surface
    print("    Applying smoothing...")
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surface_triangulator.GetOutputPort())
    smoother.SetNumberOfIterations(n_iteration_smoothing)  # Adjust based on desired smoothness
    smoother.SetBoundarySmoothing(1)
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)  # Lower is smoother. It's effect depends on the resolution of the surface
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    # Fill holes of the mesh
    print("    Filling mesh holes...")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputConnection(smoother.GetOutputPort())
    fill_holes.SetHoleSize(1000.0)  # Large enough to cover the expected hole size
    fill_holes.Update()

    # COmpute normal components for all mesh triangles
    print("    Computing normals...")
    normals_generator = vtk.vtkPolyDataNormals()
    normals_generator.SetInputConnection(fill_holes.GetOutputPort())
    normals_generator.SetAutoOrientNormals(1)
    normals_generator.SetFlipNormals(0)
    normals_generator.SetConsistency(1)
    normals_generator.SplittingOff()
    normals_generator.Update()

    # Decimate surface model
    decimate = vtk.vtkDecimatePro()
    print(f"    Decimating mesh (target reduction: {reduction_factor})...")
    decimate.SetInputConnection(normals_generator.GetOutputPort())
    decimate.SetTargetReduction(reduction_factor)
    decimate.PreserveTopologyOn()
    decimate.BoundaryVertexDeletionOn()
    decimate.Update()

    # Clean the decimated surface
    print("    Cleaning mesh...")
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(decimate.GetOutputPort())
    cleaner.Update()
    
    return cleaner.GetOutput()