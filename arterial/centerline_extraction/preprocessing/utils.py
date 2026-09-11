#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import vtk

import numpy as np

from skimage import measure

def get_bounding_box_limits_3d(img):
    """
    Computes bounding box (only z axis) of a numpy array (expects an array 
    with zeros as background).

    Parameters
    ----------
    img : numpy.array or array-like object
        3D numpy binary (0, 1) array.

    Returns
    -------
    min_lr : integer
        Lower bound on axis x, LR (in voxel coordinates).
    max_lr : integer
        Upper bound on axis x, LR (in voxel coordinates).
    min_pa : integer
        Lower bound on axis y, PA (in voxel coordinates).
    max_pa : integer
        Upper bound on axis y, PA (in voxel coordinates).
    min_is : integer
        Lower bound on axis z, IS (in voxel coordinates).
    max_is : integer
        Upper bound on axis z, IS (in voxel coordinates).

    """
    axis_left_right = np.any(img, axis=(1, 2))
    axis_posterior_anterior = np.any(img, axis=(0, 2))
    axis_inferior_superior = np.any(img, axis=(0, 1))

    min_lr, max_lr = np.where(axis_left_right)[0][[0, -1]]
    min_pa, max_pa = np.where(axis_posterior_anterior)[0][[0, -1]]
    min_is, max_is = np.where(axis_inferior_superior)[0][[0, -1]]

    return min_lr, max_lr, min_pa, max_pa, min_is, max_is

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

    # Sort values by counts
    values = values[np.argsort(counts)[::-1]]
    counts = counts[np.argsort(counts)[::-1]]

    # Compute minimum island voxel size, taking into account reference voxel size
    # voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    # Get voxel size from image
    voxel_size = np.abs(np.prod([segmentation_affine[idx, idx] for idx in range(3)]))
    # Compute approximate number of voxels
    minimum_island_voxel_size_ = round(minimum_island_voxel_size * (reference_voxel_size / voxel_size))

    segmentation_array_list = []
    for idx, value in enumerate(values):
        if value == 0: # Skip background
            continue
        if counts[idx] >= minimum_island_voxel_size_:
            segmentation_array_list.append(np.where(label_mask == value, 1., 0.))

    return segmentation_array_list

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
    if affine is None:
        raise ValueError("Affine matrix is required")
    # Ensure the affine matrix is 4x4
    if affine.shape != (4, 4):
        raise ValueError("Affine matrix must be 4x4")
    # Extract the origin (translation vector)
    origin = affine[:3, 3]
    # Extract the direction matrix (rotation + shear)
    direction_matrix = affine[:3, :3]
    # Calculate spacing from the direction matrix
    spacing = np.linalg.norm(direction_matrix, axis=0)
    # Normalize the direction matrix
    direction_matrix = direction_matrix / spacing
    # Set the origin
    vtk_image_data.SetOrigin(*origin)
    # Set the spacing
    vtk_image_data.SetSpacing(*spacing)
    # Set the direction matrix
    vtk_image_data.SetDirectionMatrix(
        direction_matrix[0, 0], direction_matrix[0, 1], direction_matrix[0, 2],
        direction_matrix[1, 0], direction_matrix[1, 1], direction_matrix[1, 2],
        direction_matrix[2, 0], direction_matrix[2, 1], direction_matrix[2, 2]
    )
    return vtk_image_data

def cut_bottom_region(polydata, bottom_height_mm=5.0):
    """
    Cuts the bottom region of a mesh by making a perpendicular cut.

    Parameters
    ----------
    polydata : vtkPolyData
        The input mesh.
    bottom_height_mm : float, optional
        Height of the bottom region to remove in mm. The default is 5.0 mm.

    Returns
    -------
    cut_bottom_region_input_port : vtkOutputPort
        The input port of the cut bottom region filter.

    """
    bounds = polydata.GetBounds()
    z_min = bounds[4]
    
    # Define cutting plane at z_min + bottom_height_mm
    cut_position = z_min + bottom_height_mm
    
    print(f"    Cutting bottom {bottom_height_mm} mm from mesh...")
    
    # Create a plane perpendicular to z-axis
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, cut_position)
    plane.SetNormal(0.0, 0.0, 1.0)
    
    # Clip the mesh using the plane (keeps everything above the plane)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(plane)
    clipper.Update()
    
    return clipper.GetOutput()

def extract_surface(vtk_image_data, reduction_factor=0.8, apply_bottom_cutting=False, bottom_height_mm=5.0, **surface_extraction_parameters):
    """
    Extract the surface vtkPolyData from a vtkImageData object.
    First caps the holes in the surface, then smooths the surface, 
    and finally computes the normals.

    Parameters
    ----------
    vtk_image_data : vtkImageData
        VTK image data object to extract surface from.
    reduction_factor : float, optional
        The factor by which to reduce the resolution of the surface. The default is 0.8.
    apply_bottom_cutting : bool, optional
        Whether to apply aggressive smoothing to the bottom region. The default is False.
    bottom_height_mm : float, optional
        Height of the bottom region to remove in mm. The default is 5.0 mm.
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
    n_iteration_smoothing = surface_extraction_parameters.get('n_iteration_smoothing', 60)
    feature_angle = surface_extraction_parameters.get('feature_angle', 180.)
    pass_band = surface_extraction_parameters.get('pass_band', 0.05)
    extract_largest_only = surface_extraction_parameters.get('extract_largest_only', True)

    # Extract surface using the marching cubes algorithm
    print("    Applying marching cubes...")
    surface_extractor = vtk.vtkMarchingCubes()
    surface_extractor.SetInputData(vtk_image_data)
    surface_extractor.SetValue(0, 0.5) 
    surface_extractor.Update()

    # Ensure that the mesh is formed by triangular cells
    print("    Triangulating surface mesh...")
    surface_triangulator = vtk.vtkTriangleFilter()
    surface_triangulator.SetInputConnection(surface_extractor.GetOutputPort())
    surface_triangulator.PassLinesOff()
    surface_triangulator.PassVertsOff()
    surface_triangulator.Update()

    # Fill holes of the mesh
    print("    Filling mesh holes...")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputConnection(surface_triangulator.GetOutputPort())
    fill_holes.SetHoleSize(1000.0)  # Large enough to cover the expected hole size
    fill_holes.Update()

    # Smooth the extracted surface
    print("    Applying smoothing...")
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(fill_holes.GetOutputPort())
    smoother.SetNumberOfIterations(n_iteration_smoothing)  # Adjust based on desired smoothness
    smoother.SetBoundarySmoothing(1)
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)  # Lower is smoother. It's effect depends on the resolution of the surface
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    # Compute normal components for all mesh triangles
    print("    Computing normals...")
    normals_generator = vtk.vtkPolyDataNormals()
    normals_generator.SetInputConnection(smoother.GetOutputPort())
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

    # Apply a connectivity filter to remove disconnected parts 
    print("    Applying connectivity filter...")
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputConnection(decimate.GetOutputPort())
    if extract_largest_only:
        connectivity_filter.SetExtractionModeToLargestRegion()
    else:
        connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.Update()

    # Clean the decimated surface
    print("    Cleaning mesh...")
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(connectivity_filter.GetOutputPort())
    cleaner.Update()

    # Cut the bottom region if requested
    if apply_bottom_cutting:
        cut_bottom_region_output = cut_bottom_region(cleaner.GetOutput(), bottom_height_mm=bottom_height_mm)
        # Close the opening from the cut
        fill_holes = vtk.vtkFillHolesFilter()
        fill_holes.SetInputData(cut_bottom_region_output)
        fill_holes.SetHoleSize(1000.0)
        fill_holes.Update()
        
        return fill_holes.GetOutput()
    else:
        return cleaner.GetOutput()
