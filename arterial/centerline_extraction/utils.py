#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk
import vmtk
import random

import numpy as np
import nibabel as nib

from skimage import measure
from scipy import ndimage

from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

def get_bounding_box_limits_3d(array):
    """
    Computes bounding box (only z axis) of a numpy array (expects an array 
    with zeros as background).

    Parameters
    ----------
    array : numpy.array or array-like object
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
    axis_left_right = np.any(array, axis=(0, 1))
    axis_posterior_anterior = np.any(array, axis=(0, 2))
    axis_inferior_superior = np.any(array, axis=(1, 2))

    min_lr, max_lr = np.where(axis_left_right)[0][[0, -1]]
    min_pa, max_pa = np.where(axis_posterior_anterior)[0][[0, -1]]
    min_is, max_is = np.where(axis_inferior_superior)[0][[0, -1]]

    return min_lr, max_lr, min_pa, max_pa, min_is, max_is

def volume_sanity_check(segmentation_array, segmentation_affine):
    """
    Check that the volume of the segmentation is within the expected range.
    Otherwise, image will be read as an artifact and an error will be raised.

    Parameters
    ----------
    segmentation_array : numpy.array or array-like object
        3D numpy binary (0, 1) array.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------

    """
    # Compute volume of the bouding box taking into account voxel size
    min_lr, max_lr, min_pa, max_pa, min_is, max_is = get_bounding_box_limits_3d(segmentation_array)
    # Compute segmentation volume taking into account voxel size
    voxel_size = np.abs(np.prod([segmentation_affine[idx, idx] for idx in range(3)]))
    segmentation_volume = np.sum(segmentation_array > 0) * voxel_size
    bouding_box_volume = (max_lr - min_lr) * (max_pa - min_pa) * (max_is - min_is) * voxel_size
    if segmentation_volume < 4.5e4: # Empirically tested
        raise ValueError("Segmentation volume is too small: {:.2f} mm3".format(segmentation_volume))
    if bouding_box_volume < 3.5e6: # Empirically tested
        raise ValueError("Bounding box volume is too small: {:.2f} mm3".format(bouding_box_volume))
    if bouding_box_volume > 3.5e7: # Empirically tested
        raise ValueError("Bounding box volume is too large: {:.2f} mm3".format(bouding_box_volume))
    if segmentation_volume < 5e4 and bouding_box_volume < 6e6: # Empirically tested
        raise ValueError("Combination of segmentation volume and bounding box volume is too small: \nSegmentation volume: {:.2f} mm3 \nBounding box volume: {:.2f}".format(segmentation_volume, bouding_box_volume))

def _compute_centerlines_network(surface_address, delaunay_address, voronoi_address, pole_ids_address, cell, points):
    """
    Compute the centerline of a single branch of the network.

    Adapted from https://github.com/vmtk/vmtk/blob/6211af00372d099454acaf5d90520ddfcdf6cf96/vmtkScripts/vmtkcenterlinesnetwork.py#L29.

    Parameters
    ----------
    surface_address : str
        Memory address of the vtkPolyData object representing the surface.
    delaunay_address : str
        Memory address of the vtkUnstructuredGrid object representing the delaunay tessellation.
    voronoi_address : str
        Memory address of the vtkPolyData object representing the voronoi diagram.
    pole_ids_address : str
        Memory address of the vtkIdList object representing the pole ids.
    cell : list
        List of integers representing the cell connectivity.
    points : numpy.array
        Array of points.

    Returns
    -------
    clConvert.ArrayDict : dict
        Dictionary containing the centerline data.
    
    """

    surface = vtk.vtkPolyData(surface_address)
    delaunay = vtk.vtkUnstructuredGrid(delaunay_address)
    voronoi = vtk.vtkPolyData(voronoi_address)
    pole_ids = vtk.vtkIdList(pole_ids_address)

    cl = _compute_centerline_branch(surface, delaunay, voronoi, pole_ids, cell, points)

    clConvert = vmtk.vmtkcenterlinestonumpy.vmtkCenterlinesToNumpy()
    clConvert.Centerlines = cl
    clConvert.LogOn = 0
    clConvert.Execute()
    
    return clConvert.ArrayDict

def _compute_centerline_branch(surface, delaunay, voronoi, pole_ids, cell, points):
    """
    Compute the centerline of a single branch of the network.
    
    Adapted from https://github.com/vmtk/vmtk/blob/6211af00372d099454acaf5d90520ddfcdf6cf96/vmtkScripts/vmtkcenterlinesnetwork.py#L55.

    Parameters
    ----------
    surface : vtkPolyData
        The surface model.
    delaunay : vtkUnstructuredGrid
        The delaunay tessellation.
    voronoi : vtkPolyData
        The voronoi diagram.
    pole_ids : vtkIdList
        The pole ids.
    cell : list
        List of integers representing the cell connectivity.
    points : numpy.array
        Array of points.

    Returns
    -------
    cl.Centerlines : vtkPolyData
        The centerline of the branch.

    """
    cell_startidx = cell[0]
    cell_end_idx = cell[-1]
    cell_startpoint = points[cell_startidx].tolist()
    cell_end_point = points[cell_end_idx].tolist()
    cl = vmtk.vmtkcenterlines.vmtkCenterlines()
    cl.Surface = surface
    cl.DelaunayTessellation = delaunay
    cl.VoronoiDiagram = voronoi
    cl.pole_ids = pole_ids
    cl.SeedSelectorName = 'pointlist'
    # since we only set one target seed at a time, setting StopFastMarchingOnReachingTarget
    # greatly speeds up algorithm execution time.
    cl.StopFastMarchingOnReachingTarget = 1
    cl.SourcePoints = cell_startpoint
    cl.TargetPoints = cell_end_point
    cl.LogOn = 0
    cl.Execute()

    return cl.Centerlines

def compute_network_centerlines(segmentation_model):
    """
    Compute the centerlines of a network of approximated centerlines.

    Adapted from https://github.com/vmtk/vmtk/blob/6211af00372d099454acaf5d90520ddfcdf6cf96/vmtkScripts/vmtkcenterlinesnetwork.py#L129.

    Parameters
    ----------
    segmentation_model : vtkPolyData
        The surface model of the vascular segmentation.

    Returns
    -------
    network_centerlines : vtkPolyData
        The network centerlines.

    """
    # feature edges are used to find any holes in the surface.
    fedges = vtk.vtkFeatureEdges()
    fedges.BoundaryEdgesOn()
    fedges.FeatureEdgesOff()
    fedges.ManifoldEdgesOff()
    fedges.SetInputData(segmentation_model)
    fedges.Update()
    ofedges = fedges.GetOutput()

    # if num_edges is not 0, then there are holes which need to be capped
    num_edges = ofedges.GetNumberOfPoints()
    if num_edges != 0:
        tempcapper = vmtk.vmtksurfacecapper.vmtkSurfaceCapper()
        tempcapper.Surface = segmentation_model
        tempcapper.Interactive = 0
        tempcapper.Execute()

        network_surface = tempcapper.Surface
    else:
        network_surface = segmentation_model

    # randomly select one cell to delete so that there is an opening for
    # vmtkNetworkExtraction to use.
    num_cells = network_surface.GetNumberOfCells()
    random_generator = random.Random()
    random_generator.seed(42)
    cell_to_delete = random_generator.randrange(0, num_cells-1)
    network_surface.BuildLinks()
    network_surface.DeleteCell(cell_to_delete)
    network_surface.RemoveDeletedCells()

    # extract the network of approximated centerlines
    net = vmtk.vmtknetworkextraction.vmtkNetworkExtraction()
    net.Surface = network_surface
    net.AdvancementRatio = 1.001
    net.Execute()
    network = net.Network

    convert = vmtk.vmtkcenterlinestonumpy.vmtkCenterlinesToNumpy()
    convert.Centerlines = network
    convert.LogOn = False
    convert.Execute()
    ad = convert.ArrayDict
    cell_data_topology = ad['CellData']['Topology']

    # the network topology identifies an the input segment with the "0" id.
    # since we artificially created this segment, we don't want to use the
    # ends of the segment as source/target points of the centerline calculation
    try:
        node_index_to_ignore = np.where(cell_data_topology[:,0] == 0)[0][0]
    except:
        node_index_to_ignore = None
    keep_cell_connectivity_list = []
    point_idx_to_keep = np.array([])
    remove_cell_length = 0
    # we remove the cell, points, and point data which are associated with the
    # segment we want to ignore
    for loop_idx, cell_connectivity_list in enumerate(ad['CellData']['CellPointIds']):
        if loop_idx == node_index_to_ignore:
            remove_cell_startidx = cell_connectivity_list[0]
            remove_cell_end_idx = cell_connectivity_list[-1]
            remove_cell_length = cell_connectivity_list.size
            if (remove_cell_end_idx + 1) - remove_cell_startidx != remove_cell_length:
                raise(ValueError)
            continue
        else:
            rescaled_cell_connectivity = np.subtract(cell_connectivity_list, remove_cell_length, where=cell_connectivity_list >= remove_cell_length)
            keep_cell_connectivity_list.append(rescaled_cell_connectivity)
            point_idx_to_keep = np.concatenate((point_idx_to_keep, cell_connectivity_list)).astype(int)
    new_points = ad['Points'][point_idx_to_keep]

    # precompute the delaunay tessellation for the whole surface.
    tessalation = vmtk.vmtkdelaunayvoronoi.vmtkDelaunayVoronoi()
    tessalation.Surface = network_surface
    tessalation.Execute()

    out = []
    import sys
    if (sys.platform == 'win32') or (sys.platform == 'win64') or (sys.platform == 'cygwin') or (sys.platform == 'darwin'):
        use_joblib = False
    else:
        use_joblib = True
    if use_joblib:
        # vtk objects cannot be serialized in python. Instead of converting the inputs to numpy arrays and having
        # to reconstruct the vtk object each time the loop executes (a slow process), we can just pass in the
        # memory address of the data objects as a string, and use the vtk python bindings to create a python name
        # referring to the data residing at that memory address. This works because joblib executes each loop
        # iteration in a fork of the original process, providing access to the original memory space.
        # However, the process does not work for return arguments, since the original process will not have access to
        # the memory space of the fork. To return results we use the vmtkCenterlinesToNumpy converter.
        network_surface_memory_address = network_surface.__this__
        delaunay_memory_address = tessalation.DelaunayTessellation.__this__
        voronoi_memory_address = tessalation.VoronoiDiagram.__this__
        pole_ids_memory_address = tessalation.pole_ids.__this__
        num_parallel_jobs = -1
        
        # note about the verbose function: while Joblib can print a progress bar output (set verbose = 20),
        # it does not implement a callback function as of version 0.11, so we cannot report progress to the user
        # if we are redirecting standard out with the self.PrintLog method.
        outlist = Parallel(n_jobs=num_parallel_jobs, backend='multiprocessing', verbose=0)(
            delayed(_compute_centerlines_network)(network_surface_memory_address,
                                            delaunay_memory_address,
                                            voronoi_memory_address,
                                            pole_ids_memory_address,
                                            cell,
                                            new_points) for cell in keep_cell_connectivity_list)
        for item in outlist:
            np_convert = vmtk.vmtknumpytocenterlines.vmtkNumpyToCenterlines()
            np_convert.ArrayDict = item
            np_convert.LogOn = 0
            np_convert.Execute()
            out.append(np_convert.Centerlines)
    else:
        for cell in keep_cell_connectivity_list:
                cl = _compute_centerline_branch(network_surface, tessalation.DelaunayTessellation, tessalation.VoronoiDiagram,
                                                tessalation.pole_ids, cell, new_points)
                out.append(cl)
                

    # Append each segment's polydata into a single polydata object
    centerline_appender = vtk.vtkAppendPolyData()
    for data in out:
        centerline_appender.AddInputData(data)
    centerline_appender.Update()

    # clean and strip the output centerlines so that redundant points are merged and tracts are combined
    centerline_cleaner = vtk.vtkCleanPolyData()
    centerline_cleaner.SetInputData(centerline_appender.GetOutput())
    centerline_cleaner.Update()

    centerline_stripper = vtk.vtkStripper()
    centerline_stripper.SetInputData(centerline_cleaner.GetOutput())
    centerline_stripper.JoinContiguousSegmentsOn()
    centerline_stripper.Update()

    network_centerlines = centerline_stripper.GetOutput()

    return network_centerlines

def get_endpoints(network_centerlines, startpoint_position):
    """ 
    Adapted from https://github.com/vmtk/SlicerExtension-VMTK/blob/3787ea4a300da28ec5f0824f0715f2713b631155/ExtractCenterline/ExtractCenterline.py#L746
    Clips the surfacePolyData on the endpoints identified using the networkPolyData.
    If startpoint_position is specified then start point will be the closest point to that position.
    Returns list of endpoint positions. Largest radius point is be the first in the list.

    Parameters
    ----------
    network_centerlines : vtkPolyData
        The network poly data.
    startpoint_position : numpy.array
        The start point position.

    Returns
    -------
    endpoint_positions : list
        List of endpoint positions.

    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(network_centerlines)
    cleaner.Update()
    network = cleaner.GetOutput()
    network.BuildCells()
    network.BuildLinks(0)

    network_points = network.GetPoints()
    radius_array = network.GetPointData().GetArray("MaximumInscribedSphereRadius")

    startpoint_id = -1
    max_radius = 0
    min_distance_2 = 0

    endpoint_ids = vtk.vtkIdList()
    for idx in range(network.GetNumberOfCells()):
        number_of_cell_points = network.GetCell(idx).GetNumberOfPoints()
        if number_of_cell_points < 2:
            continue

        for point_index in [0, number_of_cell_points - 1]:
            point_id = network.GetCell(idx).GetPointId(point_index)
            point_cells = vtk.vtkIdList()
            network.GetPointCells(point_id, point_cells)
            if point_cells.GetNumberOfIds() == 1:
                endpoint_ids.InsertUniqueId(point_id)
                if startpoint_position is not None:
                    # find start point based on position
                    position = network_points.GetPoint(point_id)
                    distance_2 = vtk.vtkMath.Distance2BetweenPoints(position, startpoint_position)
                    if startpoint_id < 0 or distance_2 < min_distance_2:
                        min_distance_2 = distance_2
                        startpoint_id = point_id
                else:
                    # find start point based on radius
                    radius = radius_array.GetValue(point_id)
                    if startpoint_id < 0 or radius > max_radius:
                        max_radius = radius
                        startpoint_id = point_id

    endpoint_positions = []
    number_of_endpoint_ids = endpoint_ids.GetNumberOfIds()
    if number_of_endpoint_ids == 0:
        return endpoint_positions
    # add the largest radius point first
    endpoint_positions.append(network_points.GetPoint(startpoint_id))
    # add all the other points
    for point_id_index in range(number_of_endpoint_ids):
        point_id = endpoint_ids.GetId(point_id_index)
        if point_id == startpoint_id:
            # already added
            continue
        endpoint_positions.append(network_points.GetPoint(point_id))

    return endpoint_positions

def aortic_arch_endpoint_check(endpoints_list, segmentation_array, segmentation_affine):
    """
    Checks that both ens of the aortic arch (AA), if present, have one associated endpoint.
    To do that, it looks at the bottom slice of the volume and analyzes the presence 
    of large connected components. Once it has recognized all large connected components,
    it checks if any endpoint is at an Euclidean distance of less than 50 mm with respect to the
    center of mass of the bottom islands. 
    
    Paremeters
    ----------
    endpoints_node : vtkMRMLMarkupsFiducialNode
        MRML node with all endpoints from the automatic endpoint detection.
    segmentation_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 85% of the segmentation's bounding box.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------
    endpoints_node : vtkMRMLMarkupsFiducialNode
        Updated MRML node with all endpoints from the automatic endpoint detection.

    """
    # We define a distance factor in case we are dealing with images of a different resolution
    # We always assume we have close-to-isotropic voxels. 0.43 corresponds to the reference voxel size
    # used to empirically define certain reference values
    factor = abs(0.43 / segmentation_affine[0, 0])
    # For AA island validation (number of foreground voxels in the bottom slice)
    # Empirically, we found that 500 is a good threshold for a voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    voxel_size = np.prod([segmentation_affine[idx, idx] for idx in range(3)])
    threshold_counts = abs(round(500 * (reference_voxel_size / voxel_size)))
    # For AA endpoints check (distance from bottom slice)
    threshold_distance = 50 * 0.4 / segmentation_affine[2, 2]

    # Divide into different connected components of the bottom slice
    label_mask = measure.label(segmentation_array[0])
    properties = measure.regionprops(label_mask.astype(int), label_mask.astype(int))
    
    # Get rid of all components below the threshold_counts
    # This is done because we expect here to only have bottom slices of the
    # ascending and descending aorta. This way we get rid of any other component
    _, counts = np.unique(label_mask, return_counts=True)
    delete_idx = []
    for idx, count in enumerate(counts):
        if count < threshold_counts:
            delete_idx.append(idx - 1)
    properties = list(np.delete(properties, delete_idx))
    
    # Access and store the coordinates of centroids in RAS coordinates
    # Notice that we set the S coordinate to 1.0 for all centroids
    centroids = np.zeros(shape = (len(properties), 3))
    for idx, prop in enumerate(properties):
        centroids[idx] = np.matmul(segmentation_affine, np.append(np.array(prop.centroid)[[1, 0]], [1.0, 1.0]))[:3]

    # Compute distance from each endpoint to all centroids of components in the bottom slice
    # The goal is to check that each component (generallly there should be 2) has one endpoint
    # nearby
    for idx, endpoint in enumerate(endpoints_list):
        delete_idx = None
        for idx_centroids, centroid in enumerate(centroids):
            # If a connnected component is found close to an endpoint, we accept it as correctly placed
            if np.linalg.norm(centroid - endpoint) < threshold_distance: # Threshold at 50 mm
                delete_idx = idx_centroids
        if delete_idx is not None:
            centroids = np.delete(centroids, delete_idx, axis=0)

    # If any connected components survive, it means that no enpoints were found close by
    if len(centroids) > 0:
        print("{} AA islands do not have associated endpoints".format(len(centroids)))
        # This way, we convert the remaining centroids to endpoints
        for centroid in centroids:
            print("Adding endpoint at", centroid)
            print()
            endpoints_list.append(np.array(centroid))

    # Now all that's left is to ensure that the startpoint is placed at the descending aorta
    # (most proximal point from femoral access in endovascular interventions)
            
    # Select distal AA endpoint as startpoint (in some cases, the distal LSA endpoint is closer to the origin)
    # The criteria will be to choose the AA endpoint (at < 50 mm from bottom slice) that is closest to the reference point
    # Check every other point's distance to origin (ijk)
    distance_to_reference = []
    for idx, endpoint in enumerate(endpoints_list):
        endpoint = np.matmul(np.linalg.inv(segmentation_affine), np.append(endpoint, 1.0))[:3]
        # Reference point set at [350, 0, 0] in LAS coordinates
        if nib.orientations.aff2axcodes(segmentation_affine) == ("R", "A", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([150.0 * factor, 0.0, 0.0])))
        elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "A", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([350.0 * factor, 0.0, 0.0])))
        elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "P", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([350.0 * factor, label_mask.shape[1], 0.0])))

    # Get order from closest to furthest
    sorted_distance_idx = np.argsort(distance_to_reference)
    for idx in sorted_distance_idx:
        startpoint = endpoints_list[idx]
        if np.matmul(np.linalg.inv(segmentation_affine), np.append(startpoint, 1.0))[2] > threshold_distance:
            print("Startpoint {} found is not in the AA region".format(idx))
            pass
        else:
            # Make sure that startpoint is close to the bottom slice
            if np.matmul(np.linalg.inv(segmentation_affine), np.append(startpoint, 1.0))[2] < threshold_distance and idx == 0:
                print("Original startpoint is at distal AA")
                break
            # If it is not, set next closest endpoint to reference as startpoint if it is closer to bottom slice
            elif np.matmul(np.linalg.inv(segmentation_affine), np.append(startpoint, 1.0))[2] < threshold_distance and idx != 0:
                print("New startpoint ({}): {}".format(idx, startpoint))
                endpoints_list[idx] = endpoints_list[0]
                endpoints_list[0] = startpoint
                break
            else: 
                pass
    
    return endpoints_list

def robust_endpoint_detection(endpoint_list, segmentation_array, segmentation_affine, window_size = 15):
    """
    Relocates automatically detected endpoints to the center of mass of the closest component
    inside a local region around the endpoint (defined by n).

    Takes the endpoint position, converts it to voxel coordinates with the affine matrix, then defines a region  
    of (2 * n) ^ 3 voxels centered around the endpoint. Then components inside the local region are treated 
    as separate objects. The minimum distance from these objects to the endpoint is computed, and from 
    these, the object with the smallest distance to the endpoint is chosen to compute the centroid, which
    is converted back to RAS with the affine matrix.

    Parameters
    ----------
    endpoint : numpy.array or array-like object 
        Position of the endpoint in RAS coordinates.
    segmentation_array : numpy.array or array-like object
        Numpy array corresponding to the masked_volume_node.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.
    window_size : int 
        Defines the size of the region around the endpoint that is analyzed for this method.
        New endpoint location will be searched within a cubic box of size 2 * n around the 
        originial endpoint location.

    Returns
    -------
    new_endpoint : numpy.array or array-like object
        New position of the endpoint.

    """
    # Invert the affine matrix
    segmentation_affine_inv = np.linalg.inv(segmentation_affine)
    for endpoint_idx, endpoint in enumerate(endpoint_list):
        # Compute endpoint ijk coordinates with affine matrix
        i, j, k = np.round(np.matmul(segmentation_affine_inv, np.append(endpoint, 1.0))[:3]).astype(int)
        if segmentation_array[i, j, k] == 0:
            print("Relocating endpoint {}: {}".format(endpoint_idx, endpoint))
            # Mask the segmentation_array (only region of interest)
            masked_segmentation = segmentation_array[
                np.max([0, i - window_size]): np.min([segmentation_array.shape[0], i + window_size]), 
                np.max([0, j - window_size]): np.min([segmentation_array.shape[1], j + window_size]),
                np.max([0, k - window_size]): np.min([segmentation_array.shape[2], k + window_size])
                ]
            
            # Divide into different connected components
            label_mask = measure.label(masked_segmentation)
            # We sort label values and ignore the background
            labels = np.sort(np.unique(label_mask))
            labels = np.delete(labels, np.where([labels == 0]))

            # Pass masked groups to one-hot encoding
            label_mask_one_hot = np.zeros([len(labels), label_mask.shape[0], label_mask.shape[1], label_mask.shape[2]], dtype=np.uint8)
            for idx, label in enumerate(labels):
                label_mask_one_hot[idx][label_mask == label] = 1

            # Invert the masks
            inverted_label_mask_one_hot = np.ones_like(label_mask_one_hot) - label_mask_one_hot
            
            # Get distance transform for each and get only closest component of the inverted masks
            # Distance transforms encode the distance of all foreground voxels
            # to the closest background element
            distance_labels = np.empty_like(labels, dtype=float)
            for idx in range(len(labels)):
                distance_labels[idx] = ndimage.distance_transform_edt(inverted_label_mask_one_hot[idx])[inverted_label_mask_one_hot.shape[1] // 2][inverted_label_mask_one_hot.shape[2] // 2][inverted_label_mask_one_hot.shape[3] // 2]
            # We keep only the closest component to the original endpoint
            mask = np.zeros_like(segmentation_array)
            mask[
                np.max([0, i - window_size]): np.min([segmentation_array.shape[0], i + window_size]), 
                np.max([0, j - window_size]): np.min([segmentation_array.shape[1], j + window_size]),
                np.max([0, k - window_size]): np.min([segmentation_array.shape[2], k + window_size])
                ] = label_mask_one_hot[np.argmin(distance_labels)]
            
            # Get the centroid of the foregroud region and turn it into the new endpoint
            properties = measure.regionprops(mask.astype(int), mask.astype(int))
            center_of_mass = np.array(properties[0].centroid)

            # Return the new position of the endpoint in RAS coordinates
            endpoint_list[endpoint_idx] = np.matmul(segmentation_affine, np.append(center_of_mass, 1.0))[:3]
            print("New endpoint position: {}".format(endpoint_list[endpoint_idx]))
        else:
            print("Endpoint {} is already inside the segmentation".format(endpoint_idx))
    
    return endpoint_list

def _robust_endpoint_detection(endpoint_data):
    """
    Same as robust_endpoint_detection but adapted to be used in parallel processing.

    Parameters
    ----------
    endpoint_data : tuple
        Tuple containing the endpoint, segmentation_array, segmentation_affine, and window_size.

    Returns
    -------
    new_endpoint : numpy.array or array-like object
        New position of the endpoint.

    """
    endpoint, segmentation_array, segmentation_affine, window_size = endpoint_data
    # All the steps remain the same as in your function until the `distance_transform_edt` call
    print("Relocating endpoint: {}".format(endpoint))
    # Compute endpoint ijk coordinates with affine matrix
    i, j, k = np.round(np.matmul(np.linalg.inv(segmentation_affine), np.append(endpoint, 1.0))[:3]).astype(int)
    # Mask the segmentation_array (only region of interest)
    masked_segmentation = segmentation_array[
        np.max([0, i - window_size]): np.min([segmentation_array.shape[0], i + window_size]), 
        np.max([0, j - window_size]): np.min([segmentation_array.shape[1], j + window_size]),
        np.max([0, k - window_size]): np.min([segmentation_array.shape[2], k + window_size])
        ]
    
    # Divide into different connected components
    label_mask = measure.label(masked_segmentation)
    # We sort label values and ignore the background
    labels = np.sort(np.unique(label_mask))
    labels = np.delete(labels, np.where([labels == 0]))

    # Pass masked groups to one-hot encoding
    label_mask_one_hot = np.zeros([len(labels), label_mask.shape[0], label_mask.shape[1], label_mask.shape[2]], dtype=np.uint8)
    for idx, label in enumerate(labels):
        label_mask_one_hot[idx][label_mask == label] = 1

    # Invert the masks
    inverted_label_mask_one_hot = np.ones_like(label_mask_one_hot) - label_mask_one_hot
    
    # Get distance transform for each and get only closest component of the inverted masks
    # Distance transforms encode the distance of all foreground voxels
    # to the closest background element
    distance_labels = np.empty_like(labels, dtype=float)
    for idx in range(len(labels)):
        distance_labels[idx] = ndimage.distance_transform_edt(inverted_label_mask_one_hot[idx])[inverted_label_mask_one_hot.shape[1] // 2][inverted_label_mask_one_hot.shape[2] // 2][inverted_label_mask_one_hot.shape[3] // 2]
    # We keep only the closest component to the original endpoint
    mask = np.zeros_like(segmentation_array)
    mask[
        np.max([0, i - window_size]): np.min([segmentation_array.shape[0], i + window_size]), 
        np.max([0, j - window_size]): np.min([segmentation_array.shape[1], j + window_size]),
        np.max([0, k - window_size]): np.min([segmentation_array.shape[2], k + window_size])
        ] = label_mask_one_hot[np.argmin(distance_labels)]
    
    # Get the centroid of the foregroud region and turn it into the new endpoint
    properties = measure.regionprops(mask.astype(int), mask.astype(int))
    center_of_mass = np.array(properties[0].centroid)
    # Return the new position of the endpoint in RAS coordinates
    new_endpoint = np.matmul(segmentation_affine, np.append(center_of_mass, 1.0))[:3]

    return new_endpoint

def multi_robust_endpoint_detection(endpoint_list, segmentation_array, segmentation_affine, window_size=15, max_workers=8):
    """
    Multi-threaded version of the robust_endpoint_detection function.

    Parameters
    ----------
    endpoint_list : list
        List of endpoints to be processed.
    segmentation_array : numpy.array
        Binary array to be segmented.
    segmentation_affine : numpy.array
        Affine transformation of the binary array.
    window_size : int
        Defines the size of the region around the endpoint that is analyzed for this method.
        New endpoint location will be searched within a cubic box of size 2 * n around the 
        originial endpoint location.
    max_workers : int
        Maximum number of workers to be used in the parallel processing.

    Returns
    -------
    new_endpoints : list
        List of new positions of the endpoints.
    """
    # Prepare data for parallel processing
    data_for_processing = [(endpoint, segmentation_array, segmentation_affine, window_size) for endpoint in endpoint_list]
    
    # Process endpoints in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        new_endpoints = list(executor.map(_robust_endpoint_detection, data_for_processing))
    
    return new_endpoints

# def compute_frenet_serret(centerline_poly_data):
#     """
#     Uses the vtkParallelTransportFrame custom filter from
#     Slicer to compute tangent, normal and binormal vectors from 
#     the Frenet-Serret frame for each point of the centerline model.

#     Parameters
#     ----------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model.
    
#     Returns
#     -------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model with tangent, normal and binormal vectors 
#         computed for each centerline point as point data.
    
#     """
#     curve_coordinate_system_generator = slicer.vtkParallelTransportFrame()
#     curve_coordinate_system_generator.SetInputData(centerline_poly_data)
#     curve_coordinate_system_generator.Update()

#     return curve_coordinate_system_generator.GetOutput()

# def compute_curvature_and_torsion(centerline_poly_data):
#     """
#     Compute cruvature and torsion using the Frenet-Serret moving frame.
#     Assumes that Frenet-Serret vectors are available as point data in the 
#     input centerline_poly_data object. Stores curvature and torsion as
#     point data arrays in the centerline_poly_data object. Also computes 
#     a smoothed out version of the curvature using a Savitzky-Golay filter.

#     Parameters
#     ----------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model.

#     Returns
#     ------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model with curvature, torsion and filtered_curvature 
#         as additional point data arrays.

#     References:
#     [1]     
    
#     """
#     # Load Frenet-Serret vectors as numpy arrays for each point
#     coordinates = np.ndarray([centerline_poly_data.GetNumberOfPoints(), 3])
#     for point_idx in range(centerline_poly_data.GetNumberOfPoints()):
#         coordinates[point_idx] = centerline_poly_data.GetPoints().GetPoint(point_idx)
#     tangents = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Tangents"))
#     normals = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Normals"))
#     binormals = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Binormals"))
    
#     # Compute curvature and torsion
#     curvature = np.linalg.norm(np.gradient(tangents, axis = 0), axis = 1)
#     torsion = (- np.gradient(binormals, axis = 0) * normals).sum(axis = 1)
#     # Compute smoothed curvature
#     filtered_curvature = savgol_filter(curvature, window_length=10, polyorder=3, mode='nearest')
    
#     # Add new point data arrays
#     centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(curvature))
#     centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Curvature")
#     centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(torsion))
#     centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Torsion")
#     centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(filtered_curvature))
#     centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Filtered curvature")

#     return centerline_poly_data