import vtk

from vmtk import vtkvmtk
from vmtk import vmtkcenterlines, vmtkcenterlinestonumpy, vmtknetworkextraction, vmtkdelaunayvoronoi, vmtknumpytocenterlines, vmtksurfacecapper
from joblib import Parallel, delayed
import random
import numpy as np

from skimage import measure
from scipy import ndimage
import nibabel as nib

def _compute_centerlines_network(surfaceAddress, delaunayAddress, voronoiAddress, poleIdsAddress, cell, points):
    '''a method to compute centerlines which can be called in parallel
    
    Arguments:
        surfaceAddress (str): the input memory address of the surface to calculate centerlines of
        delaunayAddress (str): the memory address of a previously computed delaunay triangulation of the 
            surface vtkUnstructuredGrid 
        voronoiAddress (str): the memory address of the previously computed voronoi diagram vtkPolyData
        poleIdsAddress (str): the memory address of the previously computed poleIds vtkIdList
        cell (np.array): the cellID connectivity list
        points (np.array): the x,y,z coordinates of points identified in the cell argument
    '''

    surface = vtk.vtkPolyData(surfaceAddress)
    delaunay = vtk.vtkUnstructuredGrid(delaunayAddress)
    voronoi = vtk.vtkPolyData(voronoiAddress)
    poleIds = vtk.vtkIdList(poleIdsAddress)

    cl = _compute_centerline_branch(surface, delaunay, voronoi, poleIds, cell, points)

    clConvert = vmtkcenterlinestonumpy.vmtkCenterlinesToNumpy()
    clConvert.Centerlines = cl
    clConvert.LogOn = 0
    clConvert.Execute()
    return clConvert.ArrayDict

def _compute_centerline_branch(surface, delaunay, voronoi, poleIds, cell, points):
    cellStartIdx = cell[0]
    cellEndIdx = cell[-1]
    cellStartPoint = points[cellStartIdx].tolist()
    cellEndPoint = points[cellEndIdx].tolist()
    cl = vmtkcenterlines.vmtkCenterlines()
    cl.Surface = surface
    cl.DelaunayTessellation = delaunay
    cl.VoronoiDiagram = voronoi
    cl.PoleIds = poleIds
    cl.SeedSelectorName = 'pointlist'
    # since we only set one target seed at a time, setting StopFastMarchingOnReachingTarget
    # greatly speeds up algorithm execution time.
    cl.StopFastMarchingOnReachingTarget = 1
    cl.SourcePoints = cellStartPoint
    cl.TargetPoints = cellEndPoint
    cl.LogOn = 0
    cl.Execute()
    return cl.Centerlines

def compute_network_centerlines(surface_model):
    # feature edges are used to find any holes in the surface.
    fedges = vtk.vtkFeatureEdges()
    fedges.BoundaryEdgesOn()
    fedges.FeatureEdgesOff()
    fedges.ManifoldEdgesOff()
    fedges.SetInputData(surface_model)
    fedges.Update()
    ofedges = fedges.GetOutput()

    # if numEdges is not 0, then there are holes which need to be capped
    numEdges = ofedges.GetNumberOfPoints()
    if numEdges != 0:
        tempcapper = vmtksurfacecapper.vmtkSurfaceCapper()
        tempcapper.Surface = surface_model
        tempcapper.Interactive = 0
        tempcapper.Execute()

        networkSurface = tempcapper.Surface
    else:
        networkSurface = surface_model

    # randomly select one cell to delete so that there is an opening for
    # vmtkNetworkExtraction to use.
    numCells = networkSurface.GetNumberOfCells()
    random_generator = random.Random()
    random_generator.seed(42)
    cellToDelete = random_generator.randrange(0, numCells-1)
    networkSurface.BuildLinks()
    networkSurface.DeleteCell(cellToDelete)
    networkSurface.RemoveDeletedCells()

    # extract the network of approximated centerlines
    net = vmtknetworkextraction.vmtkNetworkExtraction()
    net.Surface = networkSurface
    net.AdvancementRatio = 1.001
    net.Execute()
    network = net.Network

    convert = vmtkcenterlinestonumpy.vmtkCenterlinesToNumpy()
    convert.Centerlines = network
    convert.LogOn = False
    convert.Execute()
    ad = convert.ArrayDict
    cellDataTopology = ad['CellData']['Topology']

    # the network topology identifies an the input segment with the "0" id.
    # since we artificially created this segment, we don't want to use the
    # ends of the segment as source/target points of the centerline calculation
    nodeIndexToIgnore = np.where(cellDataTopology[:,0] == 0)[0][0]
    keepCellConnectivityList = []
    pointIdxToKeep = np.array([])
    removeCellLength = 0
    # we remove the cell, points, and point data which are associated with the
    # segment we want to ignore
    for loopIdx, cellConnectivityList in enumerate(ad['CellData']['CellPointIds']):
        if loopIdx == nodeIndexToIgnore:
            removeCellStartIdx = cellConnectivityList[0]
            removeCellEndIdx = cellConnectivityList[-1]
            removeCellLength = cellConnectivityList.size
            if (removeCellEndIdx + 1) - removeCellStartIdx != removeCellLength:
                raise(ValueError)
            continue
        else:
            rescaledCellConnectivity = np.subtract(cellConnectivityList, removeCellLength, where=cellConnectivityList >= removeCellLength)
            keepCellConnectivityList.append(rescaledCellConnectivity)
            pointIdxToKeep = np.concatenate((pointIdxToKeep, cellConnectivityList)).astype(int)
    newPoints = ad['Points'][pointIdxToKeep]

    # precompute the delaunay tessellation for the whole surface.
    tessalation = vmtkdelaunayvoronoi.vmtkDelaunayVoronoi()
    tessalation.Surface = networkSurface
    tessalation.Execute()

    out = []
    import sys
    if (sys.platform == 'win32') or (sys.platform == 'win64') or (sys.platform == 'cygwin'):
        UseJoblib = False
    else:
        UseJoblib = True
    if UseJoblib:
        # vtk objects cannot be serialized in python. Instead of converting the inputs to numpy arrays and having
        # to reconstruct the vtk object each time the loop executes (a slow process), we can just pass in the
        # memory address of the data objects as a string, and use the vtk python bindings to create a python name
        # referring to the data residing at that memory address. This works because joblib executes each loop
        # iteration in a fork of the original process, providing access to the original memory space.
        # However, the process does not work for return arguments, since the original process will not have access to
        # the memory space of the fork. To return results we use the vmtkCenterlinesToNumpy converter.
        networkSurfaceMemoryAddress = networkSurface.__this__
        delaunayMemoryAddress = tessalation.DelaunayTessellation.__this__
        voronoiMemoryAddress = tessalation.VoronoiDiagram.__this__
        poleIdsMemoryAddress = tessalation.PoleIds.__this__
        numParallelJobs = -1
        
        # note about the verbose function: while Joblib can print a progress bar output (set verbose = 20),
        # it does not implement a callback function as of version 0.11, so we cannot report progress to the user
        # if we are redirecting standard out with the self.PrintLog method.
        outlist = Parallel(n_jobs=numParallelJobs, backend='multiprocessing', verbose=0)(
            delayed(_compute_centerlines_network)(networkSurfaceMemoryAddress,
                                            delaunayMemoryAddress,
                                            voronoiMemoryAddress,
                                            poleIdsMemoryAddress,
                                            cell,
                                            newPoints) for cell in keepCellConnectivityList)
        for item in outlist:
            npConvert = vmtknumpytocenterlines.vmtkNumpyToCenterlines()
            npConvert.ArrayDict = item
            npConvert.LogOn = 0
            npConvert.Execute()
            out.append(npConvert.Centerlines)
    else:
        for cell in keepCellConnectivityList:
                cl = _compute_centerline_branch(networkSurface, tessalation.DelaunayTessellation, tessalation.VoronoiDiagram,
                                                tessalation.PoleIds, cell, newPoints)
                out.append(cl)
                

    # Append each segment's polydata into a single polydata object
    centerlineAppender = vtk.vtkAppendPolyData()
    for data in out:
        centerlineAppender.AddInputData(data)
    centerlineAppender.Update()

    # clean and strip the output centerlines so that redundant points are merged and tracts are combined
    centerlineCleaner = vtk.vtkCleanPolyData()
    centerlineCleaner.SetInputData(centerlineAppender.GetOutput())
    centerlineCleaner.Update()

    centerlineStripper = vtk.vtkStripper()
    centerlineStripper.SetInputData(centerlineCleaner.GetOutput())
    centerlineStripper.JoinContiguousSegmentsOn()
    centerlineStripper.Update()

    network_centerlines = centerlineStripper.GetOutput()

    return network_centerlines

def get_endpoints(inputNetworkPolyData, startPointPosition):
    """ 
    Adapted from https://github.com/vmtk/SlicerExtension-VMTK/blob/3787ea4a300da28ec5f0824f0715f2713b631155/ExtractCenterline/ExtractCenterline.py#L746
    Clips the surfacePolyData on the endpoints identified using the networkPolyData.
    If startPointPosition is specified then start point will be the closest point to that position.
    Returns list of endpoint positions. Largest radius point is be the first in the list.

    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(inputNetworkPolyData)
    cleaner.Update()
    network = cleaner.GetOutput()
    network.BuildCells()
    network.BuildLinks(0)

    networkPoints = network.GetPoints()
    radiusArray = network.GetPointData().GetArray("MaximumInscribedSphereRadius")

    startPointId = -1
    maxRadius = 0
    minDistance2 = 0

    endpointIds = vtk.vtkIdList()
    for i in range(network.GetNumberOfCells()):
        numberOfCellPoints = network.GetCell(i).GetNumberOfPoints()
        if numberOfCellPoints < 2:
            continue

        for pointIndex in [0, numberOfCellPoints - 1]:
            pointId = network.GetCell(i).GetPointId(pointIndex)
            pointCells = vtk.vtkIdList()
            network.GetPointCells(pointId, pointCells)
            if pointCells.GetNumberOfIds() == 1:
                endpointIds.InsertUniqueId(pointId)
                if startPointPosition is not None:
                    # find start point based on position
                    position = networkPoints.GetPoint(pointId)
                    distance2 = vtk.vtkMath.Distance2BetweenPoints(position, startPointPosition)
                    if startPointId < 0 or distance2 < minDistance2:
                        minDistance2 = distance2
                        startPointId = pointId
                else:
                    # find start point based on radius
                    radius = radiusArray.GetValue(pointId)
                    if startPointId < 0 or radius > maxRadius:
                        maxRadius = radius
                        startPointId = pointId

    endpointPositions = []
    numberOfEndpointIds = endpointIds.GetNumberOfIds()
    if numberOfEndpointIds == 0:
        return endpointPositions
    # add the largest radius point first
    endpointPositions.append(networkPoints.GetPoint(startPointId))
    # add all the other points
    for pointIdIndex in range(numberOfEndpointIds):
        pointId = endpointIds.GetId(pointIdIndex)
        if pointId == startPointId:
            # already added
            continue
        endpointPositions.append(networkPoints.GetPoint(pointId))

    return endpointPositions

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
    for endpoint_idx, endpoint in enumerate(endpoint_list):
        print("Relocating endpoint {}: {}".format(endpoint_idx, endpoint))
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
        endpoint_list[endpoint_idx] = np.matmul(segmentation_affine, np.append(center_of_mass, 1.0))[:3]
        print("New endpoint position: {}".format(endpoint_list[endpoint_idx]))
    
    return endpoint_list

from concurrent.futures import ProcessPoolExecutor

def _robust_endpoint_detection(endpoint_data):
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
    # Prepare data for parallel processing
    data_for_processing = [(endpoint, segmentation_array, segmentation_affine, window_size) for endpoint in endpoint_list]
    
    # Process endpoints in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        new_endpoints = list(executor.map(_robust_endpoint_detection, data_for_processing))
    
    return new_endpoints