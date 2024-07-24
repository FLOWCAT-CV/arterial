#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk, math

from vmtk import vtkvmtk

import numpy as np
import nibabel as nib

from skimage import measure
from scipy import ndimage
from scipy.spatial import cKDTree

from arterial.io.load_and_save_operations import *

class CenterlineComputationLogic(object):
    """
    Centerline computation logic class from Slicer's VMTK extension. This class is 
    an adapted version of the class defined in 
        https://github.com/vmtk/SlicerExtension-VMTK/blob/e9aa8e7e532299c10cfae5746e9c7ef06a4314b1/CenterlineComputation/CenterlineComputation.py#L293

    Additional functionality has been added to ensure robust endpoint detection and aortic arch endpoint check.

    """
    def __init__(self):
        """
        Class constructor. Initializes the array names for the centerline computation.
        
        """
        self.blanking_array_name = "Blanking"
        self.radius_array_name = "MaximumInscribedSphereRadius"
        self.group_ids_array_name = "GroupIds"
        self.centerline_ids_array_name = "CenterlineIds"
        self.tract_ids_array_name = "TractIds"
        self.topology_array_name = "Topology"
        self.marks_array_name = "Marks"
        self.length_array_name = "Length"
        self.curvature_array_name = "Curvature"
        self.torsion_array_name = "Torsion"
        self.tortuosity_array_name = "Tortuosity"

    def extract_centerline(self, surface_model, segmentation_array, segmentation_affine, is_first_model=True):
        """
        Extracts the centerline from the given surface model. The pipeline comprises the following steps:
        1. Prepare the model
        2. Decimate the model
        3. Open the model at the seed
        4. Extract the network
        5. Clip the surface at the endpoints
        6. Compute the centerlines

        Parameters
        ----------
        surface_model : vtkPolyData
            The surface model.
        segmentation_array : numpy.array
            The segmentation array.
        segmentation_affine : numpy.array
            The segmentation affine.
        is_first_model : bool, optional
            Whether the model is the first one. The default is True.

        Returns
        -------
        centerlines : vtkPolyData
            The centerlines.
        voronoi : vtkPolyData
            The Voronoi diagram.
        network : vtkPolyData   
            The network.
        decimated_surface_model : vtkPolyData
            The decimated surface model.
        prepared_surface_model : vtkPolyData
            The prepared surface model.
        
        """
        # Define the output models
        prepared_surface_model = vtk.vtkPolyData()
        network = vtk.vtkPolyData()
        centerlines = vtk.vtkPolyData()
        voronoi = vtk.vtkPolyData()

        # Get the seed point in RAS coordinates close to the distal AA
        current_coordinates_ras = self.get_seed_ras(segmentation_array, segmentation_affine)

        print("Setting endpoint seed at", current_coordinates_ras)

        # Prepare the model (cleaning, triangulation, smoothing, normal recalculation, capping)
        print("Preparing model...")
        prepared_surface_model.DeepCopy(self.prepare_model(surface_model, non_manifold_edges=None))

        if prepared_surface_model.GetNumberOfPoints() == 0:
            raise ValueError("Input model preparation failed. It probably has surface errors.")

        # Open the model at the seed (only for network extraction)
        self.open_surface_at_point(prepared_surface_model, current_coordinates_ras)

        print("Extracting network...")
        # Extract Network
        network.DeepCopy(self.extract_network(prepared_surface_model))
        
        # Here we start the actual centerline computation which is mathematically more robust and accurate but takes longer than the network extraction
        print("Clipping surface at endpoints...")
        # clip surface at endpoints identified by the network extraction
        clipped_surface, endpoints = self.clip_surface_at_end_points(network, prepared_surface_model)

        print(f"Found {endpoints.GetNumberOfPoints()} endpoints.")

        if is_first_model:
            # Check the presence of the aortic arch endpoints
            print("Checking aortic arch endpoints...")
            endpoints = aortic_arch_endpoint_check(endpoints, segmentation_array, segmentation_affine)
        # Computes the robust endpoints. This helps avoid centerline extraction errors due to the endpoints being outside the segmentation
        endpoints = robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10, larger_window_for_aa_startpoint=is_first_model)
        # Convert the endpoints to a JSON format (compatible with Markups module for visualization in 3D Slicer)
        endpoints_json = build_endpoints_json(endpoints)

        # Now find the one endpoint which is closest to the seed and use it as the source point for centerline computation
        # all other endpoints are the target points
        source_point = current_coordinates_ras

        # the following arrays have the same indexes and are synchronized at all times
        distances_to_seed = []
        target_points = []

        # Get distances of points from source point
        for i in range(endpoints.GetNumberOfPoints()):
            current_point = endpoints.GetPoint(i)
            # get the euclidean distance
            current_distance_to_seed = math.sqrt(math.pow((current_point[0] - source_point[0]), 2) +
                                                 math.pow((current_point[1] - source_point[1]), 2) +
                                                 math.pow((current_point[2] - source_point[2]), 2))

            target_points.append(current_point)
            distances_to_seed.append(current_distance_to_seed)

        # the index with minimal distance is the point closest to the seed, we want to set it as sourcepoint
        # all other points are the targetpoints
        source_point_index = 0
        # .. and remove it after saving it as the source_point
        source_point = target_points[source_point_index]
        distances_to_seed.pop(source_point_index)
        target_points.pop(source_point_index)

        # at this point we have the source_point and a list of real target_points

        # now create the source_id_list and target_id_list for the actual centerline computation
        source_id_list = vtk.vtkIdList()
        target_id_list = vtk.vtkIdList()

        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(clipped_surface)
        point_locator.BuildLocator()

        # locate the source on the surface
        source_id = point_locator.FindClosestPoint(source_point)
        source_id_list.InsertNextId(source_id)

        # locate the endpoints on the surface
        for p in target_points:
            id = point_locator.FindClosestPoint(p)
            target_id_list.InsertNextId(id)

        print("Computing centerlines...")
        new_centerlines, new_voronoi = self.compute_centerlines(clipped_surface, source_id_list, target_id_list)

        centerlines.DeepCopy(new_centerlines)
        voronoi.DeepCopy(new_voronoi)

        return centerlines, voronoi, endpoints_json
    
    def get_seed_ras(self, segmentation_array, segmentation_affine):
        """
        Given a segmentation array and its affine, returns the seed point in RAS coordinates.
        The reference point is close to the distal AA.

        Parameters
        ----------
        segmentation_array : numpy.array
            The segmentation array.
        segmentation_affine : numpy.array
            The segmentation affine.

        Returns
        -------
        seed_ras : numpy.array
            The seed point in RAS coordinates.

        """
        # Select startpoint (seed) near the distal AA
        factor = abs(0.43 / segmentation_affine[0, 0])
        if nib.orientations.aff2axcodes(segmentation_affine) == ("R", "A", "S"):
            seed_ijk = np.array([150.0 * factor, 0.0, 0.0])
        elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "A", "S"):
            seed_ijk = np.array([350.0 * factor, 0.0, 0.0])
        elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "P", "S"):
            seed_ijk = np.array([350.0 * factor, segmentation_array.shape[1], 0.0])

        seed_ras = np.dot(segmentation_affine, np.append(seed_ijk, 1))[:3]

        return seed_ras

    def prepare_model(self, surface_model, non_manifold_edges=None):
        """
        Prepares the given surface for centerline extraction. Basically, it cleans the surface, triangulates it, applies
        smoothing, recalculates normals, caps the surface, and checks for non-manifold edges.

        Parameters
        ----------
        surface_model : vtkPolyData
            The surface to be prepared.
        non_manifold_edges : vtkPolyData, optional
            The non-manifold edges. The default is None.

        Returns
        -------
        prepared_surface_model : vtkPolyData
            The prepared surface.
            
        """
        # Clean the surface
        surface_cleaner = vtk.vtkCleanPolyData()
        surface_cleaner.SetInputData(surface_model)
        surface_cleaner.Update()

        # Triangulate the surface
        surface_triangulator = vtk.vtkTriangleFilter()
        surface_triangulator.SetInputConnection(surface_cleaner.GetOutputPort())
        surface_triangulator.PassLinesOff()
        surface_triangulator.PassVertsOff()
        surface_triangulator.Update()

        # Recompute normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(surface_triangulator.GetOutputPort())
        normals.SetAutoOrientNormals(1)
        normals.SetFlipNormals(0)
        normals.SetConsistency(1)
        normals.SplittingOff()
        normals.Update()

        # Cap the surface
        surface_capper = vtkvmtk.vtkvmtkCapPolyData()
        surface_capper.SetInputConnection(normals.GetOutputPort())
        surface_capper.SetDisplacement(0.0)
        surface_capper.SetInPlaneDisplacement(0.0)
        surface_capper.Update()

        # Apply a connectivity filter to remove disconnected parts 
        connectivity_filter = vtk.vtkConnectivityFilter()
        connectivity_filter.SetInputConnection(surface_capper.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()

        prepared_surface_model = vtk.vtkPolyData()
        prepared_surface_model.DeepCopy(surface_capper.GetOutput())

        # Check for non-manifold edges
        if non_manifold_edges:
            self.check_non_manifold_surface(prepared_surface_model, non_manifold_edges)

        return prepared_surface_model

    def check_non_manifold_surface(self, surface_model, non_manifold_edges):
        """
        Returns pairs of point IDs with endpoints of non-manifold edgesin non_manifold_edges.

        Parameters
        ----------
        surface_model : vtkPolyData
            The surface to be checked.
        non_manifold_edges : vtkPolyData
            The non-manifold edges.

        Returns
        -------

        """
        neighborhoods = vtkvmtk.vtkvmtkNeighborhoods()
        neighborhoods.SetNeighborhoodTypeToPolyDataManifoldNeighborhood()
        neighborhoods.SetDataSet(surface_model)
        neighborhoods.Build()

        surface_model.BuildCells()
        surface_model.BuildLinks(0)

        neighborCellIds = vtk.vtkIdList()
        nonManifoldEdgeLines = vtk.vtkCellArray()
        for i in range(neighborhoods.GetNumberOfNeighborhoods()):
            neighborhood = neighborhoods.GetNeighborhood(i)
            for j in range(neighborhood.GetNumberOfPoints()):
                neighborId = neighborhood.GetPointId(j)
                if i < neighborId:
                    neighborCellIds.Initialize()
                    surface_model.GetCellEdgeNeighbors(-1,i,neighborId,neighborCellIds)
                    if neighborCellIds.GetNumberOfIds() > 2:
                        nonManifoldEdgeLines.InsertNextCell(2)
                        nonManifoldEdgeLines.InsertCellPoint(i)
                        nonManifoldEdgeLines.InsertCellPoint(neighborId)

        non_manifold_edges.Initialize()
        points = vtk.vtkPoints()
        points.DeepCopy(surface_model.GetPoints())
        non_manifold_edges.SetPoints(points)
        non_manifold_edges.SetLines(nonManifoldEdgeLines)

    def decimate_surface(self, surface_model, decimation_factor=0.75):
        """
        Decimates the given surface by the given factor.

        Parameters
        ----------  
        surface_model : vtkPolyData
            The surface to be decimated.
        decimation_factor : float, optional
            The factor by which to decimate the surface. The default is 0.75.

        Returns
        -------
        decimated_surface_model : vtkPolyData
            The decimated surface.

        """
        decimation_filter = vtk.vtkDecimatePro()
        decimation_filter.SetInputData(surface_model)
        decimation_filter.SetTargetReduction(decimation_factor)
        decimation_filter.SetBoundaryVertexDeletion(0)
        decimation_filter.PreserveTopologyOn()
        decimation_filter.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(decimation_filter.GetOutput())
        cleaner.Update()

        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(cleaner.GetOutput())
        triangle_filter.Update()

        decimated_surface_model = vtk.vtkPolyData()
        decimated_surface_model.DeepCopy(triangle_filter.GetOutput())

        return decimated_surface_model

    def open_surface_at_point(self, surface_model, seed):
        """
        Returns a new surface with an opening at the given seed.

        Parameters
        ----------
        surface_model : vtkPolyData
            The surface to be opened.
        seed : vtkPoint
            The seed point.

        Returns
        -------

        """
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(surface_model)
        point_locator.BuildLocator()

        # find the closest point next to the seed on the surface
        id = point_locator.FindClosestPoint(seed)

        if id<0:
            # Calling GetPoint(-1) would crash the application
            raise ValueError("open_surface_at_point failed: empty input polydata")

        # Tell the polydata to build "upward" links from points to cells
        surface_model.BuildLinks()
        # Mark cells as deleted
        cell_ids = vtk.vtkIdList()
        surface_model.GetPointCells(id, cell_ids)
        for cell_id_index in range(cell_ids.GetNumberOfIds()):
            surface_model.DeleteCell(cell_ids.GetId(cell_id_index))
        # Remove the marked cells
        surface_model.RemoveDeletedCells()

    def extract_network(self, surface_model):
        """
        Returns the network of the given surface.

        Parameters
        ----------
        surface_model : vtkPolyData
            The surface from which to extract the network.

        Returns
        -------
        network : vtkPolyData
            The network of the surface.
        """
        network_extraction = vtkvmtk.vtkvmtkPolyDataNetworkExtraction()
        network_extraction.SetInputData(surface_model)
        network_extraction.SetAdvancementRatio(1.05)
        network_extraction.SetRadiusArrayName(self.radius_array_name)
        network_extraction.SetTopologyArrayName(self.topology_array_name)
        network_extraction.SetMarksArrayName(self.marks_array_name)
        network_extraction.Update()

        network = vtk.vtkPolyData()
        network.DeepCopy(network_extraction.GetOutput())

        return network

    def clip_surface_at_end_points(self, network, surface_model):
        """
        Clips the surface_poly_data on the endpoints identified using the network_poly_data.

        Returns a tupel of the form [clippedPolyData, endpoints_points]

        Parameters
        ----------
        network : vtkPolyData
            The network of the surface.
        surface_model : vtkPolyData
            The surface to be clipped.

        Returns
        -------
        clipped_surface_model : vtkPolyData
            The clipped surface.
        endpoints_points : vtkPoints
            The endpoints of the clipped surface.

        """
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(network)
        cleaner.Update()
        network = cleaner.GetOutput()
        network.BuildCells()
        network.BuildLinks(0)
        endpoint_ids = vtk.vtkIdList()

        radius_array = network.GetPointData().GetArray(self.radius_array_name)

        endpoints = vtk.vtkPolyData()
        endpoints_points = vtk.vtkPoints()
        endpoints_radius = vtk.vtkDoubleArray()
        endpoints_radius.SetName(self.radius_array_name)
        endpoints.SetPoints(endpoints_points)
        endpoints.GetPointData().AddArray(endpoints_radius)

        radius_factor = 1.2
        min_radius = 0.01
        for i in range(network.GetNumberOfCells()):
            number_of_cell_points = network.GetCell(i).GetNumberOfPoints()
            point_id_0 = network.GetCell(i).GetPointId(0)
            point_id_1 = network.GetCell(i).GetPointId(number_of_cell_points - 1)

            point_cells = vtk.vtkIdList()
            network.GetPointCells(point_id_0, point_cells)
            number_of_endpoints = endpoint_ids.GetNumberOfIds()
            if point_cells.GetNumberOfIds() == 1:
                point_id = endpoint_ids.InsertUniqueId(point_id_0)
                if point_id == number_of_endpoints:
                    point = network.GetPoint(point_id_0)
                    radius = radius_array.GetValue(point_id_0)
                    radius = max(radius, min_radius)
                    endpoints_points.InsertNextPoint(point)
                    endpoints_radius.InsertNextValue(radius_factor * radius)

            point_cells = vtk.vtkIdList()
            network.GetPointCells(point_id_1, point_cells)
            number_of_endpoints = endpoint_ids.GetNumberOfIds()
            if point_cells.GetNumberOfIds() == 1:
                point_id = endpoint_ids.InsertUniqueId(point_id_1)
                if point_id == number_of_endpoints:
                    point = network.GetPoint(point_id_1)
                    radius = radius_array.GetValue(point_id_1)
                    radius = max(radius, min_radius)
                    endpoints_points.InsertNextPoint(point)
                    endpoints_radius.InsertNextValue(radius_factor * radius)

        clipped_surface_model = vtk.vtkPolyData()
        clipped_surface_model.DeepCopy(surface_model)
        number_of_endpoints = endpoints_points.GetNumberOfPoints()
        for point_index in range(number_of_endpoints):
            self.open_surface_at_point(clipped_surface_model, endpoints_points.GetPoint(point_index))

        return [clipped_surface_model, endpoints_points]

    def compute_centerlines(self, surface_model, inlet_seed_ids, outlet_seed_ids):
        """
        Computes the centerlines of the given surface, using the given inlet and outlet seed IDs.

        Returns a tupel of two vtkPolyData objects.
        The first are the centerlines, the second is the corresponding Voronoi diagram.

        Parameters
        ----------
        surface_model : vtkPolyData
            The surface from which to compute the centerlines.
        inlet_seed_ids : vtkIdList
            The seed IDs of the inlet.
        outlet_seed_ids : vtkIdList
            The seed IDs of the outlet.

        Returns
        -------
        centerlines : vtkPolyData
            The centerlines model.
        voronoi : vtkPolyData
            The Voronoi diagram.

        """
        centerline_filter = vtkvmtk.vtkvmtkPolyDataCenterlines()
        centerline_filter.SetInputData(surface_model)
        centerline_filter.SetSourceSeedIds(inlet_seed_ids)
        centerline_filter.SetTargetSeedIds(outlet_seed_ids)
        centerline_filter.SetRadiusArrayName(self.radius_array_name)
        centerline_filter.SetCostFunction("1/R")
        centerline_filter.SetFlipNormals(False)
        centerline_filter.SetAppendEndPointsToCenterlines(0)
        centerline_filter.SetSimplifyVoronoi(0)
        centerline_filter.SetCenterlineResampling(0)
        centerline_filter.SetResamplingStepLength(1.0)
        centerline_filter.Update()

        centerlines = vtk.vtkPolyData()
        centerlines.DeepCopy(centerline_filter.GetOutput())

        voronoi = vtk.vtkPolyData()
        voronoi.DeepCopy(centerline_filter.GetVoronoiDiagram())

        return [centerlines, voronoi]
    
def build_endpoints_json(endpoint_list):
    endpoints_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": "false",
                "fixedNumberOfControlPoints": "false",
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 0,
                "controlPoints": [],
                "measurements": [],
                "display": {
                    "visibility": "true",
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": "false",
                    "pointLabelsVisibility": "false",
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 2.5,
                    "useGlyphScale": "true",
                    "sliceProjection": "false",
                    "sliceProjectionUseFiducialColor": "true",
                    "sliceProjectionOutlinedBehindSlicePlane": "false",
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": "false",
                    "translationHandleVisibility": "true",
                    "rotationHandleVisibility": "true",
                    "scaleHandleVisibility": "true",
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }

    # Pass from vtkPoints to list
    endpoint_list_ = [list(endpoint_list.GetPoint(idx)) for idx in range(endpoint_list.GetNumberOfPoints())]

    for idx, endpoint in enumerate(endpoint_list_):
        endpoints_json["markups"][0]["controlPoints"].append(
            {
                "id": str(idx + 1),
                "label": "Endpoints-1",
                "description": "",
                "associatedNodeID": "",
                "position": list(endpoint),
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": "false",
                "locked": "false",
                "visibility": "true",
                "positionStatus": "defined"
            }
        )
        endpoints_json["markups"][0]["lastUsedControlPointNumber"] += 1

    return endpoints_json

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
    if segmentation_volume < 4e4: # Empirically tested
        raise ValueError("Segmentation volume is too small: {:.2f} mm3".format(segmentation_volume))
    if bouding_box_volume < 3.5e6: # Empirically tested
        raise ValueError("Bounding box volume is too small: {:.2f} mm3".format(bouding_box_volume))
    if bouding_box_volume > 3.5e7: # Empirically tested
        raise ValueError("Bounding box volume is too large: {:.2f} mm3".format(bouding_box_volume))
    if segmentation_volume < 5e4 and bouding_box_volume < 5.5e6: # Empirically tested
        raise ValueError("Combination of segmentation volume and bounding box volume is too small: \nSegmentation volume: {:.2f} mm3 \nBounding box volume: {:.2f}".format(segmentation_volume, bouding_box_volume))

def robust_endpoint_detection(endpoint_vtk_points, segmentation_array, segmentation_affine, window_size = 5, larger_window_for_aa_startpoint=False):
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
    for endpoint_idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        endpoint = endpoint_vtk_points.GetPoint(endpoint_idx)
        # Compute endpoint ijk coordinates with affine matrix
        i, j, k = np.round(np.matmul(segmentation_affine_inv, np.append(endpoint, 1.0))[:3]).astype(int)
        # Define limits of the region of interest
        if larger_window_for_aa_startpoint and endpoint_idx == 0:
            i_min, i_max = np.clip([i - (window_size + 15), i + (window_size + 15)], 0, segmentation_array.shape[0])
            j_min, j_max = np.clip([j - (window_size + 15), j + (window_size + 15)], 0, segmentation_array.shape[1])
            k_min, k_max = np.clip([k - (window_size + 15), k + (window_size + 15)], 0, segmentation_array.shape[2])
        else:
            i_min, i_max = np.clip([i - window_size, i + window_size], 0, segmentation_array.shape[0])
            j_min, j_max = np.clip([j - window_size, j + window_size], 0, segmentation_array.shape[1])
            k_min, k_max = np.clip([k - window_size, k + window_size], 0, segmentation_array.shape[2])
        # Mask the segmentation_array (only region of interest)
        masked_segmentation = segmentation_array[i_min:i_max, j_min:j_max, k_min:k_max]
        # Divide into different connected components
        label_mask = measure.label(masked_segmentation, connectivity=1)
        unique_labels = np.unique(label_mask)[1:] 

        if unique_labels.size > 1:
            # Only perform distance transformation when necessary
            distances = ndimage.distance_transform_edt(label_mask == 0, return_distances=True, return_indices=False)
            nearest_label = unique_labels[np.argmin([np.min(distances[label_mask == lbl]) for lbl in unique_labels])]
            properties = measure.regionprops((label_mask == nearest_label).astype(int))
            centroid = properties[0].centroid + np.array([i_min, j_min, k_min])
        elif unique_labels.size == 1:
            # If only one label, use its centroid directly
            properties = measure.regionprops(label_mask.astype(int), label_mask == unique_labels[0])
            centroid = properties[0].centroid + np.array([i_min, j_min, k_min])
        else:
            # Default to the original coordinates if no labels were found
            centroid = [i, j ,k]

        # Return the new position of the endpoint in RAS coordinates
        endpoint_vtk_points.SetPoint(endpoint_idx, np.matmul(segmentation_affine, np.append(centroid, 1.0))[:3])

    return endpoint_vtk_points

def aortic_arch_endpoint_check(endpoint_vtk_points, segmentation_array, segmentation_affine):
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
    # For AA endpoints check (distance from bottom slice in mm)
    threshold_distance = 50 # mm

    # Divide into different connected components of the bottom slice
    label_mask = measure.label(segmentation_array[:, :, 0])
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
    aa_centroids_to_be_found = np.zeros(shape = (len(properties), 3))
    for idx, prop in enumerate(properties):
        aa_centroids_to_be_found[idx] = np.matmul(segmentation_affine, np.append(np.array(prop.centroid), [1.0, 1.0]))[:3] # result in RAS coordinates

    # Compute distance from each endpoint to all centroids of components in the bottom slice
    # The goal is to check that each component (generallly there should be 2) has one endpoint
    # nearby
    delete_indices = []
    for endpoint_idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        endpoint = endpoint_vtk_points.GetPoint(endpoint_idx)
        for idx_centroids, centroid in enumerate(aa_centroids_to_be_found):
            # If a connnected component is found close to an endpoint, we accept it as correctly placed
            # We remove the AA centroid from the list of aa_centroids as a way of saying "this one is found" 
            if np.linalg.norm(centroid - endpoint) < threshold_distance: # Threshold at 50 mm
                delete_indices.append(idx_centroids)
    if len(delete_indices) > 0:
        aa_centroids_to_be_found = np.delete(aa_centroids_to_be_found, delete_indices, axis=0)

    # If any aa_centroids_to_be_found survive, it means that no enpoints were found close by
    if len(aa_centroids_to_be_found) > 0:
        print("{} AA islands do not have associated endpoints".format(len(aa_centroids_to_be_found)))
        # This way, we convert the remaining centroids to endpoints
        for centroid in aa_centroids_to_be_found:
            print("Adding endpoint at", centroid)
            endpoint_vtk_points.InsertNextPoint(centroid)

    # Now all that's left is to ensure that the startpoint is placed at the descending aorta
    # (most proximal point from femoral access in endovascular interventions)
            
    # Select distal AA endpoint as startpoint (in some cases, the distal LSA endpoint is closer to the origin)
    # The criteria will be to choose the AA endpoint (at < 50 mm from bottom slice) that is closest to the reference point
    # Check every other point's distance to origin (ijk)
    # Reference point set at [350, 0, 0] in LAS coordinates
    if nib.orientations.aff2axcodes(segmentation_affine) == ("R", "A", "S"):
        aa_reference_voxel_coordinates = np.array([150.0 * factor, 0.0, 0.0])
    elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "A", "S"):
        aa_reference_voxel_coordinates = np.array([350.0 * factor, 0.0, 0.0])
    elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "P", "S"):
        aa_reference_voxel_coordinates = np.array([350.0 * factor, label_mask.shape[1], 0.0])
    aa_reference_ras_coordinates = np.dot(segmentation_affine, np.append(aa_reference_voxel_coordinates, 1))[:3]

    # We store the distance to the reference point for each endpoint (in mm)
    distance_to_reference = []
    for endpoint_idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        endpoint = endpoint_vtk_points.GetPoint(endpoint_idx)
        distance_to_reference.append(np.linalg.norm(endpoint - aa_reference_ras_coordinates))

    # Get order from closest to furthest
    sorted_distance_idx = np.argsort(distance_to_reference)
    startpoint_candidate = endpoint_vtk_points.GetPoint(sorted_distance_idx[0])
    # Check if the endpoint at 0 is at the distal AA
    if sorted_distance_idx[0] == 0:
        print("Original startpoint is at distal AA")
    # If it is not, set next closest endpoint to reference as startpoint if it is closer to bottom slice
    else:
        print("New startpoint ({}): {}".format(sorted_distance_idx[0], startpoint_candidate))
        endpoint_vtk_points.SetPoint(sorted_distance_idx[0], endpoint_vtk_points.GetPoint(0))
        endpoint_vtk_points.SetPoint(0, startpoint_candidate)
    
    return endpoint_vtk_points

def consolidate_points(centerline_model, threshold=1e-2):
    """
    Maps all points that are within a threshold distance of each other to a single reference
    point, so that centerlines that overlap actually overlap (i.e. share the same points).

    Parameters
    ----------
    centerline_model : vtk.vtkPolyData
        Centerline model.
    threshold : float, optional
        Threshold distance for grouping points. The default is 1e-2.

    Returns
    -------
    index_map : dict
        Dictionary that maps original point indices to the representative point index.

    """
    points = np.array([centerline_model.GetPoint(idx) for idx in range(centerline_model.GetNumberOfPoints())])
    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, r=threshold)

    # groups is a nested list containing groups of centerline point ids that are within threshold distance of each other
    # Each entry idx (a list of point ids) represents all the point ids that are grouped with that point

    # Map original indices to new consolidated indices
    index_map = {}
    for idx, group in enumerate(groups):
        if not group:
            print(f"Empty group ({idx})")
        representative_index = group[0]  # Take the first point in group as representative. It will be the smallest index of the group, because they are always sorted
        for index in group:
            index_map[index] = representative_index # index_map is a dictionary that related point_ids with the representative point_id

    return index_map

def update_new_centerline_model(centerline_model, index_map):
    """
    Updates the polydata structure by consolidating points that are within a 
    threshold distance of each other. Basically assigns the same position and point data values
    exactly to all points within the threshold distance, keeping the original number of points 
    and cell connectivity.

    Parameters
    ----------
    centerline_model : vtk.vtkPolyData
        Centerline model.
    index_map : dict
        Dictionary that maps original point indices to the representative point index.

    Returns
    -------
    new_centerline_model : vtk.vtkPolyData
        Updated centerline model with consolidated points.
        
    """
    # Get original points and point data arrays
    original_points = centerline_model.GetPoints()
    num_point_arrays = centerline_model.GetPointData().GetNumberOfArrays()
    
    # Create new points and point data structures
    new_points = vtk.vtkPoints()
    new_point_arrays = [vtk.vtkDoubleArray() for _ in range(num_point_arrays)]
    
    # Copy the attributes and names of the original point data arrays
    for i in range(num_point_arrays):
        array = centerline_model.GetPointData().GetArray(i)
        new_point_arrays[i].SetName(array.GetName())
        new_point_arrays[i].SetNumberOfComponents(array.GetNumberOfComponents())

    # Mapping of old indices to new indices after consolidation
    new_index_map = {}

    # Ensure that each representative index has a new index
    for representative_index in index_map.values():
        if representative_index not in new_index_map:
            # Add point to new_points, and save the new index
            new_point_idx = new_points.InsertNextPoint(original_points.GetPoint(representative_index))
            new_index_map[representative_index] = new_point_idx
            # Copy data for this point
            for i in range(num_point_arrays):
                original_array = centerline_model.GetPointData().GetArray(i)
                value = [original_array.GetComponent(representative_index, j) for j in range(original_array.GetNumberOfComponents())]
                new_point_arrays[i].InsertNextTuple(value)

    new_centerline_model = vtk.vtkPolyData()
    new_centerline_model.SetPoints(new_points)
    for new_array in new_point_arrays:
        new_centerline_model.GetPointData().AddArray(new_array)

    # Remap the cells
    new_cells = vtk.vtkCellArray()
    for i in range(centerline_model.GetNumberOfCells()):
        cell = centerline_model.GetCell(i)
        new_cell_points = vtk.vtkIdList()
        for j in range(cell.GetNumberOfPoints()):
            original_index = cell.GetPointId(j)
            representative_index = index_map[original_index]
            new_index = new_index_map[representative_index]
            new_cell_points.InsertNextId(new_index)
        new_cells.InsertNextCell(new_cell_points)
    
    new_centerline_model.SetLines(new_cells)

    return new_centerline_model

def clean_centerline(centerline_model, startpoint, threshold=1e-3):
    """
    Applies the consolidate_points and update_polydata functions to clean the centerline model.
    The result is a centerline model with consolidated points, i.e., centerlines that overlap with
    points that share the exact position and data values. This helps simplify postprocessing steps.

    Also, we remove cells with less than 3 points and cells that do not start at the startpoint.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Centerline model.
    threshold : float, optional
        Threshold distance for grouping points. The default is 1e-3.

    Returns
    -------
    new_polydata : vtk.vtkPolyData
        Updated centerline model with consolidated points.
        
    """
    index_map = consolidate_points(centerline_model, threshold)

    print(f"Found {centerline_model.GetNumberOfCells()} centerline cells")

    # Remove cells with less than 3 points
    number_of_removed_cells = 0
    for idx in range(centerline_model.GetNumberOfCells()):
        cell = centerline_model.GetCell(idx)
        if np.linalg.norm(np.array(cell.GetPoints().GetPoint(0)) - startpoint) > 30:
            centerline_model.DeleteCell(idx)
            number_of_removed_cells += 1
        elif cell.GetNumberOfPoints() <= 2:
            centerline_model.DeleteCell(idx)
            number_of_removed_cells += 1

    centerline_model.RemoveDeletedCells()
    print(f"Removed {number_of_removed_cells} cells (less than 3 points or not starting at startpoint)")

    return update_new_centerline_model(centerline_model, index_map)

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