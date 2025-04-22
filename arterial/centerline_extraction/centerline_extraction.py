#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import numpy as np
import nibabel as nib

import vtk, math
from vmtk import vtkvmtk

from arterial.centerline_extraction.utils import clean_centerline, build_endpoints_json, aortic_arch_endpoint_check, robust_endpoint_relocation

def extract_centerlines_full_cta(segmentation_model, segmentation_array, segmentation_affine, is_first_model=True):
    """
    Extract centerlines from a segmentation model, following the VMTK Slicer extension logic. The only difference between
    this implementation and the Arterial 1.0 (processing within Slicer) is the generation of the segmentation model prior 
    to this process, which is done using the Arterial 2.0 segmentation model and vtk processing.

    Parameters
    ----------
    segmentation_model : vtkPolyData
        The segmentation surface model.
    segmentation_array : numpy.array
        Binary array with the original vascular segmentation.
    segmentation_affine : numpy.array
        Affine transformation of the nifti.
    is_first_model : bool
        Flag to indicate if the segmentation model is the first model of the sequence.

    Returns
    -------
    centerlines : vtkPolyData
        The centerline model.
    voronoi : vtkPolyData
        The voronoi diagram model.
    
    """
    centerline_computation_logic = CenterlineComputationLogic()
    centerlines, voronoi, endpoints_json = centerline_computation_logic.extract_centerline(segmentation_model, segmentation_array, segmentation_affine, is_first_model)
    
    if centerlines is None: # True if no endpoints are found
        return None, None, None
    
    startpoint = endpoints_json["markups"][0]["controlPoints"][0]["position"]
    if centerlines.GetNumberOfCells() == 0 or centerlines.GetNumberOfPoints() == 0:
        pass
    else:
        centerlines = clean_centerline(centerlines, startpoint=startpoint)
        print("Number of cells after centerline cleaning:", centerlines.GetNumberOfCells())
        
    if centerlines.GetNumberOfCells() == 0:
        return None, None, None
    else:
        return centerlines, voronoi, endpoints_json
    

def extract_centerline_between_endpoints(segmentation_model, segmentation_array, segmentation_affine, endpoints):
    """
    This function should extract the centerline between predefined endpoints. The endpoints should be passed either as a json
    or as a list of points. The logic will be to use the first endpoint as the startpoint and the rest as endpoints. Thus, 
    N-1 centerlines will be extracted, where N is the number of endpoints.

    Parameters
    ----------
    segmentation_model : vtkPolyData
        The segmentation surface model.
    segmentation_array : numpy.array
        Binary array with the original vascular segmentation.
    segmentation_affine : numpy.array
        Affine transformation of the nifti.
    endpoints : list
        List of points with the endpoints.

    Returns
    -------
    centerlines : vtkPolyData
        The centerline model.
    voronoi : vtkPolyData
        The voronoi diagram model.
    
    """
    centerline_computation_logic = CenterlineComputationLogic()
    centerlines, voronoi = centerline_computation_logic.extract_centerline_between_endpoints(segmentation_model, segmentation_array, segmentation_affine, endpoints)
    
    if centerlines is None: # True if no endpoints are found
        return None, None, None
        
    if centerlines.GetNumberOfCells() == 0:
        return None, None
    else:
        return centerlines, voronoi

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
        # Extract endpoints
        endpoints = self.autodetect_endpoints(network, prepared_surface_model)

        if endpoints.GetNumberOfPoints() == 0:
            if is_first_model:
                raise ValueError("No endpoints found for first model. This is a critical error. Please check the segmentation.")
            else:
                print("No endpoints were found in a secondary island, returning empty centerlines")
                return None, None, None
        
        print(f"Found {endpoints.GetNumberOfPoints()} endpoints.")

        if is_first_model:
            # Check the presence of the aortic arch endpoints
            print("Checking aortic arch endpoints...")
            endpoints = aortic_arch_endpoint_check(endpoints, segmentation_array, segmentation_affine)

        # Computes the robust endpoints. This helps avoid centerline extraction errors due to the endpoints being outside the segmentation
        endpoints = robust_endpoint_relocation(endpoints, segmentation_array, segmentation_affine, window_size=15, larger_window_for_aa_startpoint=is_first_model)
        # Convert the endpoints to a JSON format (compatible with Markups module for visualization in 3D Slicer)
        endpoints_json = build_endpoints_json(endpoints, segmentation_affine)

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
        point_locator.SetDataSet(surface_model)
        point_locator.BuildLocator()

        # locate the source on the surface
        source_id = point_locator.FindClosestPoint(source_point)
        source_id_list.InsertNextId(source_id)

        # locate the endpoints on the surface
        for p in target_points:
            id = point_locator.FindClosestPoint(p)
            target_id_list.InsertNextId(id)

        print("Computing centerlines...")
        new_centerlines, new_voronoi = self.compute_centerlines(surface_model, source_id_list, target_id_list)

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

    def autodetect_endpoints(self, network, surface_model):
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

        return endpoints_points

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
