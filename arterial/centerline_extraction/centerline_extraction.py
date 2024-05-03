#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
import vtk

import numpy as np

from vmtk import vtkvmtk

from arterial.centerline_extraction.utils import compute_network_centerlines, get_endpoints, aortic_arch_endpoint_check, robust_endpoint_detection, multi_robust_endpoint_detection

def get_robuts_endpoints(segmentation_model, segmentation_array, segmentation_affine):
    """
    Computed a set of robust endpoints for a segmentation surface model. It first
    computes the centerline network of the segmentation and then extracts the endpoints.
    The endpoints are then checked for the presence of the aortic arch and replaced with
    more robust endpoints.

    These functions intend to replicate the behiavior of the auto-detect endpoints function
    from the VKTK Slicer extension. 

    Parameters
    ----------
    segmentation_model : vtkPolyData
        The segmentation surface model.
    segmentation_array : numpy.array
        Binary array to be segmented.
    segmentation_affine : numpy.array
        Affine transformation of the binary array.

    Returns
    -------
    endpoints : list
        A list of 3D ijk coordinates of the robust endpoints.

    """
    # Computes the centerline network of the segmentation
    network_centerlines = compute_network_centerlines(segmentation_model)
    # Extracts the endpoints (direct implementation of the getEndPoints function of the VKTK Slicer extension)
    endpoints = get_endpoints(network_centerlines, None)
    # Converts the endpoints to numpy arrays
    for idx, endpoint in enumerate(endpoints):
        endpoints[idx] = np.array(endpoint)
    # Checks the presence of the aortic arch endpoints
    endpoints = aortic_arch_endpoint_check(endpoints, segmentation_array, segmentation_affine)
    # Computes the robust endpoints. This helps avoid centerline extraction errors due to the endpoints being outside the segmentation
    endpoints = robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10)
    # endpoints = multi_robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10, max_workers=10)

    return endpoints

def extract_centerlines(segmentation_model, endpoints):
    """
    Compute centerline.
    This is more robust and accurate but takes longer than the network extraction.
    :param segmentation_model:
    :param endPointsMarkupsNode:
    :return:
    """
    # Cap all the holes that are in the mesh that are not marked as endpoints
    # Maybe this is not needed.
    cap_displacement = 0.0
    surface_capper = vtkvmtk.vtkvmtkCapPolyData()
    surface_capper.SetInputData(segmentation_model)
    surface_capper.SetDisplacement(cap_displacement)
    surface_capper.SetInPlaneDisplacement(cap_displacement)
    surface_capper.Update()

    if len(endpoints) < 2:
        raise ValueError("At least two endpoints are needed for centerline extraction")

    tube_poly_data = surface_capper.GetOutput()
    pos = [0.0, 0.0, 0.0]
    # It seems that vtkvmtkComputationalGeometry does not need holes (unlike network extraction, which does need one hole)
    # # Punch holes at surface endpoints to have tubular structure
    # tube_poly_data = surface_capper.GetOutput()
    # numberOfEndpoints = endPointsMarkupsNode.GetNumberOfControlPoints()
    # for pointIndex in range(numberOfEndpoints):
    #     endPointsMarkupsNode.GetNthControlPointPosition(pointIndex, pos)
    #     self.openSurfaceAtPoint(tube_poly_data, pos)

    number_of_control_points = len(endpoints)
    found_start_point = False # Startpoint at index 0

    source_id_list = vtk.vtkIdList()
    target_id_list = vtk.vtkIdList()

    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(tube_poly_data)
    point_locator.BuildLocator()

    for control_point_index in range(number_of_control_points):
        is_target = True
        if not found_start_point and control_point_index == 0:
            # If no start point found then use the first point as source
            is_target = False
        pos = endpoints[control_point_index]
        # locate the point on the surface
        point_id = point_locator.FindClosestPoint(pos)
        if is_target:
            target_id_list.InsertNextId(point_id)
        else:
            source_id_list.InsertNextId(point_id)

    centerline_filter = vtkvmtk.vtkvmtkPolyDataCenterlines()
    centerline_filter.SetInputData(tube_poly_data)
    centerline_filter.SetSourceSeedIds(source_id_list)
    centerline_filter.SetTargetSeedIds(target_id_list)
    centerline_filter.SetRadiusArrayName("MaximumInscribedSphereRadius")
    centerline_filter.SetCostFunction('1/R')  # this makes path search prefer go through points with large radius
    centerline_filter.SetFlipNormals(False)
    centerline_filter.SetAppendEndPointsToCenterlines(0)

    # Voronoi smoothing slightly improves connectivity
    # Unfortunately, Voronoi smoothing is broken if VMTK is used with VTK9, therefore
    # disable this feature for now (https://github.com/vmtk/SlicerExtension-VMTK/issues/34)
    enable_voronoi_smoothing = False
    centerline_filter.SetSimplifyVoronoi(enable_voronoi_smoothing)

    centerline_filter.SetCenterlineResampling(0)
    centerline_filter.SetResamplingStepLength(1.0)
    centerline_filter.Update()

    if not centerline_filter.GetOutput():
        raise ValueError("Failed to compute centerline (no output was generated)")
    centerlines = vtk.vtkPolyData()
    centerlines.DeepCopy(centerline_filter.GetOutput())

    if not centerline_filter.GetVoronoiDiagram():
        raise ValueError("Failed to compute centerline (no Voronoi diagram was generated)")
    voronoi_diagram = vtk.vtkPolyData()
    voronoi_diagram.DeepCopy(centerline_filter.GetVoronoiDiagram())

    return centerlines, voronoi_diagram