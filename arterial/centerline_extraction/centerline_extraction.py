#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.centerline_extraction.utils import CenterlineComputationLogic, clean_centerline

def extract_centerlines(segmentation_model, segmentation_array, segmentation_affine, is_first_model=True):
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
    centerlines, voronoi = centerline_computation_logic.extract_centerline(segmentation_model, segmentation_array, segmentation_affine, is_first_model)
    centerlines = clean_centerline(centerlines)
    
    return centerlines, voronoi


# from arterial.centerline_extraction.utils import compute_network_centerlines, get_endpoints, aortic_arch_endpoint_check, robust_endpoint_detection, multi_robust_endpoint_detection

# def get_robuts_endpoints(segmentation_model, segmentation_array, segmentation_affine, is_first_model=True):
#     """
#     Computed a set of robust endpoints for a segmentation surface model. It first
#     computes the centerline network of the segmentation and then extracts the endpoints.
#     The endpoints are then checked for the presence of the aortic arch and replaced with
#     more robust endpoints.

#     These functions intend to replicate the behiavior of the auto-detect endpoints function
#     from the VKTK Slicer extension. 

#     Parameters
#     ----------
#     segmentation_model : vtkPolyData
#         The segmentation surface model.
#     segmentation_array : numpy.array
#         Binary array to be segmented.
#     segmentation_affine : numpy.array
#         Affine transformation of the binary array.

#     Returns
#     -------
#     endpoints : list
#         A list of 3D ijk coordinates of the robust endpoints.

#     """
#     # Computes the centerline network of the segmentation
#     network_centerlines = compute_network_centerlines(segmentation_model)
#     # Extracts the endpoints (direct implementation of the getEndPoints function of the VKTK Slicer extension)
#     endpoints = get_endpoints(network_centerlines, None)
#     # Converts the endpoints to numpy arrays
#     for idx, endpoint in enumerate(endpoints):
#         endpoints[idx] = np.array(endpoint)
#     if is_first_model:
#         # Checks the presence of the aortic arch endpoints
#         endpoints = aortic_arch_endpoint_check(endpoints, segmentation_array, segmentation_affine)
#     # Computes the robust endpoints. This helps avoid centerline extraction errors due to the endpoints being outside the segmentation
#     endpoints = robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10)
#     # endpoints = multi_robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10, max_workers=10)

#     return endpoints

# def extract_centerlines(segmentation_model, endpoints):
#     """
#     Compute centerline.
#     This is more robust and accurate but takes longer than the network extraction.
#     :param segmentation_model:
#     :param endPointsMarkupsNode:
#     :return:
#     """
#     if len(endpoints) < 2:
#         raise ValueError("At least two endpoints are needed for centerline extraction")
    
#     pos = [0.0, 0.0, 0.0]

#     number_of_control_points = len(endpoints)
#     found_start_point = False # Startpoint at index 0

#     source_id_list = vtk.vtkIdList()
#     target_id_list = vtk.vtkIdList()

#     point_locator = vtk.vtkPointLocator()
#     point_locator.SetDataSet(segmentation_model)
#     point_locator.BuildLocator()

#     for control_point_index in range(number_of_control_points):
#         is_target = True
#         if not found_start_point and control_point_index == 0:
#             # If no start point found then use the first point as source
#             is_target = False
#         pos = endpoints[control_point_index]
#         # locate the point on the surface
#         point_id = point_locator.FindClosestPoint(pos)
#         if is_target:
#             target_id_list.InsertNextId(point_id)
#         else:
#             source_id_list.InsertNextId(point_id)

#     centerline_filter = vtkvmtk.vtkvmtkPolyDataCenterlines()
#     centerline_filter.SetInputData(segmentation_model)
#     centerline_filter.SetSourceSeedIds(source_id_list)
#     centerline_filter.SetTargetSeedIds(target_id_list)
#     centerline_filter.SetRadiusArrayName("MaximumInscribedSphereRadius")
#     centerline_filter.SetCostFunction('1/R')  # this makes path search prefer go through points with large radius
#     centerline_filter.SetFlipNormals(False)
#     centerline_filter.SetAppendEndPointsToCenterlines(0)
#     centerline_filter.SetSimplifyVoronoi(0)
#     centerline_filter.SetCenterlineResampling(0)
#     centerline_filter.SetResamplingStepLength(1.0)

#     # Voronoi smoothing slightly improves connectivity
#     # Unfortunately, Voronoi smoothing is broken if VMTK is used with VTK9, therefore
#     # disable this feature for now (https://github.com/vmtk/SlicerExtension-VMTK/issues/34)
#     enable_voronoi_smoothing = False
#     centerline_filter.SetSimplifyVoronoi(enable_voronoi_smoothing)

#     centerline_filter.SetCenterlineResampling(0)
#     centerline_filter.SetResamplingStepLength(1.)
#     centerline_filter.Update()

#     if not centerline_filter.GetOutput():
#         raise ValueError("Failed to compute centerline (no output was generated)")
#     centerlines = vtk.vtkPolyData()
#     centerlines.DeepCopy(centerline_filter.GetOutput())

#     if not centerline_filter.GetVoronoiDiagram():
#         raise ValueError("Failed to compute centerline (no Voronoi diagram was generated)")
#     voronoi_diagram = vtk.vtkPolyData()
#     voronoi_diagram.DeepCopy(centerline_filter.GetVoronoiDiagram())

#     # Clean vtkpolydata
#     cleaner = vtk.vtkCleanPolyData()
#     cleaner.SetInputData(centerlines)
#     cleaner.Update()

#     centerlineStripper = vtk.vtkStripper()
#     centerlineStripper.SetInputData(cleaner.GetOutput())
#     centerlineStripper.JoinContiguousSegmentsOn()
#     centerlineStripper.Update()
#     centerlines = centerlineStripper.GetOutput()

#     cleaner = vtk.vtkCleanPolyData()
#     cleaner.SetInputData(voronoi_diagram)
#     cleaner.Update()
#     voronoi_diagram = cleaner.GetOutput()

#     return centerlines, voronoi_diagram