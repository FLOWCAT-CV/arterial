#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.centerline_extraction.utils import CenterlineComputationLogic, clean_centerline

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
    centerlines, network, voronoi, endpoints_json = centerline_computation_logic.extract_centerline_full_cta(segmentation_model, segmentation_array, segmentation_affine, is_first_model)
    
    if centerlines is None: # True if no endpoints are found
        return None, None, None, None
    
    startpoint = endpoints_json["markups"][0]["controlPoints"][0]["position"]
    if centerlines.GetNumberOfCells() == 0 or centerlines.GetNumberOfPoints() == 0:
        pass
    else:
        centerlines = clean_centerline(centerlines, startpoint=startpoint)
        print("Number of cells after centerline cleaning:", centerlines.GetNumberOfCells())
    
    # For the intended use of the extract_centerlines_full_cta function, returning one single centerline cell in the first model is completely
    # unexpected. From experience, this tends to happen in cases where the mesh generation from the segmentation of the bottom-most part of the aortic arch 
    # is irregular, leading to a faulty network extraction and consequently, a faulty endpoint extraction.
    # Therefore, we return None if the number of centerline cells is 1 and is_first_model is True. In that case, the contingency would be to reprocess the mesh
    # generation preprocessing step, removing the bottom-most part of the aortic arch segmentation mesh and repeating the centerline extraction process
    if is_first_model and centerlines.GetNumberOfCells() == 1:
        return None, None, None, None

    if centerlines.GetNumberOfCells() == 0:
        return None, None, None, None
    else:
        return centerlines, network, voronoi, endpoints_json
    
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
        return None, None
    
    # startpoint = endpoints[0]
    # if centerlines.GetNumberOfCells() == 0 or centerlines.GetNumberOfPoints() == 0:
    #     pass
    # else:
    #     centerlines = clean_centerline(centerlines, startpoint=startpoint)
    #     print("Number of cells after centerline cleaning:", centerlines.GetNumberOfCells())
        
    if centerlines.GetNumberOfCells() == 0:
        return None, None
    else:
        return centerlines, voronoi