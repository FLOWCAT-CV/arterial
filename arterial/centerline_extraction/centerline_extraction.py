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
    centerlines, voronoi, endpoints_json = centerline_computation_logic.extract_centerline(segmentation_model, segmentation_array, segmentation_affine, is_first_model)
    startpoint = endpoints_json["markups"][0]["controlPoints"][0]["position"]
    centerlines = clean_centerline(centerlines, startpoint=startpoint)

    print("Number of cells after centerline cleaning:", centerlines.GetNumberOfCells())

    if centerlines.GetNumberOfCells() == 0:
        return None, None, None
    else:
        return centerlines, voronoi, endpoints_json