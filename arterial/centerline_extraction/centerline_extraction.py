#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import slicer
import vtk

import numpy as np
import nibabel as nib

from utils import aortic_arch_endpoint_check, robust_end_point_detection, inspect_circular_centerlines

def centerline_extraction(case_dir, segmentation_node, masked_volume_array):
    """
    Extracts centerline using Slicer's VMTK module. Processes all segments in the input 
    segmentation_node individually to generate independent centerline models for each 
    segmentation. 

    Uses VMTK's auto-endpoint detection optimized for improved robustness.
    
    Writes centerlines{idx}.vtk for idx in range(N), where N is the number of independent 
    segments, containing the vtkPolyData object of the centerline models, for the number 
    of present segments, as well as a decimated surface model, (decimatedSegmnetation{idx}.vtk) 
    reduced by 70% from the original amount of triangles for speed.

    Saves centerlines and surface models as:

    >>> case_dir/centerlines/centerlines{idx}.vtk
    >>> case_dir/segmentations/segmentation{idx}.vtk

    Paremeters
    ----------
    case_dir : string or path-like object 
        Path to the directory containing the binary mask nifti. All segmentations will be 
        saved in this directory.
    segmentation_node : slicer segmentation_node
        Segmentation node containing one or more separate segments.
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 80% of the segmentation's bounding box.

    Returns
    -------
    
    """
    # Create directories to store centerline and segmentation volume models
    if not os.path.isdir(os.path.join(case_dir, "centerlines")): os.mkdir(os.path.join(case_dir, "centerlines"))
    if not os.path.isdir(os.path.join(case_dir, "segmentations")): os.mkdir(os.path.join(case_dir, "segmentations"))

    # Get the affine matrix
    aff = nib.load(os.path.join(case_dir, "{}_segmentation.nii.gz".format(os.path.basename(case_dir)))).affine

    print("Beginning centerline extraction. Total number of segments: {}".format(segmentation_node.GetSegmentation().GetNumberOfSegments()))
    # Now, we iterate over all segments to perform centerline extraction separately
    for segment_id in range(segmentation_node.GetSegmentation().GetNumberOfSegments()):
        print("Segment {}".format(segment_id))
        print("Saving segmentations...")
        # Saving segmentation (undivided)
        surface_model = vtk.vtkPolyData()
        segmentation_node.GetClosedSurfaceRepresentation(segmentation_node.GetSegmentation().GetNthSegmentID(segment_id), surface_model)

        # Decimating model
        decimator = vtk.vtkDecimatePro()
        decimator.SetTargetReduction(0.7)
        decimator.AddInputData(surface_model)
        decimator.Update()
        decimated_surface_model = decimator.GetOutput()
        # Saving decimated model
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(decimated_surface_model)
        writer.SetFileName(os.path.join(case_dir, "segmentations", f"segmentation{segment_id}.vtk"))
        writer.Write()

        # Extract the centerline of the segment_id segment
        centerline_poly_data = extract_centerline(segmentation_node, segment_id, masked_volume_array, aff)
        # Saving centerlines separately
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(centerline_poly_data)
        writer.SetFileName(os.path.join(case_dir, "centerlines", f"centerlines{segment_id}.vtk"))
        writer.Write()

        # For the largest segment, we check the existence of circular centerlines
        if segment_id == 0:
            centerline_poly_data = inspect_circular_centerlines(centerline_poly_data, decimated_surface_model, segmentation_node, segment_id, aff)

        # Overwriting centerlines separately after circular centerline inspection and extraction
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(centerline_poly_data)
        writer.SetFileName(os.path.join(case_dir, "centerlines", f"centerlines{segment_id}.vtk"))
        writer.Write()

def extract_centerline(segmentation_node, segment_id, masked_volume_array, aff):  
    """ 
    Extracts the centerline model node from the segmentation_node for the corresponding 
    segment_id using Slicer's VMTK extension.

    Parameters
    ---------- 
    segmentation_node : vtkMRMLSegmentationNode
        MRML segmentation node.
    segment_id : integer
        Segment identifier in the segmentation_node.
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 80% of the segmentation's bounding box.
    aff : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------
    centerline_poly_data : vtk.vtkPolyData
        Centerline model in vtkPolyData form for segment_id segment.

    """
    # Set up extract centerline widget
    extract_centerline_widget = None
    parameter_node = None
    # Instance Extract Centerline Widget
    extract_centerline_widget = slicer.modules.extractcenterline.widgetRepresentation().self()
    # Set up parameter node
    parameter_node = slicer.mrmlScene.GetSingletonNode("ExtractCenterline", "vtkMRMLScriptedModuleNode")
    extract_centerline_widget.setParameterNode(parameter_node)
    extract_centerline_widget.setup()

    # Update from GUI to get segmentation_node as inputSurfaceNode
    extract_centerline_widget.updateParameterNodeFromGUI()
    # Set network node reference to new empty node
    extract_centerline_widget._parameterNode.SetNodeReferenceID("InputSurface", segmentation_node.GetID())
    extract_centerline_widget.ui.inputSegmentSelectorWidget.setCurrentSegmentID(segmentation_node.GetSegmentation().GetNthSegmentID(segment_id))

    print("Automatic endpoint extraction...")
    # Autodetect endpoints
    extract_centerline_widget.onAutoDetectEndPoints()
    extract_centerline_widget.updateGUIFromParameterNode()

    # Get volume node array from segmentation node
    label_map_volume_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentation_node, label_map_volume_node)
    segmentation_array = slicer.util.arrayFromVolume(label_map_volume_node)

    # Get affine matrix from segmentation label_map_volume_node
    vtk_aff = vtk.vtkMatrix4x4()
    aff_eye = np.eye(4)
    label_map_volume_node.GetIJKToRASMatrix(vtk_aff)
    vtk_aff.DeepCopy(aff_eye.ravel(), vtk_aff)

    # Get endpoints node
    endpointsNode = slicer.util.getNode(extract_centerline_widget._parameterNode.GetNodeReferenceID("EndPoints"))

    # Check if both ends of the aortic arch have at least one endpoint
    if segment_id == 0:
        endpointsNode = aortic_arch_endpoint_check(endpointsNode, masked_volume_array, aff)

    print("Relocating endpoints for robust centerline extraction...")
    # Relocate endpoints for robust centerline extraction 
    for idx in range(endpointsNode.GetNumberOfControlPoints()):
        endpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(idx))
        newEndpoint = robust_end_point_detection(endpoint, segmentation_array, aff_eye) # Center of mass of closest component method
        endpointsNode.SetNthControlPointPosition(idx, newEndpoint[0],
                                                      newEndpoint[1],
                                                      newEndpoint[2])

    print("Extracting centerline...")
    # Create new Surface model node for the centerline model
    centerline_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    # Set centerline node reference to new empty node
    extract_centerline_widget._parameterNode.SetNodeReferenceID("CenterlineModel", centerline_model_node.GetID())
    extract_centerline_widget.onApplyButton()

    print("Checking for floating centerlines (errors)...")
    # Check if all centerlines depart from the same origin. Dismiss the ones that don't, they are most likely floating
    centerline_poly_data = centerline_model_node.GetPolyData()

    # Declare empty arrays
    cells_id_array = np.ndarray([centerline_poly_data.GetNumberOfCells()], dtype=int)
    cell_first_coordinate_array = np.ndarray([centerline_poly_data.GetNumberOfCells(), 3])

    # Iterate over cells to extract cell IDs, positions and radii. Store lengths of cells
    for cell_id in range(centerline_poly_data.GetNumberOfCells()):
        cells_id_array[cell_id] = cell_id
        cell = vtk.vtkGenericCell()
        centerline_poly_data.GetCell(cell_id, cell)
        cell_first_coordinate_array[cell_id] = np.matmul(aff, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

    uniquecell_first_coordinate_array, counts = np.unique(cell_first_coordinate_array, return_counts=True, axis=0)

    # Get all those that do not start at the startpoint
    remove_floating = []
    for idx in range(len(uniquecell_first_coordinate_array)):
        if idx != np.argmax(counts):
            for idx2 in range(centerline_poly_data.GetNumberOfCells()):
                if (uniquecell_first_coordinate_array[idx] == cell_first_coordinate_array[idx2]).all(): remove_floating.append(idx2)

    # Finally, check if there are any floating centerlines
    if len(remove_floating) > 0:
        print("Found floating centerlines:", remove_floating)
    else:
        print("No errors found")

    for idx in remove_floating:
        centerline_poly_data.DeleteCell(idx)

    centerline_poly_data.RemoveDeletedCells()
    
    return centerline_poly_data