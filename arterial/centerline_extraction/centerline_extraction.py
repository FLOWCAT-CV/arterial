#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import slicer
import vtk

import numpy as np
import nibabel as nib

from utils import aortic_arch_endpoint_check, robust_end_point_detection, ica_endpoint_check, inspect_circular_centerlines, compute_frenet_serret, compute_curvature_and_torsion

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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
    affine = nib.load(os.path.join(case_dir, "{}_vessel_segmentation.nii.gz".format(os.path.basename(case_dir)))).affine

    print("Beginning centerline extraction. Total number of segments: {}".format(segmentation_node.GetSegmentation().GetNumberOfSegments()))
    # Now, we iterate over all segments to perform centerline extraction separately
    for segment_id in range(segmentation_node.GetSegmentation().GetNumberOfSegments()):
        try:
            with time_limit(3 * 60):
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
                surface_model = decimator.GetOutput()
                # We can to compute the normals_filter for all mesh triangles to ensure correct orientation
                normals_filter = vtk.vtkPolyDataNormals()
                normals_filter.SetInputData(surface_model)
                normals_filter.SetFeatureAngle(80)
                normals_filter.AutoOrientNormalsOn()
                normals_filter.UpdateInformation()
                normals_filter.Update()
                surface_model = normals_filter.GetOutput()

                # Saving decimated model
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileVersion(42)
                writer.SetInputData(surface_model)
                writer.SetFileName(os.path.join(case_dir, "segmentations", f"vessel_segmentation_{segment_id}.vtk"))
                writer.Write()

                # Extract the centerline of the segment_id segment
                centerline_poly_data = extract_centerline(segmentation_node, segment_id, masked_volume_array, affine)

                # # Decimate centerline model (at the moment we do not use it)
                # decimator = vtk.vtkDecimatePolylineFilter()
                # decimator.SetTargetReduction(0.5)
                # decimator.AddInputData(centerline_poly_data)
                # decimator.Update()
                # centerline_poly_data = decimator.GetOutput()

                # Saving centerlines separately
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileVersion(42)
                writer.SetInputData(centerline_poly_data)
                writer.SetFileName(os.path.join(case_dir, "centerlines", f"vessel_centerlines_{segment_id}.vtk"))
                writer.Write()

                # For the largest segment, we check the existence of circular centerlines. Needs further testing
                # if segment_id == 0:
                #     centerline_poly_data = inspect_circular_centerlines(case_dir, centerline_poly_data, surface_model, segmentation_node, segment_id, affine)

                # Smooth centerline model
                smoothing_filter = vtk.vtkSmoothPolyDataFilter()
                smoothing_filter.SetInputData(centerline_poly_data)
                smoothing_filter.SetNumberOfIterations(50)
                smoothing_filter.SetRelaxationFactor(0.1)
                smoothing_filter.FeatureEdgeSmoothingOff()
                smoothing_filter.BoundarySmoothingOn()
                smoothing_filter.Update()
                centerline_poly_data = smoothing_filter.GetOutput()

                # # Decimate centerline model
                # decimator = vtk.vtkDecimatePolylineFilter()
                # decimator.SetTargetReduction(0.8)
                # decimator.AddInputData(centerline_poly_data)
                # decimator.Update()
                # centerline_poly_data = decimator.GetOutput()

                # Compute Frenet-Serret frame vectors at each point
                centerline_poly_data = compute_frenet_serret(centerline_poly_data)
                # Compute curvature and smoothed curvature
                centerline_poly_data = compute_curvature_and_torsion(centerline_poly_data)

                # Overwriting centerlines separately after circular centerline inspection and extraction
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileVersion(42)
                writer.SetInputData(centerline_poly_data)
                writer.SetFileName(os.path.join(case_dir, "centerlines", f"vessel_centerlines_{segment_id}.vtk"))
                writer.Write()
        except TimeoutException as e:
            print("Timed out for {}.".format(segment_id))

def intracranial_centerline_extraction(case_dir, segmentation_node, masked_volume_array):
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
    affine = nib.load(os.path.join(case_dir, "{}_intracranial_vessel_segmentation.nii.gz".format(os.path.basename(case_dir)))).affine

    print("Beginning centerline extraction. Total number of segments: {}".format(segmentation_node.GetSegmentation().GetNumberOfSegments()))
    # Now, we iterate over all segments to perform centerline extraction separately
    for segment_id in range(segmentation_node.GetSegmentation().GetNumberOfSegments()):
        try:
            with time_limit(3 * 60):
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
                surface_model = decimator.GetOutput()
                # We can to compute the normals_filter for all mesh triangles to ensure correct orientation
                normals_filter = vtk.vtkPolyDataNormals()
                normals_filter.SetInputData(surface_model)
                normals_filter.SetFeatureAngle(80)
                normals_filter.AutoOrientNormalsOn()
                normals_filter.UpdateInformation()
                normals_filter.Update()
                surface_model = normals_filter.GetOutput()

                # Saving decimated model
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileVersion(42)
                writer.SetInputData(surface_model)
                writer.SetFileName(os.path.join(case_dir, "segmentations", f"intracranial_vessel_segmentation_{segment_id}.vtk"))
                writer.Write()

                # Extract the centerline of the segment_id segment
                centerline_poly_data = extract_centerline(segmentation_node, segment_id, masked_volume_array, affine, intracranial = True)

                # # Decimate centerline model (at the moment we do not use it)
                # decimator = vtk.vtkDecimatePolylineFilter()
                # decimator.SetTargetReduction(0.5)
                # decimator.AddInputData(centerline_poly_data)
                # decimator.Update()
                # centerline_poly_data = decimator.GetOutput()

                # Saving centerlines separately
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileVersion(42)
                writer.SetInputData(centerline_poly_data)
                writer.SetFileName(os.path.join(case_dir, "centerlines", f"intracranial_vessel_centerlines_{segment_id}.vtk"))
                writer.Write()

                # For the largest segment, we check the existence of circular centerlines. Needs further testing
                # if segment_id == 0:
                #     centerline_poly_data = inspect_circular_centerlines(case_dir, centerline_poly_data, surface_model, segmentation_node, segment_id, affine)

                # Smooth centerline model
                smoothing_filter = vtk.vtkSmoothPolyDataFilter()
                smoothing_filter.SetInputData(centerline_poly_data)
                smoothing_filter.SetNumberOfIterations(50)
                smoothing_filter.SetRelaxationFactor(0.1)
                smoothing_filter.FeatureEdgeSmoothingOff()
                smoothing_filter.BoundarySmoothingOn()
                smoothing_filter.Update()
                centerline_poly_data = smoothing_filter.GetOutput()

                # # Decimate centerline model
                # decimator = vtk.vtkDecimatePolylineFilter()
                # decimator.SetTargetReduction(0.8)
                # decimator.AddInputData(centerline_poly_data)
                # decimator.Update()
                # centerline_poly_data = decimator.GetOutput()

                # Compute Frenet-Serret frame vectors at each point
                centerline_poly_data = compute_frenet_serret(centerline_poly_data)
                # Compute curvature and smoothed curvature
                centerline_poly_data = compute_curvature_and_torsion(centerline_poly_data)

                # Overwriting centerlines separately after circular centerline inspection and extraction
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileVersion(42)
                writer.SetInputData(centerline_poly_data)
                writer.SetFileName(os.path.join(case_dir, "centerlines", f"intracranial_vessel_centerlines_{segment_id}.vtk"))
                writer.Write()
        except TimeoutException as e:
            print("Timed out for {}.".format(segment_id))

def extract_centerline(segmentation_node, segment_id, masked_volume_array, affine, intracranial = False):  
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
    affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.
    intracranial: bool, default: False
        If not intracranial, checks that the AA endpoints are well-placed. If intracranial, 
        it checks that both ICAs have endpoints and that the startpoint is the left ICA.

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
    endpoints_node = slicer.util.getNode(extract_centerline_widget._parameterNode.GetNodeReferenceID("EndPoints"))

    if intracranial:
        if segment_id == 0:
            pass
            # endpoints_node = ica_endpoint_check(endpoints_node, masked_volume_array, affine)
    else:
        # Check if both ends of the aortic arch have at least one endpoint
        if segment_id == 0:
            endpoints_node = aortic_arch_endpoint_check(endpoints_node, masked_volume_array, affine)

    print("Relocating endpoints for robust centerline extraction...")
    # Relocate endpoints for robust centerline extraction 
    for idx in range(endpoints_node.GetNumberOfControlPoints()):
        endpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(idx))
        if idx == 0:
            new_endpoint = robust_end_point_detection(endpoint, segmentation_array, aff_eye, n = 30) # Center of mass of closest component method
        else:
            new_endpoint = robust_end_point_detection(endpoint, segmentation_array, aff_eye) # Center of mass of closest component method
        endpoints_node.SetNthControlPointPosition(idx, new_endpoint[0],
                                                       new_endpoint[1],
                                                       new_endpoint[2])

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
        cell_first_coordinate_array[cell_id] = np.matmul(affine, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

    unique_cell_first_coordinate_array, counts = np.unique(cell_first_coordinate_array, return_counts=True, axis=0)

    # Get all those that do not start at the startpoint
    remove_floating = []
    for idx in range(len(unique_cell_first_coordinate_array)):
        if idx != np.argmax(counts):
            for idx2 in range(centerline_poly_data.GetNumberOfCells()):
                if (unique_cell_first_coordinate_array[idx] == cell_first_coordinate_array[idx2]).all(): remove_floating.append(idx2)

    # Finally, check if there are any floating centerlines
    if len(remove_floating) > 0:
        print("Found floating centerlines:", remove_floating)
    else:
        print("No errors found")

    for idx in remove_floating:
        centerline_poly_data.DeleteCell(idx)

    centerline_poly_data.RemoveDeletedCells()
    
    return centerline_poly_data

def thrombus_centerline_extraction(case_dir, segmentation_node):
    """
    Extracts thrombus centerline using Slicer's VMTK module. Processes all segments in the input 
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

    Returns
    -------
    
    """
    # Create directories to store centerline and segmentation volume models
    if not os.path.isdir(os.path.join(case_dir, "centerlines")): os.mkdir(os.path.join(case_dir, "centerlines"))
    if not os.path.isdir(os.path.join(case_dir, "segmentations")): os.mkdir(os.path.join(case_dir, "segmentations"))

    # Get the affine matrix
    affine = nib.load(os.path.join(case_dir, "{}_thrombus_segmentation.nii.gz".format(os.path.basename(case_dir)))).affine

    print("Beginning centerline extraction. Total number of segments: {}".format(segmentation_node.GetSegmentation().GetNumberOfSegments()))

    # Now, we iterate over all segments to perform centerline extraction separately
    for segment_id in range(segmentation_node.GetSegmentation().GetNumberOfSegments()):        
        print("Segment {}".format(segment_id))
        print("Saving thrombus segmentations...")

        # Saving segmentation (undivided)
        surface_model = vtk.vtkPolyData()
        segmentation_node.GetClosedSurfaceRepresentation(segmentation_node.GetSegmentation().GetNthSegmentID(segment_id), surface_model)

        # Decimating model
        decimator = vtk.vtkDecimatePro()
        decimator.SetTargetReduction(0.5)
        decimator.AddInputData(surface_model)
        decimator.Update()
        surface_model = decimator.GetOutput()
        # We can to compute the normals_filter for all mesh triangles to ensure correct orientation
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(surface_model)
        normals_filter.SetFeatureAngle(80)
        normals_filter.AutoOrientNormalsOn()
        normals_filter.UpdateInformation()
        normals_filter.Update()
        surface_model = normals_filter.GetOutput()
        # Saving decimated model
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(surface_model)
        writer.SetFileName(os.path.join(case_dir, "segmentations", f"thrombus_segmentation_{segment_id}.vtk"))
        writer.Write()

        # Extract the centerline of the segment_id segment
        centerline_poly_data = extract_thrombus_centerline(segmentation_node, segment_id, affine)

        # Saving centerlines separately
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(centerline_poly_data)
        writer.SetFileName(os.path.join(case_dir, "centerlines", f"thrombus_centerlines_{segment_id}.vtk"))
        writer.Write()

def extract_thrombus_centerline(segmentation_node, segment_id, affine):  
    """ 
    Extracts the centerline model from the thombus from the segmentation_node for the corresponding 
    segment_id using Slicer's VMTK extension.

    Parameters
    ---------- 
    segmentation_node : vtkMRMLSegmentationNode
        MRML segmentation node.
    segment_id : integer
        Segment identifier in the segmentation_node.
    affine : numpy.array or array-like object. Shape: 4 x 4
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

    # Get affine matrix from segmentation label_map_volume_node
    vtk_aff = vtk.vtkMatrix4x4()
    aff_eye = np.eye(4)
    label_map_volume_node.GetIJKToRASMatrix(vtk_aff)
    vtk_aff.DeepCopy(aff_eye.ravel(), vtk_aff)

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
        cell_first_coordinate_array[cell_id] = np.matmul(affine, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

    unique_cell_first_coordinate_array, counts = np.unique(cell_first_coordinate_array, return_counts=True, axis=0)

    # Get all those that do not start at the startpoint
    remove_floating = []
    for idx in range(len(unique_cell_first_coordinate_array)):
        if idx != np.argmax(counts):
            for idx2 in range(centerline_poly_data.GetNumberOfCells()):
                if (unique_cell_first_coordinate_array[idx] == cell_first_coordinate_array[idx2]).all(): remove_floating.append(idx2)

    # Finally, check if there are any floating centerlines
    if len(remove_floating) > 0:
        print("Found floating centerlines:", remove_floating)
    else:
        print("No errors found")

    for idx in remove_floating:
        centerline_poly_data.DeleteCell(idx)

    centerline_poly_data.RemoveDeletedCells()
    
    return centerline_poly_data