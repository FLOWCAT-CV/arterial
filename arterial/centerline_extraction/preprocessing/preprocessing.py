#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import slicer

import numpy as np

from preprocessing.utils import get_bounding_box_limits_3d

def preprocessing(master_volume_node):
    """
    Performs segmentation of a binary mask using Slicer's segmentEditorWidget to create
    volume model of the segmented bodies. Also, it applies preprocessing of the resulting
    segmentation.
    
    Thresholding is applied to segments the loaded master_volume_node corresponding 
    to a binary nifti. A Gaussian smoothing filter with a standard deviation 
    of 1 mm is applied and the Islands tool is used to remove all islands smaller
    than 10,000 voxels, as well as to split all islands left into different segments.
    The resulting segmentation node is returned for further processing.

    At the moment, we disregard the voxels in the upper 20% of the image, as we are 
    focusing on the more reliably segmented region near the aortic arch, up to the 
    distal end of the ICAs (syph).

    Parameters
    ----------
    master_volume_node : slicer volumeNode
        Master volume node containing the binary nifti loaded onto Slicer.

    Returns
    -------
    segmentation_node : slicer segmentationNode
        Segmentation node containing the predicted segmentation. This will be 
        piped to centerline extraction.
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 80% of the segmentation's bounding box.

    """
    # Set to 0 the voxels in the upper 20% of the bounding box
    masked_volume_array = slicer.util.arrayFromVolume(master_volume_node)
    _, _, _, _, min_is, max_is = get_bounding_box_limits_3d(masked_volume_array)
    masked_volume_array[int(np.round((max_is - min_is) * 0.80)):] = 0
    # Update volume in slicer
    slicer.util.updateVolumeFromArray(master_volume_node, masked_volume_array)

    # Create segmentation node
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes() # only needed for display

    # Create segment editor to get access to effects
    segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
    segment_editor_widget.setMRMLScene(slicer.mrmlScene)
    segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segment_editor_node.SetOverwriteMode(2)
    segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
    # Set the master_volume_node as reference of the segmentation_node
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(master_volume_node)
    # Add new empty segmentation
    _ = segmentation_node.GetSegmentation().AddEmptySegment("Segmentation")
    # Set segmentation_node and master_volume_node in the segment_editor_widget
    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setSourceVolumeNode(master_volume_node)

    # Apply thresholding
    segment_editor_widget.setActiveEffectByName("Threshold")
    effect = segment_editor_widget.activeEffect()
    effect.setParameter("MinimumThreshold", "1")
    effect.setParameter("MaximumThreshold", "1")
    effect.self().onApply()

    # Apply smoothing
    # segment_editor_widget.setActiveEffectByName("Smoothing")
    # effect = segment_editor_widget.activeEffect()
    # effect.setParameter("SmoothingMethod", "GAUSSIAN")
    # effect.setParameter("GaussianStandardDeviationMm", 1)
    # effect.self().onApply()

    # Remove small islands
    segment_editor_widget.setActiveEffectByName("Islands")
    effect = segment_editor_widget.activeEffect()
    effect.setParameter("Operation", "REMOVE_SMALL_ISLANDS")
    effect.setParameter("MinimumSize", 10000)
    effect.self().onApply()

    # Split remaining islands into individual segments
    segment_editor_widget.setActiveEffectByName("Islands")
    effect = segment_editor_widget.activeEffect()
    effect.setParameter("Operation", "SPLIT_ISLANDS_TO_SEGMENTS")
    effect.self().onApply()

    # Clean up
    segment_editor_widget = None
    slicer.mrmlScene.RemoveNode(segment_editor_node)

    # Create closed surface representation of segmentation
    segmentation_node.CreateClosedSurfaceRepresentation()

    return segmentation_node, masked_volume_array