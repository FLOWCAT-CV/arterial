#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import slicer

import numpy as np
import nibabel as nib

from preprocessing.utils import get_bounding_box_limits_3d

def preprocessing(case_dir, master_volume_node, fast_segmentation = False):
    """
    Performs segmentation of a binary mask using Slicer's segmentEditorWidget to create
    volume model of the segmented bodies. Also, it applies preprocessing of the resulting
    segmentation.
    
    Thresholding is applied to segments the loaded master_volume_node corresponding 
    to a binary nifti. A Gaussian smoothing filter with a standard deviation 
    of 1 mm is applied and the Islands tool is used to remove all islands smaller
    than 10,000 voxels (that is, with a voxel size of 0.43 * 0.43 * 0.4 mm^3), 
    as well as to split all islands left into different segments.
    The resulting segmentation node is returned for further processing.

    At the moment, we disregard the voxels in the upper 15% of the image, as we are 
    focusing on the more reliably segmented region near the aortic arch, up to the 
    distal end of the ICAs (syph).

    Parameters
    ----------
    case_dir : string or path-like object 
        Path to the directory containing the binary mask nifti. All segmentations will be 
        saved in this directory.
    master_volume_node : slicer volumeNode
        Master volume node containing the binary nifti loaded onto Slicer.

    Returns
    -------
    segmentation_node : slicer segmentationNode
        Segmentation node containing the predicted segmentation. This will be 
        piped to centerline extraction.
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 85% of the segmentation's bounding box.

    """
    # Set to 0 the voxels in the upper 20% of the bounding box
    masked_volume_array = slicer.util.arrayFromVolume(master_volume_node)
    # If fast segmentation was performed, we ignore the voxels in the upper 15% 
    # of the foreground bounding box in the binary map
    if fast_segmentation:
        _, _, _, _, min_is, max_is = get_bounding_box_limits_3d(masked_volume_array)
        masked_volume_array[int(np.round((max_is - min_is) * 0.85)):] = 0
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

    # Choose the number of voxels for small island threshold
    # Empirically, we found that 10000 is a good threshold for a 
    # voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    # Get voxel size from image
    voxel_size = np.prod(nib.load(os.path.join(case_dir, "{}_segmentation.nii.gz".format(os.path.basename(case_dir)))).header["pixdim"][1:4])
    # Compute approximate number of voxels
    number_of_voxels_threshold = round(10000 * (reference_voxel_size / voxel_size))
    # Remove small islands
    segment_editor_widget.setActiveEffectByName("Islands")
    effect = segment_editor_widget.activeEffect()
    effect.setParameter("Operation", "REMOVE_SMALL_ISLANDS")
    effect.setParameter("MinimumSize", number_of_voxels_threshold)
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