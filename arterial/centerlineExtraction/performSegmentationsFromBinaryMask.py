import slicer
import numpy as np

def performSegmentationsFromBinaryMask(masterVolumeNode):
    ''' Performs segmentation of a binary mask using Slicer's segmentEditorWidget. 
    Thresholding is applied to segments the loaded masterVolumeNode (corresponding) 
    to a binary NIfTI. Then, a Gaussian smoothing filter with a standard deviation 
    of 0.5 mm is applied and the Islands tool is used to remove all islands smaller
    than 10,000 voxels, as well as to split all islands left into different segments.
    The resulting segmentation node is returned for further processing.

    At the moment, we disregard the voxels in the upper 20% of the image, as we are 
    focusing on the more reliably segmented region near the aortic arch, up to the 
    distal end of the ICAs (syph).

    Arguments:
        - masterVolumeNode <slicer volumeNode>: master volume node containing the binary 
        nifti loaded onto Slicer.

    Returns:
        - segmentationNode <slicer segmentationNode>: segmentation node containing the 
        resampled segmentation. This will be piped to centerline extraction.
    '''

    print("Masking segmentation (we only keep the lower 80% of the image for robustness)...")
    # Set to 0 the voxels in the upper 20% of the bounding box
    maskedVolumeArray = slicer.util.arrayFromVolume(masterVolumeNode)
    _, _, _, _, minIS, maxIS = bbox_3D(maskedVolumeArray)
    maskedVolumeArray[int(np.round((maxIS - minIS) * 0.80)):] = 0
    slicer.util.updateVolumeFromArray(masterVolumeNode, maskedVolumeArray)
    print("done")

    print("Performing segmentation of resampled volume")
    # Create segmentation node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.CreateDefaultDisplayNodes() # only needed for display
    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorNode.SetOverwriteMode(2)
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
    _ = segmentationNode.GetSegmentation().AddEmptySegment("Segmentation")
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)

    # Thresholding
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold", "1")
    effect.setParameter("MaximumThreshold", "1")
    effect.self().onApply()

    # Smoothing
    segmentEditorWidget.setActiveEffectByName("Smoothing")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("SmoothingMethod", "GAUSSIAN")
    effect.setParameter("GaussianStandardDeviationMm", 1)
    effect.self().onApply()

    # Remove small islands
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "REMOVE_SMALL_ISLANDS")
    effect.setParameter("MinimumSize", 1000)
    effect.self().onApply()

    # Split large islands into individual segments
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "SPLIT_ISLANDS_TO_SEGMENTS")
    effect.self().onApply()

    # Clean up
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    # Create closed surface representation of segmentation
    segmentationNode.CreateClosedSurfaceRepresentation()

    return segmentationNode

def bbox_3D(img):
    ''' Computes bounding box (only z axis) of a numpy array (expects an array with zeros as background).

    Arguments:
        - img <numpy array>: 3D numpy binary (0, 1) array.

    Returns:
        - minLR <int>: lower bound on axis x, LR (in voxel coordinates).
        - maxLR <int>: upper bound on axis x, LR (in voxel coordinates).
        - minPA <int>: lower bound on axis y, PA (in voxel coordinates).
        - maxPA <int>: upper bound on axis y, PA (in voxel coordinates).
        - minIS <int>: lower bound on axis z, IS (in voxel coordinates).
        - maxIS <int>: upper bound on axis z, IS (in voxel coordinates).
    '''
    LR = np.any(img, axis=(0, 1))
    PA = np.any(img, axis=(0, 2))
    IS = np.any(img, axis=(1, 2))

    minLR, maxLR = np.where(LR)[0][[0, -1]]
    minPA, maxPA = np.where(PA)[0][[0, -1]]
    minIS, maxIS = np.where(IS)[0][[0, -1]]

    return minLR, maxLR, minPA, maxPA, minIS, maxIS