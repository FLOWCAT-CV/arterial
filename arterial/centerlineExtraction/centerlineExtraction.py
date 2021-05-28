import os
import vtk
import slicer
import numpy as np

def centerlineExtraction(caseDir, segmentationNode):
    ''' Exracts centerline using Slicer's VMTK module. Uses auto-endpoint detection.
    Stores radius information corresponding to the model with grown margins. This has to be
    recalculated later on for feature extraction.

    Writes centerlines.vtk containing the vtkPolyData object of the centerline model.

    Arguments:
        - caseDir <str>: path to the directory containing the binary mask. All 
        segmentations will be saved in this dir.
        - segmentationNode2 <slicer segmentationNode>: segmentation node containing the 
        extruded segmentation (with grown margins). This will be piped to centerline extraction.

    Returns:
        
    '''

    if not os.path.isdir(os.path.join(caseDir, "centerlines")): os.mkdir(os.path.join(caseDir, "centerlines"))
    # if not os.path.isdir(os.path.join(caseDir, "segmentations")): os.mkdir(os.path.join(caseDir, "segmentations"))
    if not os.path.isdir(os.path.join(caseDir, "decimatedSegmentations")): os.mkdir(os.path.join(caseDir, "decimatedSegmentations"))

    centerlineIds = []

    print("Beginning centerline extraction. Total number of segments:", segmentationNode.GetSegmentation().GetNumberOfSegments())
    for segmentId in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
        print("Segment", segmentId)

        print("Saving segmentations...")
        # Saving segmentation (undivided)
        surfaceModel = vtk.vtkPolyData()
        segmentationNode.GetClosedSurfaceRepresentation(segmentationNode.GetSegmentation().GetNthSegmentID(segmentId), surfaceModel)
        # # Saving segmentation
        # writer = vtk.vtkPolyDataWriter()
        # writer.SetInputData(surfaceModel)
        # writer.SetFileName(os.path.join(caseDir, "segmentations", f"segmentation{segmentId}.vtk"))
        # writer.Write()
        # Decimating model
        decimator = vtk.vtkDecimatePro()
        decimator.SetTargetReduction(0.7)
        decimator.AddInputData(surfaceModel)
        decimator.Update()
        decimatedSurfaceModel = decimator.GetOutput()
        # Saving decimated model
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(decimatedSurfaceModel)
        writer.SetFileName(os.path.join(caseDir, "decimatedSegmentations", f"decimatedSegmentation{segmentId}.vtk"))
        writer.Write()

        # Set up extract centerline widget
        extractCenterlineWidget = None
        parameterNode = None
        # Instance Extract Centerline Widget
        extractCenterlineWidget = slicer.modules.extractcenterline.widgetRepresentation().self()
        # Set up parameter node
        parameterNode = slicer.mrmlScene.GetSingletonNode("ExtractCenterline", "vtkMRMLScriptedModuleNode")
        extractCenterlineWidget.setParameterNode(parameterNode)
        extractCenterlineWidget.setup()

        # Update from GUI to get segmentationNode as inputSurfaceNode
        extractCenterlineWidget.updateParameterNodeFromGUI()
        # Set network node reference to new empty node
        extractCenterlineWidget._parameterNode.SetNodeReferenceID("InputSurface", segmentationNode.GetID())
        extractCenterlineWidget.ui.inputSegmentSelectorWidget.setCurrentSegmentID(segmentationNode.GetSegmentation().GetNthSegmentID(segmentId))

        print("Automatic endpoint extraction...")
        # Autodetect endpoints
        extractCenterlineWidget.onAutoDetectEndPoints()
        extractCenterlineWidget.updateGUIFromParameterNode()
        print("done")

        print("Relocating endpoints to center of mass of local closest object...")
        # Get volume node array from segmentation node
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode)
        segmentationArray = slicer.util.arrayFromVolume(labelmapVolumeNode)

        # Get affine matrix from segmentation labelMapVolumeNode
        vtkAff = vtk.vtkMatrix4x4()
        aff = np.eye(4)
        labelmapVolumeNode.GetIJKToRASMatrix(vtkAff)
        vtkAff.DeepCopy(aff.ravel(), vtkAff)

        # Get endpoints node
        endpointsNode = slicer.util.getNode(extractCenterlineWidget._parameterNode.GetNodeReferenceID("EndPoints"))
        # Relocate endpoints for robust centerline extraction 
        for idx in range(endpointsNode.GetNumberOfFiducials()):
            endpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(idx))
            newEndpoint = robustEndPointDetection(endpoint, segmentationArray, aff) # Center of mass of closest component method
            endpointsNode.SetNthFiducialPosition(idx, newEndpoint[0],
                                                      newEndpoint[1],
                                                      newEndpoint[2])

        print("done")

        print("Extracting centerline...")
        # Create new Surface model node for the centerline model
        centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        centerlineIds.append(centerlineModelNode.GetID())
        # Set centerline node reference to new empty node
        extractCenterlineWidget._parameterNode.SetNodeReferenceID("CenterlineModel", centerlineModelNode.GetID())
        extractCenterlineWidget.onApplyButton()
        print("done")
        
        print("Checking for floating centerlines (errors)...")
        # Check if all centerlines depart from the same origin. Dismiss the ones that don't, they are most likely floating
        # Get number of cells
        numberOfCells = centerlineModelNode.GetPolyData().GetNumberOfCells()

        # Declare empty arrays
        cellsIdArray = np.ndarray([numberOfCells], dtype=int)
        cellsFirstCoordinateArray = np.ndarray([numberOfCells, 3])

        # Iterate over cells to extract cell IDs, positions and radii. Store lengths of cells
        for cellID in range(numberOfCells):
            cellsIdArray[cellID] = cellID
            cell = vtk.vtkGenericCell()
            centerlineModelNode.GetPolyData().GetCell(cellID, cell)
            cellsFirstCoordinateArray[cellID] = np.matmul(aff, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

        uniqueCellsFirstCoordinateArray, counts = np.unique(cellsFirstCoordinateArray, return_counts=True, axis=0)

        # Get all those that do not start at the startpoint
        removeFloating = []
        for idx in range(len(uniqueCellsFirstCoordinateArray)):
            if idx != np.argmax(counts):
                for idx2 in range(numberOfCells):
                    if (uniqueCellsFirstCoordinateArray[idx] == cellsFirstCoordinateArray[idx2]).all(): removeFloating.append(idx2)

        if len(removeFloating) > 0:
            print("   Found floating centerlines:", removeFloating)
        else:
            print("   No errors found")

        for idx in removeFloating:
            centerlineModelNode.GetPolyData().DeleteCell(idx)

        centerlineModelNode.GetPolyData().RemoveDeletedCells()
        print("done")
        print("    ")

        # Saving centerlines separately
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(centerlineModelNode.GetPolyData())
        writer.SetFileName(os.path.join(caseDir, "centerlines", f"centerlines{segmentId}.vtk"))
        writer.Write()
        
    # print("Uniting all centerline models...")
    # # Initialize the vtkPoints and the vtkCellArray objects
    # cellArray = vtk.vtkCellArray()
    # points = vtk.vtkPoints()
    # finalRadiusArray = np.array([])

    # # Accumulate number of points for all models to define point ids
    # pointsFromPreviousModels = 0

    # # Iterate over all models to store cells, points and radius array as point data
    # for centerlineId in centerlineIds:
    #     # Get vtkPolyData
    #     centerlinePolyData = slicer.mrmlScene.GetNodeByID(centerlineId).GetPolyData()
    #     # Get radius numpy array
    #     radiusArray = vtk.util.numpy_support.vtk_to_numpy(centerlinePolyData.GetPointData().GetArray(0))
        
    #     # Iterate over all cells for each model
    #     for idx in range(centerlinePolyData.GetNumberOfCells()):
    #         polyLine = centerlinePolyData.GetCell(idx)
        
    #         # Iterate over all points in each cell to define new point ids and get corresponding radius array
    #         for idx in range(polyLine.GetNumberOfPoints()):
    #             # Insert point in vtkPoints
    #             points.InsertNextPoint(polyLine.GetPoints().GetPoint(idx))
    #             # Get radius data for each point
    #             finalRadiusArray = np.append(finalRadiusArray, radiusArray[polyLine.GetPointId(idx)])
    #             # Change point id for each point in the cell, taking into account accumulated number of points
    #             polyLine.GetPointIds().SetId(idx, polyLine.GetPointId(idx) + pointsFromPreviousModels)
                
    #         # Insert cell in vtkCellArray
    #         cellArray.InsertNextCell(polyLine)
        
    #     # Update total number of points from previous models
    #     pointsFromPreviousModels += centerlinePolyData.GetNumberOfPoints()
                                    
    # # Store all data in new vtkPolyData
    # finalCenterlinePolyData = vtk.vtkPolyData()
    # finalCenterlinePolyData.SetPoints(points)
    # finalCenterlinePolyData.SetLines(cellArray)
    # finalCenterlinePolyData.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(finalRadiusArray))
    # finalCenterlinePolyData.GetPointData().GetArray(0).SetName("Radius")

    # print("done")
    # print("    ")

    # # Write vtkPolyData to a .vtk file
    # print("Saving vtkPolyData centerline file...")
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetInputData(centerlineModelNode.GetPolyData())
    # writer.SetFileName(os.path.join(caseDir, "centerlines.vtk"))
    # writer.Write()

    # print("done")
    # print("    ")


def robustEndPointDetection(endpoint, segmentation, aff, n=10):
    ''' Relocates automatically detected endpoints to the center of mass of the closest component
    inside a local region around the endpoint (defined by n).

    Takes the endpoint position, converts it to voxel coordinates with the affine matrix, then defines a region  
    of (2 * n) ^ 3 voxels centered around the endpoint. Then components inside the local region are treated 
    as separate objects. The minimum distance from theese objects to the endpoint is computed, and from 
    these, the object with the smallest distance to the endpoint is chosen to compute the centroid, which
    is converted back to RAS with the affine matrix.

    Arguments:
        - endpoint <np.array>: position of the endpoint in RAS coordinates.
        - segmentation <np.array>: numpy array corresponding to the croppedVolumeNode.
        - aff <np.array>: affine matrix corresponding ot he nifti file.
        - n <int>: defines size of the region around the endpoint that is analyzed for this method.

    Returns:
        - newEndpoint <np.array>: new position of the endpoint.

    '''

    from skimage.measure import regionprops, label
    from scipy import ndimage

    # Compute RAS coordinates with affine matrix
    R0, A0, S0 = np.round(np.matmul(np.linalg.inv(aff), np.append(endpoint, 1.0))[:3]).astype(int)
    
    # Mask the segmentation (Only region of interest)
    maskedSegmentation = segmentation[np.max([0, S0 - n]): np.min([segmentation.shape[0], S0 + n]), 
                                      np.max([0, A0 - n]): np.min([segmentation.shape[1], A0 + n]),
                                      np.max([0, R0 - n]): np.min([segmentation.shape[2], R0 + n])]
    
    # Divide into different connected components
    labelMask = label(maskedSegmentation)
    
    labels = np.sort(np.unique(labelMask))
    labels = np.delete(labels, np.where([labels == 0]))
    
    labelMaskOneHot = np.zeros([len(labels), labelMask.shape[0], labelMask.shape[1], labelMask.shape[2]], dtype=np.uint8)
    for idx, label in enumerate(labels):
        labelMaskOneHot[idx][labelMask == label] = 1
        
    invertedLabelMaskOneHot = np.ones_like(labelMaskOneHot) - labelMaskOneHot
    
    # Get distance transform for each and get only closest component
    distanceLabels = np.empty_like(labels, dtype=np.float)
    for idx in range(len(labels)):
        distanceLabels[idx] = ndimage.distance_transform_edt(invertedLabelMaskOneHot[idx])[invertedLabelMaskOneHot.shape[1] // 2][invertedLabelMaskOneHot.shape[2] // 2][invertedLabelMaskOneHot.shape[3] // 2]

    mask = np.zeros_like(segmentation)
    mask[np.max([0, S0 - n]): np.min([segmentation.shape[0], S0 + n]), 
         np.max([0, A0 - n]): np.min([segmentation.shape[1], A0 + n]),
         np.max([0, R0 - n]): np.min([segmentation.shape[2], R0 + n])] = labelMaskOneHot[np.argmin(distanceLabels)]
    
    # Get the centroid of the foregroud region
    properties = regionprops(mask.astype(np.int), mask.astype(np.int))
    centerOfMass = np.array(properties[0].centroid)[[2, 1, 0]]
    
    # Return the new position of the endpoint in RAS coordinates
    return np.matmul(aff, np.append(centerOfMass, 1.0))[:3]