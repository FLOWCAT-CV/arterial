import os
import vtk
import slicer
import numpy as np

import nibabel as nib

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from time import time

def centerlineExtraction(casePath, segmentationNode, maskedVolumeArray):
    ''' Extracts centerline using Slicer's VMTK module. Processes all segments in the input segmentationNode 
    individually to generate independent centerline models for each segmentation. 

    Uses VMTK's auto-endpoint detection optimized for improved robustness.
    
    Writes centerlines{idx}.vtk containing the vtkPolyData object of the centerline models, for the
    number of present segments, as well as a decimated surface model, (decimatedSegmnetation{idx}.vtk) reduced 
    by 70% from the original amount of triangles.

    Arguments:
        - caseDir <str>: path to the directory containing the binary mask. All 
        segmentations will be saved in this dir.
        - segmentationNode <slicer segmentationNode>: segmentation node containing separate segments
        resulting from `performSegmentationFromBinaryMask.py`.

    Returns:
        
    '''

    caseDir = os.path.abspath(os.path.dirname(casePath))

    if not os.path.isdir(os.path.join(caseDir, "centerlines")): os.mkdir(os.path.join(caseDir, "centerlines"))
    # if not os.path.isdir(os.path.join(caseDir, "segmentations")): os.mkdir(os.path.join(caseDir, "segmentations"))
    if not os.path.isdir(os.path.join(caseDir, "decimatedSegmentations")): os.mkdir(os.path.join(caseDir, "decimatedSegmentations"))

    # Get the affine matrix
    aff = nib.load(casePath).affine

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
        decimator.SetTargetReduction(0.8)
        decimator.AddInputData(surfaceModel)
        decimator.Update()
        decimatedSurfaceModel = decimator.GetOutput()
        # Saving decimated model
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(decimatedSurfaceModel)
        writer.SetFileName(os.path.join(caseDir, "decimatedSegmentations", f"decimatedSegmentation{segmentId}.vtk"))
        writer.Write()

        start0 = time()
        # Extract the centerline of the segmentId segment
        centerlinePolyData = extractCenterline(segmentationNode, segmentId, maskedVolumeArray, aff)
        # Saving centerlines separately
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(centerlinePolyData)
        writer.SetFileName(os.path.join(caseDir, "centerlines", f"centerlines{segmentId}.vtk"))
        writer.Write()
        start1 = time()
        print(f"Centerline extraction took {start1 - start0} s")
        # For the largest segment, we check the existence of circular centerlines
        if segmentId == 0:
            centerlinePolyData = inspectCircularCenterlines(centerlinePolyData, decimatedSurfaceModel, segmentationNode, segmentId, aff)
        start2 = time()
        print(f"Circular centerline extraction took {start2 - start1} s")

        # Saving centerlines separately
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(centerlinePolyData)
        writer.SetFileName(os.path.join(caseDir, "centerlines", f"centerlines{segmentId}.vtk"))
        writer.Write()


def robustEndPointDetection(endpoint, segmentation, aff, n=15):
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

def AAEndpointCheck(endpointsNode, maskedVolumeArray, aff):
    ''' Checks that both ens of the AA, if present, have one associated endpoint.
    To do that, it looks at the bottom slice of the volume and analyzes the presence 
    of large connected components. Once it has recognized all large connected components,
    it checks if any endpoint is at an Euclidean distance of less than 30 mm with respect to the
    center of mass of the bottom islands. If none are found, 
    
    Arguments:
        - endpointsNode <vtkMRMLMarkupsFiducialNode>: MRML node with all endpoints from the automatic endpoint detection.
        - maskedVolumeArray <np.array>: binary numpy array corresponding to the segmented volume.
        - aff <np.array>: affine matrix from RAS to ijk tranformation.

    Returns:
        - endpointsNode <vtkMRMLMarkupsFiducialNode>: new MRML node with all endpoints from the automatic endpoint detection.
    '''
    from skimage.measure import regionprops, label
    
    thresholdCounts = 500 # For AA island validation (number of foreground voxels in the bottom slice)
    thresholdDistance = 50 # For AA endpoints check (distance from bottom slice)
    
    labelMask = label(maskedVolumeArray[0])
    properties = regionprops(labelMask.astype(np.int), labelMask.astype(np.int))
    
    _, counts = np.unique(labelMask, return_counts=True)
    deleteIdx = []
    for idx, count in enumerate(counts):
        if count < thresholdCounts:
            deleteIdx.append(idx - 1)
    
    properties = list(np.delete(properties, deleteIdx))
    
    # Access the coordinates of centroids
    centroids = np.zeros(shape = (len(properties), 3))
    for idx, prop in enumerate(properties):
        centroids[idx] = np.matmul(aff, np.append(np.array(prop.centroid)[[1, 0]], [1.0, 1.0]))[:3]

    for idx in range(endpointsNode.GetNumberOfFiducials()):
        endpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(idx))
        deleteIdx = None
        for idx, centroid in enumerate(centroids):
            if np.linalg.norm(centroid - endpoint) < thresholdDistance: # Threshold at 30 mm
                deleteIdx = idx
        if deleteIdx is not None:
            centroids = np.delete(centroids, deleteIdx, axis=0)

    if len(centroids) > 0:
        print(f"{len(centroids)} AA islands do not have associated endpoints")
        for centroid in centroids:
            print("Adding endpoint at", centroid)
            print()
            endpointsNode.AddFiducialFromArray(np.array(centroid))
            
    # Select distal AA endpoint as startpoint (in some cases, the distal LSA endpoint is closer to the origin)
    # The criteria will be to choose the AA endpoint (at < 30 mm from bottom slice) that is closest to the reference point
    # Check every other point's distance to origin (ijk)
    print("Ensuring startpoint is at distal AA end")
    distanceToRASOrigin = []
    for idx in range(endpointsNode.GetNumberOfFiducials()):
        endpoint = np.matmul(np.linalg.inv(aff), np.append(np.array(endpointsNode.GetCurvePoints().GetPoint(idx)), 1.0))[:3]
        # Reference point set at [350, 0, 0] in LAS coordinates
        distanceToRASOrigin.append(np.linalg.norm(endpoint - np.array([350.0, 0.0, 0.0])))
    # Get order from closest to furthest
    sortedDistanceIdx = np.argsort(distanceToRASOrigin)
    for idx in sortedDistanceIdx:
        startpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(idx))
        if np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] > thresholdDistance:
            print(f"Startpoint {idx} found is not in the AA region")
            pass
        else:
            # Make sure it is close to the bottom slice
            if np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] < thresholdDistance and idx == 0:
                print("Original startpoint is at distal AA")
                break
            elif np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] < thresholdDistance and idx != 0:
                print("New startpoint:", idx, startpoint)
                endpointsNode.SetNthFiducialPositionFromArray(idx, endpointsNode.GetCurvePoints().GetPoint(0))
                endpointsNode.SetNthFiducialPositionFromArray(0, startpoint)
                break
            else: 
                pass
    
    return endpointsNode

def extractCenterline(segmentationNode, segmentId, maskedVolumeArray, aff):  
    ''' Extracts the centerline model node from the segmentationNode for the corresponding 
    segmentId using Slicer's VMTK extension.

    Arguments: 
        - segmentationNode <vtkMRMLSegmentationNode>: MRML segmentation node.
        - segmentId <int>: segment identifier in the segmentationNode.
        - maskedVolumeArray <np.array>: binary numpy array corresponding to the segmented volume.
        - aff <np.array>: affine matrix from RAS to ijk tranformation.

    Returns:
        - centerlinePolyData <vtkPolyData>: centerline model in vtkPolyData form for segmentId segment.

    '''
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

    # Get volume node array from segmentation node
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode)
    segmentationArray = slicer.util.arrayFromVolume(labelmapVolumeNode)

    # Get affine matrix from segmentation labelMapVolumeNode
    vtkAff = vtk.vtkMatrix4x4()
    affEye = np.eye(4)
    labelmapVolumeNode.GetIJKToRASMatrix(vtkAff)
    vtkAff.DeepCopy(affEye.ravel(), vtkAff)

    # Get endpoints node
    endpointsNode = slicer.util.getNode(extractCenterlineWidget._parameterNode.GetNodeReferenceID("EndPoints"))

    # Check if both ends of the aortic arch have at least one endpoint
    if segmentId == 0:
        endpointsNode = AAEndpointCheck(endpointsNode, maskedVolumeArray, aff)

    print("Relocating endpoints to center of mass of local closest object...")
    # Relocate endpoints for robust centerline extraction 
    for idx in range(endpointsNode.GetNumberOfFiducials()):
        endpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(idx))
        newEndpoint = robustEndPointDetection(endpoint, segmentationArray, affEye) # Center of mass of closest component method
        endpointsNode.SetNthFiducialPosition(idx, newEndpoint[0],
                                                  newEndpoint[1],
                                                  newEndpoint[2])

    print("Extracting centerline...")
    # Create new Surface model node for the centerline model
    centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    # Set centerline node reference to new empty node
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("CenterlineModel", centerlineModelNode.GetID())
    extractCenterlineWidget.onApplyButton()

    print("Checking for floating centerlines (errors)...")
    # Check if all centerlines depart from the same origin. Dismiss the ones that don't, they are most likely floating
    centerlinePolyData = centerlineModelNode.GetPolyData()

    # Declare empty arrays
    cellsIdArray = np.ndarray([centerlinePolyData.GetNumberOfCells()], dtype=int)
    cellsFirstCoordinateArray = np.ndarray([centerlinePolyData.GetNumberOfCells(), 3])

    # Iterate over cells to extract cell IDs, positions and radii. Store lengths of cells
    for cellID in range(centerlinePolyData.GetNumberOfCells()):
        cellsIdArray[cellID] = cellID
        cell = vtk.vtkGenericCell()
        centerlinePolyData.GetCell(cellID, cell)
        cellsFirstCoordinateArray[cellID] = np.matmul(aff, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

    uniqueCellsFirstCoordinateArray, counts = np.unique(cellsFirstCoordinateArray, return_counts=True, axis=0)

    # Get all those that do not start at the startpoint
    removeFloating = []
    for idx in range(len(uniqueCellsFirstCoordinateArray)):
        if idx != np.argmax(counts):
            for idx2 in range(centerlinePolyData.GetNumberOfCells()):
                if (uniqueCellsFirstCoordinateArray[idx] == cellsFirstCoordinateArray[idx2]).all(): removeFloating.append(idx2)

    if len(removeFloating) > 0:
        print("   Found floating centerlines:", removeFloating)
    else:
        print("   No errors found")

    for idx in removeFloating:
        centerlinePolyData.DeleteCell(idx)

    centerlinePolyData.RemoveDeletedCells()
    
    return centerlinePolyData

def inspectCircularCenterlines(centerlinePolyData, surfaceModel, segmentationNode, segmentId, aff):
    ''' Analyzes surface model and centerline model to recognize large surface areas without associated centerline.
    This can happen due to either the presence of circular segments (VMTK does not contemplate this possibility) or due
    to suboptimal segmentationm at some point of the volume, causing a vessel to be too thin for the centerline to 
    be correctly extracted.

    If a centerline-less segment is found, it performs a special centerline extraction based on endpoint detection
    over the segmnent of interest and the previously extracted centerline model.

    Arguments:
        - centerlinePolyData <vtkPolyData>: centerline model in vtkPolyData form for segmentId segment.
        - surfaceModel <vtkMRMLModelNode>: surface model.
        - segmentationNode <vtkMRMLSegmentationNode>: MRML segmentation node.
        - segmentId <int>: segment identifier in the segmentationNode.
        - aff <np.array>: affine matrix from RAS to ijk tranformation.

    Returns:
        - centerlinePolyData <vtkPolyData>: new centerline model in vtkPolyData form for segmentId segment.
    
    '''
    start10 = time()
    print("Inspecting centerline")
    # Check for potential circular centerlines
    decimationFactor = 10
    # Minimum fraction of surface model points at a distance larger than 2 * radius to consider presence of centerlineless segment
    thresholdRadiusRatio = 0.05
    # Distance in ijk units to propagate a single centerlineless segment
    thresholdDistanceProp = 5
    # Minimum number of surface model points to consider unique propagated segment as centerlineless
    thresholdCounts = 250
    
    # First we pool all centerline points and their associated radius
    centerlinePositionsArray = np.ndarray([centerlinePolyData.GetNumberOfPoints(), 3])
    radiusArray = vtk_to_numpy(centerlinePolyData.GetPointData().GetArray("Radius"))

    for idx in range(centerlinePolyData.GetNumberOfPoints()):
        centerlinePositionsArray[idx] = np.array(centerlinePolyData.GetPoint(idx))

    # Then, we pool all surface model points (we take the average position of each cell)
    surfaceModelPositionsArray = np.ndarray([surfaceModel.GetNumberOfCells(), 3])

    for cellIdx in range(surfaceModel.GetNumberOfCells()):
        averagePositionAux = np.ndarray([3, 3])
        for idx in range(3):
            averagePositionAux[idx] = np.array([surfaceModel.GetCell(cellIdx).GetPoints().GetPoint(idx)])
        surfaceModelPositionsArray[cellIdx] = np.mean(averagePositionAux, axis = 0)

    # Now we compute the distance of each surface model point to the closest centerline point. We also store the associated radius
    surfaceModelFeatureArray = np.ndarray([surfaceModel.GetNumberOfCells(), 2])
    # We decimate the surface model points for computation speed. We can do it aggressively and results are not significantly affected
    decimatedSurfaceModelFeatureArray = surfaceModelFeatureArray[::decimationFactor, :]
    decimatedSurfaceModelPositionsArray = surfaceModelPositionsArray[::decimationFactor, :]

    for idx, point in enumerate(decimatedSurfaceModelPositionsArray):
        idxMin = np.argmin(np.linalg.norm(centerlinePositionsArray - point, axis = 1))
        decimatedSurfaceModelFeatureArray[idx][0] = np.linalg.norm(centerlinePositionsArray[idxMin] - point, axis = 0)
        decimatedSurfaceModelFeatureArray[idx][1] = radiusArray[idxMin]

    # In order to identify if there is a potential significant segment without associated centerline, we can check the percentage of surface model cells at a large distance to any centerline point
    distanceRadiusRatio = decimatedSurfaceModelFeatureArray[:, 0] / decimatedSurfaceModelFeatureArray[:, 1]
    
    start11 = time()
    print(f"Uncenterlined segments check took {start11 - start10} s")
    print("% of surface model points at > 2 * radius to any centerline:", 100 * len(distanceRadiusRatio[distanceRadiusRatio > 2]) / len(distanceRadiusRatio))
    
    if len(distanceRadiusRatio[distanceRadiusRatio > 2]) / len(distanceRadiusRatio) < thresholdRadiusRatio:
        print("No significant segments without centerline were found")
        print()
        
        return centerlinePolyData
        
    else:
        print("A significant segment without centerline was identified")
        print()

        # Label all surface model cells with no associated centerline (at a distance > 2 * radius)
        labels = np.zeros([len(decimatedSurfaceModelFeatureArray)])
        for idx in range(len(decimatedSurfaceModelFeatureArray)):
            if decimatedSurfaceModelFeatureArray[idx, 0] / decimatedSurfaceModelFeatureArray[idx, 1] > 2:
                labels[idx] = 1

        # We have to filter out potential islands that may have been incorrectly formed by clustering
        # Search for number of connected components with labels != 0
        # Establish a lower threshold for the number of connected triangles with label != 0
        decimatedSurfaceModelPositionsArrayOnes = decimatedSurfaceModelPositionsArray[labels == 1]

        # We have designed a region growing algorithm to cluster the multiple islands that may have been formed, taking the Euclidean distance as the base metric
        # The algorithm does as follows:
        #     1) Selects an initial point (we take that with the lowest z coordinate)
        #     2) Assigns a label to that point
        #     3) From there, it computes the distance from this to all other points and attributes the same label to those at a distance smaller than 10 mm
        #     4) Checks if the next closest point is at a distance smaller than 10 mm to any point of the current cluster
        #     5) If so, that point passes as the initial seed and the process is repeated until we find that the closest point to the cluster is at a distance larger than 10 mm
        #     6) When this condition fails, the closes point to the cluster is taken as the initial seed for the next cluster, and the clustering label gets updated
        #     7) The algorithm stops when all points have been assigned to a clustering label
        label2 = 1
        label2Array = np.zeros([len(decimatedSurfaceModelPositionsArrayOnes)])
        initialPoint = decimatedSurfaceModelPositionsArrayOnes[np.argmin(decimatedSurfaceModelPositionsArrayOnes[:, 2])]
        label2Array[np.argmin(decimatedSurfaceModelPositionsArrayOnes[:, 2])] = label2

        # The algorithm will continue until all points belong to any cluster
        while 0 in np.unique(label2Array):
            distancesArray = []
            # Checks distance to all unassigned points
            for idx, point in enumerate(decimatedSurfaceModelPositionsArrayOnes):
                if label2Array[idx] == 0:
                    distance = np.linalg.norm(initialPoint - point)
                    # If distance is smaller than 10 mm, assign to same cluster
                    if distance < thresholdDistanceProp:
                        label2Array[idx] = label2
                    # Otherwise, keep distance to current cluster seed
                    else:
                        distancesArray.append(distance)

            if len(distancesArray) > 0:
                # Convert closest point to new seed
                minPointDistanceArg = np.argmin(distancesArray)
                initialPoint = decimatedSurfaceModelPositionsArrayOnes[label2Array == 0][minPointDistanceArg]
                newMinDistance = np.amin(np.linalg.norm(decimatedSurfaceModelPositionsArrayOnes[label2Array == label2] - initialPoint, axis=1))
                # If distance from new seed to prior cluster is less than 10 mm, keep same clustering label
                if newMinDistance < thresholdDistanceProp:
                    pass
                # Otherwise, start a new cluster
                else:
                    label2 += 1
                labelMinDistanceArg = np.argmin(np.linalg.norm(decimatedSurfaceModelPositionsArrayOnes - initialPoint, axis=1))
                label2Array[labelMinDistanceArg] = label2
            
        start12 = time()
        print(f"Uncenterlined segments labelling took {start12 - start11} s")

        # To decide wether to keep an initially identified cluster or not, we can look at the counts of the islands
        # If counts are less than a given thresholds, it means that the island is very small and it is probably an error (remember that this value is decimated by a factor, so this means that the undecimated cutoff is much larger)
        values, counts = np.unique(label2Array, return_counts=True)
       
        circularSegmentModelNode = None
        
        for valIdx, value in enumerate(values):
            labelsCopy = labels.copy()
            idxAux = 0
            # If counts are larger than thresholdCounts, it means it is probably a well-identified segment, then we keep it
            # Otherwise, we discard it as it is probably a sparse island, incorrecly clustered
            if counts[valIdx] > thresholdCounts:
                start13 = time()
                print(f"Segment with label {int(value)}, found with {counts[valIdx]} counts. Creating closed surface model and importing into Slicer...")
                print()
                for idx, labelCluster in enumerate(labels):
                    # For originally clustered points
                    if labelCluster == 1: 
                        if label2Array[idxAux] != value:
                            labelsCopy[idx] = 0
                        idxAux += 1

                # Finally, we assign a label to all surface model points that were initially ignored
                # We attibute the label of the closest labelled point
                labelsArray = np.ndarray([surfaceModel.GetNumberOfCells()])
                for idx, point in enumerate(surfaceModelPositionsArray):
                    idxMin = np.argmin(np.linalg.norm(decimatedSurfaceModelPositionsArray - point, axis = 1))
                    labelsArray[idx] = labelsCopy[idxMin]

                labeledDecimatedSurfaceModel = vtk.vtkPolyData()
                labeledDecimatedSurfaceModel.DeepCopy(surfaceModel)
                labeledDecimatedSurfaceModel.GetCellData().AddArray(numpy_to_vtk(labelsArray))
                labeledDecimatedSurfaceModel.GetCellData().GetArray(0).SetName("Label")

                start14 = time()
                print(f"Labelled surface model generation took {start14 - start13} s")

                # writer = vtk.vtkPolyDataWriter()
                # writer.SetInputData(labeledDecimatedSurfaceModel)
                # writer.SetFileName(os.path.join(caseDir, f"labelSurfaceModel{valIdx}.vtk"))
                # writer.Write()

                # In order to compute centerlines for the centerline-less vessel, 
                # we split it from the rest of the arterial tree in the form of a closed surface model
                # To do that:
                    # 1) Create an open surface model
                    # 2) Load into Slicer
                    # 3) Perform centerline extraction

                # Pass cell labels for points
                labelsPointArray = np.zeros([surfaceModel.GetNumberOfPoints()])

                for cellIdx in range(surfaceModel.GetNumberOfCells()):
                    for idx in range(surfaceModel.GetCell(cellIdx).GetNumberOfPoints()):
                        labelsPointArray[surfaceModel.GetCell(cellIdx).GetPointId(idx)] = labelsArray[cellIdx]

                # Initialize the vtkPoints and the vtkCellArray objects for the circularSegmentModel
                pointsCircularSegmentModel = vtk.vtkPoints()
                cellArrayCircularSegmentModel = vtk.vtkCellArray()

                pointIdArray = np.arange(surfaceModel.GetNumberOfPoints())[labelsPointArray == 1]
                cellIdArray = np.arange(surfaceModel.GetNumberOfCells())[labelsArray == 1]

                for idx in pointIdArray:
                    pointsCircularSegmentModel.InsertNextPoint(surfaceModel.GetPoint(idx))

                # Insert the cells with the corresponding label=1 to the new vtkCellArray
                for idx in cellIdArray:
                    cell = vtk.vtkTriangle()
                    cell.GetPointIds().SetNumberOfIds(surfaceModel.GetCell(idx).GetNumberOfPoints())
                    for idx2 in range(surfaceModel.GetCell(idx).GetNumberOfPoints()):
                        if surfaceModel.GetCell(idx).GetPointId(idx2) in pointIdArray:
                            cell.GetPointIds().SetId(idx2, int(np.where(pointIdArray == surfaceModel.GetCell(idx).GetPointId(idx2))[0]))
                        else:
                            pointIdArray = np.append(pointIdArray, surfaceModel.GetCell(idx).GetPointId(idx2))
                            pointsCircularSegmentModel.InsertNextPoint(surfaceModel.GetPoint(surfaceModel.GetCell(idx).GetPointId(idx2)))
                            cell.GetPointIds().SetId(idx2, int(np.where(pointIdArray == surfaceModel.GetCell(idx).GetPointId(idx2))[0]))
                    cellArrayCircularSegmentModel.InsertNextCell(cell)

                print("Creating circularPolyData")

                # First we create a new vtkPolyData for the labelled surface model, only including labelled cells
                circularSegment = vtk.vtkPolyData()
                circularSegment.SetPoints(pointsCircularSegmentModel)
                circularSegment.SetPolys(cellArrayCircularSegmentModel)

                # We can to compute the normals for all mesh triangles
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(circularSegment)
                normals.SetFeatureAngle(80)
                normals.AutoOrientNormalsOn()
                normals.UpdateInformation()
                normals.Update()
                circularSegment = normals.GetOutput()
                # Clean the vtkPolyData. This allows edges to be correctly identified
                cleanPolyData = vtk.vtkCleanPolyData()
                cleanPolyData.SetInputData(circularSegment)
                cleanPolyData.Update()
#                 circularSegment = cleanPolyData.GetOutput()
                cleanCircularSegment = cleanPolyData.GetOutput()

                # Associate it with a MRML model node to load it into the Slicer context
                circularSegmentModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
                circularSegmentModelNode.SetAndObservePolyData(cleanCircularSegment)

                start15 = time()
                print(f"Surface model creation took {start15 - start14} s")

                # Apply the centerline extraction algorithm for circular segments
                centerlinePolyData = extractCenterlineCircularSegment(circularSegmentModelNode, centerlinePolyData, segmentationNode, segmentId, aff)

        return centerlinePolyData
    
def extractCenterlineCircularSegment(inputSurfaceModelNode, centerlinePolyData, segmentationNode, segmentId, aff): 
    ''' Special centerline extraction for circular segments
    This methods is only applied when a circular segment is recognised. The closed surface model obtained from
    the circular segment inspection is used to obtain the start- and endpoints for the centerline extraction. 
    Moreover, the startpoint from the original segment is used to generate a smooth centerline with the same format 
    as the rest of the centerline cells from the original centerlinePolyData. 
    
    The startpoint will always correspond to the proximal end of the circular segment. The distal end's endpoint is
    used to recognize the alternative path's centerline, in order to get the actual endpoint of the centerline cell.
    Then, two centerlines should be extracted:
        1) One that joins the proximal end of the circular segment with the endpoint of the alternative segment (except last 4 points).
        2) One that joins the proximal end of the circular segment with the startpoint of the overall segment's centerline model.
        
    The 2nd segment is inverted and then joint to the 1st, and the resulting centerline cell is added to the original 
    centerlinePolyData as a new centerline cell. Then, for a correct extraction of the branch and clipped models, a new centerline cell 
    is created by continuing the alternate segment's cell and the last 4 points of the circular centerline cell.

    Arguments:
        - inputSurfaceModelNode <vtkMRMLModelNode>: surface model node corresponding to the centerline-less segment.
        - centerlinePolyData <vtkPolyData>: centerline model in vtkPolyData form for segmentId segment.
        - segmentationNode <vtkMRMLSegmentationNode>: MRML segmentation node.
        - segmentId <int>: segment identifier in the segmentationNode.
        - aff <np.array>: affine matrix from RAS to ijk tranformation.
    
    Returns:
        - finalCenterlinePolyData <vtkPolyData>: final centerline segment with the new circular segment cells added to centerlinePolyData.
    
    '''
    start20 = time()
    print("Extracting circular segment...")

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
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("InputSurface", inputSurfaceModelNode.GetID())

    print("Automatic endpoint extraction...")
    # Autodetect endpoints
    extractCenterlineWidget.onAutoDetectEndPoints()
    extractCenterlineWidget.updateGUIFromParameterNode()
    
    # Set network node reference to original segment
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("InputSurface", segmentationNode.GetID())
    extractCenterlineWidget.ui.inputSegmentSelectorWidget.setCurrentSegmentID(segmentationNode.GetSegmentation().GetNthSegmentID(segmentId))

    start21 = time()
    print(f"Automatic endpoint extraction took {start21 - start20} s")

    print("Relocating endpoints to center of mass of local closest object...")
    # Get volume node array from segmentation node
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode)
    segmentationArray = slicer.util.arrayFromVolume(labelmapVolumeNode)

    # Get affine matrix from segmentation labelMapVolumeNode
    vtkAff = vtk.vtkMatrix4x4()
    affEye = np.eye(4)
    labelmapVolumeNode.GetIJKToRASMatrix(vtkAff)
    vtkAff.DeepCopy(affEye.ravel(), vtkAff)

    # Get endpoints node
    endpointsNode = slicer.util.getNode(extractCenterlineWidget._parameterNode.GetNodeReferenceID("EndPoints"))
    startpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(0))
    while endpointsNode.GetNumberOfFiducials() > 2:
        distancesEndpoints = []
        for idx in range(1, endpointsNode.GetNumberOfFiducials()):
            distancesEndpoints.append(np.linalg.norm(startpoint - np.array(endpointsNode.GetCurvePoints().GetPoint(idx))))
        endpointsNode.RemoveMarkup(np.argmin(distancesEndpoints) + 1)
        
    # Relocate endpoints for robust centerline extraction 
    for idx in range(endpointsNode.GetNumberOfFiducials()):
        endpoint = np.array(endpointsNode.GetCurvePoints().GetPoint(idx))
        newEndpoint = robustEndPointDetection(endpoint, segmentationArray, affEye) # Center of mass of closest component method
        # Search for alternative centerline
        if idx > 0:
            closestCellPoints = np.ndarray([centerlinePolyData.GetNumberOfCells()])
            for cellIdx in range(centerlinePolyData.GetNumberOfCells()):
                centerlineCellPoints = np.ndarray([centerlinePolyData.GetCell(cellIdx).GetNumberOfPoints(), 3])
                for pointIdx in range(centerlinePolyData.GetCell(cellIdx).GetNumberOfPoints()):
                    centerlineCellPoints[pointIdx] = centerlinePolyData.GetCell(cellIdx).GetPoints().GetPoint(pointIdx)
                closestCellPoints[cellIdx]= np.amin(np.linalg.norm(newEndpoint - centerlineCellPoints, axis = 1))
            closestCell = np.argmin(closestCellPoints)
            newEndpoint = centerlinePolyData.GetCell(closestCell).GetPoints().GetPoint(centerlinePolyData.GetCell(closestCell).GetNumberOfPoints() - 1)
        endpointsNode.SetNthFiducialPosition(idx, newEndpoint[0],
                                                  newEndpoint[1],
                                                  newEndpoint[2])
    
    # We add startpoint of original segment as endpoint
    startpoint = centerlinePolyData.GetCell(0).GetPoints().GetPoint(0)
    endpointsNode.AddFiducialFromArray(np.array(startpoint))

    start22 = time()
    print(f"Endpoint relocation took {start22 - start21} s")

    print("Extracting centerline...")
    # Create new Surface model node for the centerline model
    centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    # Set centerline node reference to new empty node
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("CenterlineModel", centerlineModelNode.GetID())
    extractCenterlineWidget.onApplyButton()

    start23 = time()
    print(f"Centerline extraction took {start23 - start22} s")

    print("Checking for floating centerlines (errors)...")
    # Check if all centerlines depart from the same origin. Dismiss the ones that don't, they are most likely floating
    circularCenterlinePolyData = centerlineModelNode.GetPolyData()

    # Declare empty arrays
    cellsIdArray = np.ndarray([circularCenterlinePolyData.GetNumberOfCells()], dtype=int)
    cellsFirstCoordinateArray = np.ndarray([circularCenterlinePolyData.GetNumberOfCells(), 3])

    # Iterate over cells to extract cell IDs, positions and radii. Store lengths of cells
    for cellID in range(circularCenterlinePolyData.GetNumberOfCells()):
        cellsIdArray[cellID] = cellID
        cell = vtk.vtkGenericCell()
        circularCenterlinePolyData.GetCell(cellID, cell)
        cellsFirstCoordinateArray[cellID] = np.matmul(aff, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

    uniqueCellsFirstCoordinateArray, counts = np.unique(cellsFirstCoordinateArray, return_counts=True, axis=0)

    # Get all those that do not start at the startpoint
    removeFloating = []
    for idx in range(len(uniqueCellsFirstCoordinateArray)):
        if idx != np.argmax(counts):
            for idx2 in range(circularCenterlinePolyData.GetNumberOfCells()):
                if (uniqueCellsFirstCoordinateArray[idx] == cellsFirstCoordinateArray[idx2]).all(): removeFloating.append(idx2)

    if len(removeFloating) > 0:
        print("   Found floating centerlines:", removeFloating)
    else:
        print("   No errors found")

    for idx in removeFloating:
        circularCenterlinePolyData.DeleteCell(idx)

    circularCenterlinePolyData.RemoveDeletedCells()

    start24 = time()
    print(f"Error checking took {start24 - start23} s")

    print("Building final vtkPolyData")
    
    # We join both cells to create the final cell of the original centerlinePolyData 
    # Then we add it to the centerlinePolyData as a new vtkPolyLine cell
    
    # Initialize the new vtkPoints with those from the original centerlinePolyData
    finalPoints = vtk.vtkPoints()
    finalPoints.DeepCopy(centerlinePolyData.GetPoints())
    # Get the numpy arrays for the pointData arrays
    radiusNumpy = vtk_to_numpy(centerlinePolyData.GetPointData().GetArray("Radius"))
    edgeArrayNumpy = vtk_to_numpy(centerlinePolyData.GetPointData().GetArray("EdgeArray"))
    edgePCoordArrayNumpy = vtk_to_numpy(centerlinePolyData.GetPointData().GetArray("EdgePCoordArray"))
    # We start a new vtkCellArrray with the cells from the original centerlinePolyData
    finalCellArray = vtk.vtkCellArray()
    for cellIdx in range(centerlinePolyData.GetNumberOfCells()):
        finalCellArray.InsertNextCell(centerlinePolyData.GetCell(cellIdx))
    
    # Now, we join both cells from the circularCenterlinePolyData as one with the correct order
    assert circularCenterlinePolyData.GetNumberOfCells() == 2
    
    # We have to search for the exact preceeding centerline segment from the original centerlinePolyData
    # First store all centerlines from the original centerlinePolyData in numpy arrays for speed
    centerlineACellArrays = np.ndarray([centerlinePolyData.GetNumberOfCells()], dtype = object)
    for cellIdxA in range(centerlinePolyData.GetNumberOfCells()):
        centerlineACellArrays[cellIdxA] = np.ndarray([centerlinePolyData.GetCell(cellIdxA).GetNumberOfPoints(), 3])
        for pointIdxA in range(centerlinePolyData.GetCell(cellIdxA).GetNumberOfPoints()):
            centerlineACellArrays[cellIdxA][pointIdxA] = centerlinePolyData.GetCell(cellIdxA).GetPoints().GetPoint(pointIdxA)
    # Then, for each centerline point, search for closest point to each centerlinePolyData cell and get the first under 0.1 mm    
    for pointIdxB in range(circularCenterlinePolyData.GetCell(1).GetNumberOfPoints()):
        point = circularCenterlinePolyData.GetCell(1).GetPoints().GetPoint(pointIdxB)
        for cellIdxA in range(centerlinePolyData.GetNumberOfCells()):
            distances = np.linalg.norm(point - centerlineACellArrays[cellIdxA], axis = 1)
            if np.amin(distances) < 0.1:
                pointIdxA = np.argmin(distances)
                break # cellIdxA, pointIdxA correspond to the preceeding centerline
                      # pointIdxB - 1 will be the first point used from the circularCenterline
                      # We will concatenate both
        if np.amin(distances) < 0.1:
            break
            
    # We can also search for the distal branch of the centerline model. Branching and clipping present problems if nothing else is done
    # The idea is to split the distal end into a new centerline cell
    # Then, for each centerline point, search for closest point to each centerlinePolyData cell (now cell 2) and get the first under 0.1 mm    
    for pointIdx2B in range(circularCenterlinePolyData.GetCell(0).GetNumberOfPoints()):
        point = circularCenterlinePolyData.GetCell(0).GetPoints().GetPoint(pointIdx2B)
        for cellIdx2A in range(centerlinePolyData.GetNumberOfCells()):
            distances = np.linalg.norm(point - centerlineACellArrays[cellIdx2A], axis = 1)
            if np.amin(distances) < 0.1:
                pointIdx2A = np.argmin(distances)
                break # cellIdxA, pointIdxA correspond to the preceeding centerline
                      # pointIdxB - 1 will be the first point used from the circularCenterline
                      # We will concatenate both
        if np.amin(distances) < 0.1:
            break
    
    # We define the necessary objects for the circular centerline cell
    circularCenterlinePolyLine = vtk.vtkPolyLine()
    circularCenterlinePolyLinePoints = vtk.vtkPoints()
    circularCenterlinePolyLinePointIds = []
    # We have to be coherent with the point Ids of the new cell
    previousNumberOfPoints = centerlinePolyData.GetNumberOfPoints()
    idxAux = 0
    
    # We build the final centerlinePolyData
    for cellIdx in range(circularCenterlinePolyData.GetNumberOfCells() - 1, -1, -1):
        circularCenterlineCell = circularCenterlinePolyData.GetCell(cellIdx)
        # Start from the second cell, in inverse order
        if cellIdx == 1:
            # We first add the first segment fromt he original centerlinePolyData
            for pointIdx in range(pointIdxA):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circularCenterlinePolyLinePoints.InsertNextPoint(centerlinePolyData.GetCell(cellIdxA).GetPoints().GetPoint(pointIdx))
                finalPoints.InsertNextPoint(centerlinePolyData.GetCell(cellIdxA).GetPoints().GetPoint(pointIdx))
                circularCenterlinePolyLinePointIds.append(previousNumberOfPoints + idxAux)
                idxAux += 1
                # Append pointData values to the numpy arrays
                radiusNumpy = np.append(radiusNumpy, centerlinePolyData.GetPointData().GetArray("Radius").GetValue(centerlinePolyData.GetCell(cellIdxA).GetPointId(pointIdx)))
                edgeArrayNumpy = np.append(edgeArrayNumpy, np.array([centerlinePolyData.GetPointData().GetArray("EdgeArray").GetTuple(centerlinePolyData.GetCell(cellIdxA).GetPointId(pointIdx))]), axis=0)
                edgePCoordArrayNumpy = np.append(edgePCoordArrayNumpy, centerlinePolyData.GetPointData().GetArray("EdgePCoordArray").GetValue(centerlinePolyData.GetCell(cellIdxA).GetPointId(pointIdx)))                
            # We then add the rest of the second cell (1) of the circularCenterlinePolyData
            for pointIdx in range(pointIdxB - 1, -1, -1):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circularCenterlinePolyLinePoints.InsertNextPoint(circularCenterlineCell.GetPoints().GetPoint(pointIdx))
                finalPoints.InsertNextPoint(circularCenterlineCell.GetPoints().GetPoint(pointIdx))
                circularCenterlinePolyLinePointIds.append(previousNumberOfPoints + idxAux)
                idxAux += 1
                # Append pointData values to the numpy arrays
                radiusNumpy = np.append(radiusNumpy, circularCenterlinePolyData.GetPointData().GetArray("Radius").GetValue(circularCenterlineCell.GetPointId(pointIdx)))
                edgeArrayNumpy = np.append(edgeArrayNumpy, np.array([circularCenterlinePolyData.GetPointData().GetArray("EdgeArray").GetTuple(circularCenterlineCell.GetPointId(pointIdx))]), axis=0)
                edgePCoordArrayNumpy = np.append(edgePCoordArrayNumpy, circularCenterlinePolyData.GetPointData().GetArray("EdgePCoordArray").GetValue(circularCenterlineCell.GetPointId(pointIdx)))

        elif cellIdx == 0:
#             for pointIdx in range(circularCenterlineCell.GetNumberOfPoints()):
            for pointIdx in range(pointIdx2B - 3):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circularCenterlinePolyLinePoints.InsertNextPoint(circularCenterlineCell.GetPoints().GetPoint(pointIdx))
                finalPoints.InsertNextPoint(circularCenterlineCell.GetPoints().GetPoint(pointIdx))
                circularCenterlinePolyLinePointIds.append(previousNumberOfPoints + idxAux)
                idxAux += 1
                # Append pointData values to the numpy arrays
                radiusNumpy = np.append(radiusNumpy, circularCenterlinePolyData.GetPointData().GetArray("Radius").GetValue(circularCenterlineCell.GetPointId(pointIdx)))
                edgeArrayNumpy = np.append(edgeArrayNumpy, np.array([circularCenterlinePolyData.GetPointData().GetArray("EdgeArray").GetTuple(circularCenterlineCell.GetPointId(pointIdx))]), axis=0)
                edgePCoordArrayNumpy = np.append(edgePCoordArrayNumpy, circularCenterlinePolyData.GetPointData().GetArray("EdgePCoordArray").GetValue(circularCenterlineCell.GetPointId(pointIdx)))
            # Create a new cell with the cellIdx2A and a few points (2-10) from the circular segment. This can solve several issues in branch model, clipped model and segmentsArray computation
            # We define the necessary objects for the circular centerline cell
            circularCenterlinePolyLine2 = vtk.vtkPolyLine()
            circularCenterlinePolyLinePoints2 = vtk.vtkPoints()
            circularCenterlinePolyLinePointIds2 = []
            for pointIdx in range(pointIdx2A):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circularCenterlinePolyLinePoints2.InsertNextPoint(centerlinePolyData.GetCell(cellIdx2A).GetPoints().GetPoint(pointIdx))
                finalPoints.InsertNextPoint(centerlinePolyData.GetCell(cellIdx2A).GetPoints().GetPoint(pointIdx))
                circularCenterlinePolyLinePointIds2.append(previousNumberOfPoints + idxAux)
                idxAux += 1
                # Append pointData values to the numpy arrays
                radiusNumpy = np.append(radiusNumpy, centerlinePolyData.GetPointData().GetArray("Radius").GetValue(centerlinePolyData.GetCell(cellIdx2A).GetPointId(pointIdx)))
                edgeArrayNumpy = np.append(edgeArrayNumpy, np.array([centerlinePolyData.GetPointData().GetArray("EdgeArray").GetTuple(centerlinePolyData.GetCell(cellIdx2A).GetPointId(pointIdx))]), axis=0)
                edgePCoordArrayNumpy = np.append(edgePCoordArrayNumpy, centerlinePolyData.GetPointData().GetArray("EdgePCoordArray").GetValue(centerlinePolyData.GetCell(cellIdx2A).GetPointId(pointIdx)))
            for pointIdxAux in range(4, -1, -1):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circularCenterlinePolyLinePoints2.InsertNextPoint(circularCenterlineCell.GetPoints().GetPoint(pointIdx2B - 4 + pointIdxAux))
                finalPoints.InsertNextPoint(circularCenterlineCell.GetPoints().GetPoint(pointIdx2B - 4 + pointIdxAux))
                circularCenterlinePolyLinePointIds2.append(previousNumberOfPoints + idxAux)
                idxAux += 1
                # Append pointData values to the numpy arrays
                radiusNumpy = np.append(radiusNumpy, circularCenterlinePolyData.GetPointData().GetArray("Radius").GetValue(circularCenterlineCell.GetPointId(pointIdx2B - 4 + pointIdxAux)))
                edgeArrayNumpy = np.append(edgeArrayNumpy, np.array([circularCenterlinePolyData.GetPointData().GetArray("EdgeArray").GetTuple(circularCenterlineCell.GetPointId(pointIdx2B - 4 + pointIdxAux))]), axis=0)
                edgePCoordArrayNumpy = np.append(edgePCoordArrayNumpy, circularCenterlinePolyData.GetPointData().GetArray("EdgePCoordArray").GetValue(circularCenterlineCell.GetPointId(pointIdx2B - 4 + pointIdxAux)))                           
    
    # Build final cell for the circular centerline and ad it to the rest in the vtkCellArray
    circularCenterlinePolyLine.Initialize(len(circularCenterlinePolyLinePointIds), circularCenterlinePolyLinePointIds, circularCenterlinePolyLinePoints)
    finalCellArray.InsertNextCell(circularCenterlinePolyLine)
    circularCenterlinePolyLine2.Initialize(len(circularCenterlinePolyLinePointIds2), circularCenterlinePolyLinePointIds2, circularCenterlinePolyLinePoints2)
    finalCellArray.InsertNextCell(circularCenterlinePolyLine2)
    # Build a new vtkPolyData object and add the vtkPoints and the vtkCellArray objects
    finalCenterlinePolyData = vtk.vtkPolyData()
    finalCenterlinePolyData.SetPoints(finalPoints)
    finalCenterlinePolyData.SetLines(finalCellArray)
    # Add the PointData
    finalCenterlinePolyData.GetPointData().AddArray(numpy_to_vtk(radiusNumpy))
    finalCenterlinePolyData.GetPointData().GetArray(0).SetName("Radius")
    finalCenterlinePolyData.GetPointData().AddArray(numpy_to_vtk(edgeArrayNumpy))
    finalCenterlinePolyData.GetPointData().GetArray(1).SetName("EdgeArray")
    finalCenterlinePolyData.GetPointData().AddArray(numpy_to_vtk(edgePCoordArrayNumpy))
    finalCenterlinePolyData.GetPointData().GetArray(2).SetName("EdgePCoordArray")

    start25 = time()
    print(f"Building final polyData took {start25 - start24} s")
    
    return finalCenterlinePolyData