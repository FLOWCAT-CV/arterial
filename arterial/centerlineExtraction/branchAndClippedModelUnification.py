import os
import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def branchAndClippedModelUnification(caseDir):
    ''' Reads all branchModels (branchModels/branchModels{idx}.vtk) derived from the 
    centerlines/centerline{idx}.vtk files and creates a unified branchModel.vtk.

    Arguments:
        - caseDir <str>: path to the directory containing the binary mask. All 
        segmentations will be saved in this dir.

    Returns:

    '''

    centerlineList = [centerlineFile for centerlineFile in os.listdir(os.path.join(caseDir, "centerlines")) if centerlineFile.endswith(".vtk")]
        
    print("Unifying all branch models...")
    # Initialize the vtkPoints and the vtkCellArray objects for the branchModel
    cellArrayBranchModel = vtk.vtkCellArray()
    pointsBranchModel = vtk.vtkPoints()

    finalCellDataArrayBranchModel = np.ndarray([4, 0])
    finalRadiusArray = np.ndarray([1, 0])

    accCenterlineId = 0
    accGroupIdBranchModel = 0
    pointsFromPreviousBranchModels = 0

    # Initialize the vtkPoints and the vtkCellArray objects for the clippedModel
    cellArrayClippedModel = vtk.vtkCellArray()
    pointsClippedModel = vtk.vtkPoints()
    finalGroupIdPointArrayClippedModel = vtk.vtkIntArray()
    finalGroupIdPointArrayClippedModel.SetName("GroupIds")

    accGroupIdClippedModel = 0
    pointsFromPreviousClippedModels = 0

    for centerlineModelId, _ in enumerate(centerlineList):
        branchModelPath = os.path.join(caseDir, "branchModels", f"branchModel{centerlineModelId}.vtk")
        
        # Load branch model
        vtkPolyDataReader = vtk.vtkPolyDataReader()
        vtkPolyDataReader.SetFileName(branchModelPath)
        vtkPolyDataReader.Update()
        branchModel = vtkPolyDataReader.GetOutput()

        clippedModelPath = os.path.join(caseDir, "clippedModels", f"clippedModel{centerlineModelId}.vtk")
        
        # Load clipped model
        vtkPolyDataReader = vtk.vtkPolyDataReader()
        vtkPolyDataReader.SetFileName(clippedModelPath)
        vtkPolyDataReader.Update()
        clippedModel = vtkPolyDataReader.GetOutput()

        if branchModel.GetNumberOfCells() == 0:
            print(f"Error in branch model {centerlineModelId}. Skipping")
        elif clippedModel.GetNumberOfCells() == 0:
            print(f"Error in clipped model {centerlineModelId}. Skipping")
        else:
            # Get cell data
            cellDataArray = np.ndarray([4, branchModel.GetNumberOfCells()], dtype=np.int64)
            cellDataArray[0] = vtk_to_numpy(branchModel.GetCellData().GetArray("CenterlineIds")) # centerlinesId -> connections between origin and endpoints
            cellDataArray[1] = vtk_to_numpy(branchModel.GetCellData().GetArray("TractIds")) # tractId -> following a centerline Id, tract number (closest to origin is 0, next is 1 and so on)
            cellDataArray[2] = vtk_to_numpy(branchModel.GetCellData().GetArray("Blanking")) # blanking -> transition to a new branch
            cellDataArray[3] = vtk_to_numpy(branchModel.GetCellData().GetArray("GroupIds")) # groupId -> indicates is the centerline is inside of the tract 
            # Collect max centerlineId and groupId
            maxCenterlineId = np.amax(cellDataArray[0])
            maxGroupId = np.amax(cellDataArray[3])
            # Add centerlineId and groupId (add previous summed maximums)
            cellDataArray[0] = cellDataArray[0] + accCenterlineId
            cellDataArray[3] = cellDataArray[3] + accGroupIdBranchModel
            # Update accumulated centerlineId and groupId
            accCenterlineId += maxCenterlineId + 1
            accGroupIdBranchModel += maxGroupId + 1
            # Append cellDataArray from present branchModel
            finalCellDataArrayBranchModel = np.append(finalCellDataArrayBranchModel, cellDataArray, axis=1)
            
            # Get point data (we only get radius)
            radiusArray = vtk_to_numpy(branchModel.GetPointData().GetArray("Radius"))

            # On rare occasions, there is a mismatch (a gap) between the number of points of the vtkPolyData and the sum of the number of points from each cell
            # These should be restarted for each branchModelIdx
            gap = 0
            idxPointsMinusGap = 0

            for idx in range(branchModel.GetNumberOfCells()):
                polyLine = branchModel.GetCell(idx)
                newPolyLine = vtk.vtkPolyLine()
                newPolyLinePoints = vtk.vtkPoints()
                newPolyLinePointsIds = []
                for idx2 in range(polyLine.GetNumberOfPoints()):
                    # Condition tells us if there is a diference between current cell point and branchModel point with accumulated gap
                    condition = np.abs(np.sum(np.array(polyLine.GetPoints().GetPoint(idx2)) - np.array(branchModel.GetPoints().GetPoint(idxPointsMinusGap + gap)))) < 0.01
                    while not condition:
                        # Insert point in vtkPoints
                        pointsBranchModel.InsertNextPoint(branchModel.GetPoints().GetPoint(idxPointsMinusGap + gap))
                        # Insert point in newPolyLine
                        newPolyLinePoints.InsertNextPoint(branchModel.GetPoints().GetPoint(idxPointsMinusGap + gap))
                        # Get radius data for each point
                        finalRadiusArray = np.append(finalRadiusArray, radiusArray[idxPointsMinusGap + gap])
                        # Change point id for each point in the cell, taking into account accumulated number of points
                        newPolyLinePointsIds.append(idxPointsMinusGap + pointsFromPreviousBranchModels + gap)
                        # Update gap
                        gap += 1
                        # Recompute condition
                        condition = np.abs(np.sum(np.array(polyLine.GetPoints().GetPoint(idx2)) - np.array(branchModel.GetPoints().GetPoint(idxPointsMinusGap + gap)))) < 0.01

                    # Insert point in vtkPoints
                    pointsBranchModel.InsertNextPoint(branchModel.GetPoints().GetPoint(idxPointsMinusGap + gap))
                    # Insert point in newPolyLine
                    newPolyLinePoints.InsertNextPoint(branchModel.GetPoints().GetPoint(idxPointsMinusGap + gap))
                    # Get radius data for each point
                    finalRadiusArray = np.append(finalRadiusArray, radiusArray[idxPointsMinusGap + gap])
                    # Change point id for each point in the cell, taking into account accumulated number of points
                    newPolyLinePointsIds.append(idxPointsMinusGap + pointsFromPreviousBranchModels + gap)
                    # Update idxPointsMinusGap
                    idxPointsMinusGap += 1   
                
                newPolyLine.Initialize(len(newPolyLinePointsIds), newPolyLinePointsIds, newPolyLinePoints)  
                # Insert cell in vtkCellArray
                cellArrayBranchModel.InsertNextCell(newPolyLine)
            
            # Update total number of points from previous models
            pointsFromPreviousBranchModels += branchModel.GetNumberOfPoints()

            print(f"Processing clipped model {centerlineModelId}...")
            
            # Get point data (we only get groupId)
            groupIdPointArrayClippedModel = vtk_to_numpy(clippedModel.GetPointData().GetArray("GroupIds"))
            # Store max groupId
            maxGroupId = np.amax(groupIdPointArrayClippedModel)
            # Update groupIds of current clippedModel
            groupIdPointArrayClippedModel = groupIdPointArrayClippedModel + accGroupIdClippedModel
            # Update accumulated groupId
            accGroupIdClippedModel += maxGroupId + 1
            # Get number of points in each clipped model cell (= 3)
            numberOfPointIds = clippedModel.GetCell(0).GetPointIds().GetNumberOfIds()
            # Generally, points are placed as cell indices go up, but this is not always the case
            # To speed up computations, we only search for cells with higher cellIds than the ones already searched for, 
            # But in the cases where a pointIdx has not been found, we search across all cells of the model, in order
            # to ensure that no pointIdx is missed
            lastCell = 0
            # We iterate through every pointId
            for pointIdx in range(clippedModel.GetNumberOfPoints()):
                # We need a boolean variable to stop the iterative search when a point is found to speed up computations
                foundPoint = False
                # We primarily only search for cells with a cellId larger than the ones analyzed
                # Limiting up the search dramatically speeds up computations
                for cellIdx in range(max(0, lastCell - 1), clippedModel.GetNumberOfCells()):
                    # Iterate over points in cell
                    for idx in range(numberOfPointIds):
                        # If a point is found with pointId equal to the next pointIdx
                        if clippedModel.GetCell(cellIdx).GetPointId(idx) == pointIdx:
                            # Keep cellIdx to limit cell of the next pointIdx
                            lastCell = cellIdx
                            # Insert next point in final clipped model point object and groupId point array
                            pointsClippedModel.InsertNextPoint(clippedModel.GetCell(cellIdx).GetPoints().GetPoint(idx))
                            finalGroupIdPointArrayClippedModel.InsertNextValue(groupIdPointArrayClippedModel[clippedModel.GetCell(cellIdx).GetPointId(idx)])
                            # Update boolean marker to stop the search for the current pointidx
                            foundPoint = True
                            break
                    # Break cell serach if point is found
                    if foundPoint:
                        break
                # If point is not found, search all throughout the cell pool, including cells with a smaller cellIdx than lastCell
                # These searches are significantly longer than the general case, but we only apply them when needed
                # This is very rare but if not done, it will mess up the final model
                if not foundPoint:
                    # If pointIdx has not been found, we also look at the previous cells (rare but it happens)
                    for cellIdx in range(clippedModel.GetNumberOfCells()):
                        # Iterate over points in cell
                        for idx in range(numberOfPointIds):
                            # If a point is found with pointId equal to the next pointIdx
                            if clippedModel.GetCell(cellIdx).GetPointId(idx) == pointIdx:
                                # Keep cellIdx to limit cell of the next pointIdx
                                lastCell = cellIdx
                                # Insert next point in final clipped model point object and groupId point array
                                pointsClippedModel.InsertNextPoint(clippedModel.GetCell(cellIdx).GetPoints().GetPoint(idx))
                                finalGroupIdPointArrayClippedModel.InsertNextValue(groupIdPointArrayClippedModel[clippedModel.GetCell(cellIdx).GetPointId(idx)])
                                # Update boolean marker to stop the search for the current pointidx
                                foundPoint = True
                        # Break cell serach if point is found
                        if foundPoint:
                            break


            # We need this to set the new pointIds for the triangles with the SetId method. This will be 3
            # Insert the cells with the corresponding groupId to the new vtkCellArray
            # for idx in cellIdArray:
            for cellIdx in range(clippedModel.GetNumberOfCells()):
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetNumberOfIds(numberOfPointIds)
                # print(cellIdx)
                for idx in range(numberOfPointIds):
                    cell.GetPointIds().SetId(idx, clippedModel.GetCell(cellIdx).GetPointId(idx) + pointsFromPreviousClippedModels)
                cellArrayClippedModel.InsertNextCell(cell)
            
            # Update total number of points from previous models
            pointsFromPreviousClippedModels += clippedModel.GetNumberOfPoints()
            
    # Store all branch model data in new vtkPolyData
    finalBranchModel = vtk.vtkPolyData()
    finalBranchModel.SetPoints(pointsBranchModel)
    finalBranchModel.SetLines(cellArrayBranchModel)

    for idx in range(branchModel.GetCellData().GetNumberOfArrays()):
        finalBranchModel.GetCellData().AddArray(numpy_to_vtk(finalCellDataArrayBranchModel[idx], array_type=vtk.VTK_INT))
        finalBranchModel.GetCellData().GetArray(idx).SetName(branchModel.GetCellData().GetArrayName(idx))

    finalBranchModel.GetPointData().AddArray(numpy_to_vtk(finalRadiusArray))
    finalBranchModel.GetPointData().GetArray(0).SetName("Radius")

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(finalBranchModel)
    writer.SetFileName(os.path.join(caseDir, "branchModel.vtk"))
    writer.Write()

    # Store all clipped model data in new vtkPolyData
    finalClippedModel = vtk.vtkPolyData()
    finalClippedModel.SetPoints(pointsClippedModel)
    finalClippedModel.SetPolys(cellArrayClippedModel)
    finalClippedModel.GetPointData().AddArray(finalGroupIdPointArrayClippedModel)

    # We can to compute the normals for all mesh triangles
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(finalClippedModel)
    normals.SetFeatureAngle(80)
    normals.AutoOrientNormalsOn()
    normals.UpdateInformation()
    normals.Update()
    finalClippedModel = normals.GetOutput()

    # We also pass a clean vtkPolyData filter for good measure
    cleanPolyData = vtk.vtkCleanPolyData()
    cleanPolyData.SetInputData(finalClippedModel)
    cleanPolyData.Update()
    finalClippedModel = cleanPolyData.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(finalClippedModel)
    writer.SetFileName(os.path.join(caseDir, "clippedModel.vtk"))
    writer.Write()