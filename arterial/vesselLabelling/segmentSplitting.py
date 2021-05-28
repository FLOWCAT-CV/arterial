import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def segmentSplitting(caseDir):
    ''' Inputs the decimated surface model of the segmentation resulting from the 
    vmtkbranchclipper and writes a separate file for each segment that makes up the model.

    The step of writing the segments individually could be omitted for conciseness, but at the
    moment it helps to understand what we are working on.

    Arguments:
        - caseDir <str>: path to the directory containing the binary mask. All 
        segmentations will be saved in this dir.

    Returns:

    '''
    clippedModel = os.path.join(caseDir, "clippedModel.vtk")
    if not os.path.isdir(os.path.join(caseDir, "surfaceSegments")): os.mkdir(os.path.join(caseDir, "surfaceSegments"))

    # Load clipped surface model
    vtkPolyDataReader = vtk.vtkPolyDataReader()
    vtkPolyDataReader.SetFileName(clippedModel)
    vtkPolyDataReader.Update()
    clippedModelPolyData = vtkPolyDataReader.GetOutput()

    # Pass groupIds array to numpy array
    pointGroupIdsArray = vtk_to_numpy(clippedModelPolyData.GetPointData().GetArray("GroupIds"))
    # Make auxiliary array for point positions in original vtkPolyData object
    pointIdsArray = np.arange(len(pointGroupIdsArray))
    # Some ids are literally point-less. Get rid of them
    uniqueIds = np.sort(np.unique(pointGroupIdsArray))

    # Make auxiliary array to store the groupIds of the vtk triangles forming the mesh
    cellGroupIdsArray = np.ndarray([clippedModelPolyData.GetNumberOfCells()])
    # Also, make auxiliary array for the cell positions in the vtkPolyData object
    cellIdsArray = np.arange(len(cellGroupIdsArray))
    # Get the groupIds from the points forming the vtkTriangle
    # All three points forming a polygon are always from the same groupId
    for idx in range(clippedModelPolyData.GetNumberOfCells()):
        cellGroupIdsArray[idx] = pointGroupIdsArray[clippedModelPolyData.GetCell(idx).GetPointId(0)]

    # We form an individual mesh for each groupId
    for groupId in uniqueIds:
        # Get points and cells from the given groupId
        pointIdArray = pointIdsArray[pointGroupIdsArray == groupId]
        cellIdArray = cellIdsArray[cellGroupIdsArray == groupId]
        
        # Initialize the vtkPoints and the vtkCellArray objects
        cellArray = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        
        # Add the data (only groupId atm) of each point
        pointData = vtk.vtkIntArray()
        pointData.SetName("GroupIds")
        
        # Insert points with the groupId to the new vtkPoints object
        for idx in pointIdArray:
            points.InsertNextPoint(clippedModelPolyData.GetPoint(idx))
            pointData.InsertNextValue(groupId)
        
        # We have to substract the initial pointId from the original vtkPolyData for each groupId
        iniId = clippedModelPolyData.GetCell(cellIdArray[0]).GetPointId(0)
        # We need this to set the new pointIds for the triangles with the SetId method
        numberOfIds = clippedModelPolyData.GetCell(idx).GetPointIds().GetNumberOfIds()
        
        # Insert the cells with the corresponding groupId to the new vtkCellArray
        for idx in cellIdArray:
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetNumberOfIds(numberOfIds)
            for idx2 in range(numberOfIds):
                cell.GetPointIds().SetId(idx2, clippedModelPolyData.GetCell(idx).GetPointId(idx2) - iniId)
            cellArray.InsertNextCell(cell)
            
        # Associate points, polygons and data to a new vtkPolyData object
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetPolys(cellArray)
        polyData.GetPointData().AddArray(pointData)
        
        # Write (save) the new vtkPolyData object
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polyData)
        writer.SetFileName(os.path.join(caseDir, "surfaceSegments", f"segment{groupId}.vtk"))
        writer.Write()