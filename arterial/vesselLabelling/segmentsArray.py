import os
import vtk
import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def centerlineSegmentsArray(caseDir, make_plot=False):
    ''' Inputs a vtkPolyData object containing the centerline model and generates
    segmentsArray, spliting the centerline cells into individual segments between bifurcations,
    associating a new identifyier to them.

    The output is piped to the generation of the corresponding graph. Saves segmentsArray as a 
    .npy file.

    Arguments:
        - centerlinePolyData <vtkPolyData>: vtkPolyData object containing centerline model.
        - caseDir <str>: path to the directory containing the binary mask. All 
        segmentations will be saved in this dir.
        - make_plot <bool>: if True, this will display the centerline segments in a 3D plot.
        Set to False if omitted.

    Returns:
        - segmentsArray <numpy array>: numpy array containing the position of the centerline points, 
        as well as the associated radius to each point, and split into individual segments between 
        bifurcations.
    '''
    
    centerlineList = [centerlineFile for centerlineFile in os.listdir(os.path.join(caseDir, "centerlines")) if centerlineFile.endswith(".vtk")]
    finalSegmentsArray = np.ndarray([0, 2])

    for idx, _ in enumerate(centerlineList):
        # Load centerlines.vtk as a vtkPolyData object
        centerlinePolyDataReader = vtk.vtkPolyDataReader()
        centerlinePolyDataReader.SetFileName(os.path.join(caseDir, "centerlines", f"centerlines{idx}.vtk"))
        centerlinePolyDataReader.Update()
        centerlineModel = centerlinePolyDataReader.GetOutput()
    
        # Define number of cells and points in the vtkPolyData
        numberOfCells = centerlineModel.GetNumberOfCells()

        # Declare empty arrays
        cellsIdArray = np.ndarray([numberOfCells], dtype=int)
        cellsCoordinateArray = np.ndarray([numberOfCells], dtype=object)
        cellsRadiusArray = np.ndarray([numberOfCells], dtype=object)
        lengthCoordinateArray = np.ndarray([numberOfCells], dtype=int)

        # Define affine matrix and invert
        aff = np.linalg.inv(nib.load(os.path.join(caseDir, os.path.basename(caseDir) + ".nii.gz")).affine)

        # Centerline point data
        radiusArray = vtk.util.numpy_support.vtk_to_numpy(centerlineModel.GetPointData().GetArray(0))

        # Iterate over cells to extract cell IDs, positions and radii. Store lengths of cells
        for cellID in range(numberOfCells):
            cellsIdArray[cellID] = cellID
            cell = vtk.vtkGenericCell()
            centerlineModel.GetCell(cellID, cell)
            numberOfCellPoints = cell.GetNumberOfPoints()
            cellsCoordinateArray[cellID] = np.ndarray([numberOfCellPoints, 3])
            cellsRadiusArray[cellID] = np.flip(radiusArray[cell.GetPointId(cell.GetPointIds().GetNumberOfIds() - 1):cell.GetPointId(0) + 1])
            for idx in range(numberOfCellPoints):
                cellsCoordinateArray[cellID][idx] = np.matmul(aff, np.append(cell.GetPoints().GetPoint(idx), 1.0))[:3]
            lengthCoordinateArray[cellID] = numberOfCellPoints

        fromShortestToLongest = np.sort(lengthCoordinateArray)
        auxFromShortestToLongest = np.argsort(lengthCoordinateArray) # Auxiliar array that will be iteratively deleted
        bifurcationsArray = np.ndarray([0, 2], dtype=int)

        # We iterate over all segments starting from the shortest segment, except for the longest (unnecesary)
        for shortestLength in fromShortestToLongest[:-1]:
            # New array created for this iteration. Only contains segments with length longer than currently analyzed, 
            # and up to the length of the segment. The goal is to identify bifurcations along each curve.
            auxArray = np.ndarray([len(auxFromShortestToLongest), shortestLength, 3]) 

            idxAux = 0 # We order them from shortest to longest only in the auxArray
            for idx1 in auxFromShortestToLongest:
                auxArray[idxAux] = cellsCoordinateArray[idx1][:shortestLength]
                idxAux += 1

            for idx1 in range(len(auxFromShortestToLongest)):
                cellID = auxFromShortestToLongest[idx1]
                # cellID indicates cellID of an alternative cell
                auxArray2 = auxArray # We generate a second copy of the suxiliar array that we delete iteratively
                # We iterate over coordinates of each cell with a lenght longer or equal than
                # the current shortest cell, searching for bifurcation points
                for idxCoord in range(shortestLength):
                    # idxCoord indicates point position on the current cell
                    # For each, coordinate, we iterate over all other cells to try to locate a bifurcation
                    for idx2 in range(1, auxArray2.shape[0]): # We iterate over the cells that are still on the path
                        # idx2 indicates cellID of an alternative cell
                        if not (auxArray2[0][idxCoord] == auxArray2[idx2][idxCoord]).all():
                            bifurcationsArray = np.append(bifurcationsArray, [[cellID, idxCoord - 1]], 0)
                            # Only continue with branches overlapping with current cellID
                            auxArray2 = np.delete(auxArray2, np.where(auxArray2[:, idxCoord, 0] != auxArray2[0, idxCoord, 0]), axis=0)
                            # Skip to next point
                            break

                auxArray = np.roll(auxArray, -1, axis=0)

            # When all positions have been covered, get rid of the shortest cell and repeat with the next one
            auxFromShortestToLongest = auxFromShortestToLongest[1:]

        if bifurcationsArray.size == 0:
            bifurcationsArray = np.array([[0, 0]], dtype=int)

        bifurcationsArray = np.unique(bifurcationsArray, axis=0)

        segmentsPositionArray = np.ndarray([0, 3], dtype=int)

        # Now we generate the segments from the bifurcation points
        for cellID in range(numberOfCells):
            auxBifurcationsID = np.squeeze(np.array(np.where(bifurcationsArray[:, 0] == cellID)), axis=0)
            for idx in range(len(auxBifurcationsID)):
                if idx == 0: # First segment of the cell
                    segmentsPositionArray = np.append(segmentsPositionArray, [[cellID, 0, bifurcationsArray[auxBifurcationsID[idx], 1]]], axis=0)
                else: # Segments in the middle
                    segmentsPositionArray = np.append(segmentsPositionArray, [[cellID, bifurcationsArray[auxBifurcationsID[idx-1], 1], bifurcationsArray[auxBifurcationsID[idx], 1]]], axis=0)
                if idx == len(auxBifurcationsID) - 1: # Last segment of the cell
                        segmentsPositionArray = np.append(segmentsPositionArray, [[cellID, bifurcationsArray[auxBifurcationsID[idx], 1], -1]], axis=0)

        # We can find the start- and endpoints in RAS coordinates
        segmentsCoordinateArray = np.ndarray([len(segmentsPositionArray), 2, 3])

        for idx in range(len(segmentsCoordinateArray)):
            segmentsCoordinateArray[idx, 0] = cellsCoordinateArray[segmentsPositionArray[idx, 0]][segmentsPositionArray[idx, 1]]
            segmentsCoordinateArray[idx, 1] = cellsCoordinateArray[segmentsPositionArray[idx, 0]][segmentsPositionArray[idx, 2]]

        # With the RAS coordinates, we can eliminate duplicate segments
        _, order = np.unique(segmentsCoordinateArray, return_index=True, axis=0)
        uniqueSegmentsPositionArray = segmentsPositionArray[np.sort(order)]

        # We create the segmentsArray that will contain the non-overlapping cells
        segmentsArray = np.ndarray([len(uniqueSegmentsPositionArray), 2], dtype=object)

        for idx in range(len(uniqueSegmentsPositionArray)):
            if uniqueSegmentsPositionArray[idx, 2] == -1:
                # We have to make a distinction for the cases that end at an endpoint, rather than at a bifurcation
                segmentsArray[idx, 0] = cellsCoordinateArray[uniqueSegmentsPositionArray[idx, 0]][uniqueSegmentsPositionArray[idx, 1]:]
                segmentsArray[idx, 1] = cellsRadiusArray[uniqueSegmentsPositionArray[idx, 0]][uniqueSegmentsPositionArray[idx, 1]:]
            else:    
                segmentsArray[idx, 0] = cellsCoordinateArray[uniqueSegmentsPositionArray[idx, 0]][uniqueSegmentsPositionArray[idx, 1]:uniqueSegmentsPositionArray[idx, 2] + 1]
                segmentsArray[idx, 1] = cellsRadiusArray[uniqueSegmentsPositionArray[idx, 0]][uniqueSegmentsPositionArray[idx, 1]:uniqueSegmentsPositionArray[idx, 2] + 1]

        removeStraights = []
        for idx, segment in enumerate(segmentsArray):
            if len(segment[0]) < 2:
                removeStraights.append(idx)

        segmentsArray = np.delete(segmentsArray, removeStraights, axis=0)
            
        finalSegmentsArray = np.append(finalSegmentsArray, segmentsArray, axis=0)

    if make_plot:
        _ = plt.figure()
        ax = plt.axes(projection='3d')
        for idx in range(len(finalSegmentsArray)):
            ax.plot(-finalSegmentsArray[idx, 0][:, 0], finalSegmentsArray[idx, 0][:, 1], finalSegmentsArray[idx, 0][:, 2])
        ax.legend(range(len(finalSegmentsArray)))
        plt.show()
    
    np.save(os.path.join(caseDir, "segmentsArray.npy"), finalSegmentsArray)

    return finalSegmentsArray