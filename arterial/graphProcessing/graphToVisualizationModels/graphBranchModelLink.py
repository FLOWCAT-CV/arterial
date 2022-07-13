import os
import shutil
import json

import numpy as np
import networkx as nx
import nibabel as nib

import vtk
from vtk.util.numpy_support import vtk_to_numpy

def graphBranchModelLink(caseDir):
    ''' Links all predicted vessel types and cellIds fro segmentsArray to 
    VMTK identifiers for groupIds in branchModel and clippedModel. Generates dictionary
    with groupIds to vessel types and groupIds to vessel filenames for surface segments in 
    os.path.join(caseDir, "surfaceSegments").

    Arguments:
        - caseDir <str or path>: path to the caseDir of the case.
        
    Returns:
    
    '''

    # Load predicted graph, segmentsArray and branchModel
    labeledGraph = nx.read_gpickle(os.path.join(caseDir, "graph_pred.pickle"))
    segmentsArray = np.load(os.path.join(caseDir, "segmentsArray.npy"), allow_pickle=True)
    branchModelPath = os.path.join(caseDir, "branchModel.vtk")

    # Get cellID and predicted nodetype for all edges of predicted graph
    edgeTypes = makeDicts()
    edgeTypesGraph = {}
    for n0, n1 in labeledGraph.edges:
        cellId = int(labeledGraph[n0][n1]["cellId"])
        edgeType = int(labeledGraph[n0][n1]["Vessel type"])
        edgeTypesGraph[cellId] = edgeType

    # Load branch model
    vtkPolyDataReader = vtk.vtkPolyDataReader()
    vtkPolyDataReader.SetFileName(branchModelPath)
    vtkPolyDataReader.Update()
    branchModel = vtkPolyDataReader.GetOutput()
    # Get cell data (blanking and groupId)
    cellData = branchModel.GetCellData()
    cellDataArray = np.ndarray([2, branchModel.GetNumberOfCells()], dtype=np.int64)
    cellDataArray[0] = vtk_to_numpy(cellData.GetArray(2)) # blanking -> indicates if the centerline is inside of clipped surface tract 
    cellDataArray[1] = vtk_to_numpy(cellData.GetArray(3)) # groupId -> clipped surface tractId

    # Define empty arrays to store centerline cells
    branchModelSegments = np.ndarray([branchModel.GetNumberOfCells()], dtype=object)
    branchModelSegmentsIds = np.arange(branchModel.GetNumberOfCells())

    # Store point positions of the centerline cells
    for idx in range(branchModel.GetNumberOfCells()):
        cell = branchModel.GetCell(idx)
        branchModelSegments[idx] = np.ndarray([cell.GetNumberOfPoints(), 3])
        for idx2 in range(cell.GetNumberOfPoints()):
            branchModelSegments[idx][idx2] = cell.GetPoints().GetPoint(idx2)

    # Remove ovelapping segments
    removeRepeats = []
    for idx in range(branchModel.GetNumberOfCells())[1:]:    
        for idx2 in range(idx):
            if len(branchModelSegments[idx]) == len(branchModelSegments[idx2]):
                # 0.5 is hard coded, but it is essentially a measure of similarity. Sometimes overlapping segments from different cells slightly vary in one or several points
                if np.sum(np.abs(branchModelSegments[idx] - branchModelSegments[idx2])) < 0.5 and idx not in removeRepeats:
                    removeRepeats.append(idx)
                    break
    uniqueBranchModelSegments = np.delete(branchModelSegments, removeRepeats)
    uniqueBranchModelSegmentsIds = np.delete(branchModelSegmentsIds, removeRepeats)

    # Search for centerline segments containing bifurcations. We want to divide these cells into parent/children separate cells
    containsBifurcations = []
    for idx in range(len(uniqueBranchModelSegments))[1:]:
        for idx2 in range(idx):
            if np.sum(np.abs(uniqueBranchModelSegments[idx][0] - uniqueBranchModelSegments[idx2][0])) < 0.1:
                containsBifurcations.append([uniqueBranchModelSegmentsIds[idx2], uniqueBranchModelSegmentsIds[idx]])

    # For each bifurcation, we generate 3 new cells with non-over overlapping segments (parent and children)
    branchModelSegmentsWithBifurcations = np.ndarray([branchModel.GetNumberOfCells() + 3 * len(containsBifurcations)], dtype=object)
    branchModelSegmentsWithBifurcations[:branchModel.GetNumberOfCells()] = branchModelSegments
    branchModelSegmentsIdsWithBifurcations = branchModelSegmentsIds
    cellDataArrayWithBifurcations = cellDataArray

    # We initialize a list to remove possible segments that do not continue after the bifurcation point. This is rare but it happens
    removeNones = []
    for idxAux, pairId in enumerate(containsBifurcations):
        for idx in range(len(branchModelSegments[pairId[0]])):
            # We can't find the bifurcation point (these segments overlap all the way, and one of the two ends at the bifurcation while the other one continues)
            if idx == len(branchModelSegments[pairId[0]]) - 1 or idx == len(branchModelSegments[pairId[1]]) - 1:
                if len(branchModelSegments[pairId[0]]) > len(branchModelSegments[pairId[1]]):
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 0] = branchModelSegments[pairId[0]][:idx]
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 1] = branchModelSegments[pairId[0]][idx:]
                    branchModelSegmentsIdsWithBifurcations = np.append(branchModelSegmentsIdsWithBifurcations, [pairId[0], pairId[0], pairId[0]]) # Third doesn't matter, will be removed (but has to be there)
                    cellDataArrayWithBifurcations = np.append(cellDataArrayWithBifurcations, np.transpose(np.array([cellDataArray[:, pairId[0]], cellDataArray[:, pairId[0]], cellDataArray[:, pairId[0]]])), axis=1)
                else:
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 0] = branchModelSegments[pairId[1]][:idx]
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 1] = branchModelSegments[pairId[1]][idx:]
                    branchModelSegmentsIdsWithBifurcations = np.append(branchModelSegmentsIdsWithBifurcations, [pairId[1], pairId[1], pairId[1]])
                    cellDataArrayWithBifurcations = np.append(cellDataArrayWithBifurcations, np.transpose(np.array([cellDataArray[:, pairId[1]], cellDataArray[:, pairId[1]], cellDataArray[:, pairId[1]]])), axis=1)
                # In this case, we directly set blanking as the parent. Other cases would not enter this loop (either len(parent) = 0 [this would enter the previous if] 
                # or segments are equal [these would be removed in the first removeRepeats])
                cellDataArrayWithBifurcations[0, -3] = 1 
                cellDataArrayWithBifurcations[0, -2] = 0
                # Add third slot to removeNones
                removeNones.append(branchModel.GetNumberOfCells() + 3 * idxAux + 2)
                break
            # We can find the bifurcation point (first point where the pair differs)
            else:
                if not (branchModelSegments[pairId[0]][idx] == branchModelSegments[pairId[1]][idx]).all():
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 0] = branchModelSegments[pairId[0]][:idx]
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 1] = branchModelSegments[pairId[0]][idx:]
                    branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 2] = branchModelSegments[pairId[1]][idx:]
                    branchModelSegmentsIdsWithBifurcations = np.append(branchModelSegmentsIdsWithBifurcations, [pairId[0], pairId[0], pairId[1]])
                    cellDataArrayWithBifurcations = np.append(cellDataArrayWithBifurcations, np.transpose(np.array([cellDataArray[:, pairId[0]], cellDataArray[:, pairId[0]], cellDataArray[:, pairId[1]]])), axis=1)
                    # Blanking for child cells set to 1 (unless no overlapping). If no overlapping (len(parent) = 0), only child vessels have blanking = 0
                    if len(branchModelSegments[pairId[0]][:idx]) == 0:
                        cellDataArrayWithBifurcations[0, -3] = 1 
                        cellDataArrayWithBifurcations[0, -2] = 0
                        cellDataArrayWithBifurcations[0, -1] = 0 
                    else:
                        cellDataArrayWithBifurcations[0, -3] = 0 
                        cellDataArrayWithBifurcations[0, -2] = 1 
                        cellDataArrayWithBifurcations[0, -1] = 1
                    break

    # After adding individual segments from those cells containing bifurcations, we have to remove the repeated segments again. We also add the ones from removeNones
    removeRepeats2 = removeNones
    for idx in range(branchModel.GetNumberOfCells() + 3 * len(containsBifurcations))[1:]:    
        for idx2 in range(idx):
            if idx not in removeNones and idx2 not in removeNones and len(branchModelSegmentsWithBifurcations[idx]) == len(branchModelSegmentsWithBifurcations[idx2]):
                # 0.5 is hard coded, but it is essentially a measure of similarity
                if np.sum(np.abs(branchModelSegmentsWithBifurcations[idx] - branchModelSegmentsWithBifurcations[idx2])) < 0.5 and idx not in removeRepeats2:
                    removeRepeats2.append(idx)
                    break
    
    # We delete the repeated segments as well as the original ones containing the bifurcations (these get subbed for the three new ones generated in the previous step)
    uniqueBranchModelSegmentsWithBifurcations = np.delete(branchModelSegmentsWithBifurcations, np.append(removeRepeats2, np.array(containsBifurcations).flatten()))
    uniqueCellDataArrayWithBifurcations = np.delete(cellDataArrayWithBifurcations, np.append(removeRepeats2, np.array(containsBifurcations).flatten()), axis=1)

    # segmentsArray segments are converted through the affine matrix to a unified coordinate system (not in mm, but in voxel coordinates)
    # We repeat the transformation for the branchModel cells to compare both families of cells
    aff = np.linalg.inv(nib.load(os.path.join(caseDir, os.path.basename(caseDir) + ".nii.gz")).affine)
    uniqueBranchModelSegmentsWithBifurcationsAff = np.empty_like(uniqueBranchModelSegmentsWithBifurcations)
    for idx in range(len(uniqueBranchModelSegmentsWithBifurcationsAff)):
        uniqueBranchModelSegmentsWithBifurcationsAff[idx] = np.empty_like(uniqueBranchModelSegmentsWithBifurcations[idx])
        for idx2 in range(len(uniqueBranchModelSegmentsWithBifurcations[idx])):
            uniqueBranchModelSegmentsWithBifurcationsAff[idx][idx2] = np.matmul(aff, np.append(uniqueBranchModelSegmentsWithBifurcations[idx][idx2], 1.0))[:3]

    # We search for overlapping between the branchModel cells and the segmentsArray cells, which contain centerline segments joining two bifurcations
    # segmentsArray cells are those labeled by the GNN
    linkedPairsList = []
    for idx, branchCell in enumerate(uniqueBranchModelSegmentsWithBifurcationsAff):
        if uniqueCellDataArrayWithBifurcations[0, idx] == 0: # Blanking equal to 0
            if searchSequence(segmentsArray, branchCell) is not None:
                linkedPairsList.append([idx, searchSequence(segmentsArray, branchCell)])

    linkedPairs = np.array(linkedPairsList)
    groupIds = uniqueCellDataArrayWithBifurcations[1, linkedPairs[:, 0]]
    linkedUniqueIds = np.arange(len(groupIds))
    uniqueGroupIds, countsGroupIds = np.unique(groupIds, return_counts=True)

    vesselTypes = np.empty_like(linkedPairs[:, 1])
    for idx, cellID in enumerate(linkedPairs[:, 1]):
        vesselTypes[idx] = edgeTypesGraph[cellID]

    deleteIdx = []

    # Current criteria chooses the branchModel cell's groupId depending on 1) Presence of AA and otherwise 2) the radius variation of the associated  
    # segment in the segmentsArray. We look at the radius at the first and middle points of all segmentsArray segments associated to one groupId,
    # and we choose the one with the least variation to be the final groupId chosen
    for idx, groupId in enumerate(uniqueGroupIds):
        if countsGroupIds[idx] > 1:
            auxIds = linkedUniqueIds[groupIds == groupId]
            vesselTypesAux = vesselTypes[groupIds == groupId]
            if 1 in vesselTypesAux:
                for idx2 in range(len(auxIds)):
                    if vesselTypesAux[idx2] != 1:
                        deleteIdx.append(auxIds[idx2])
            else:
                radDiffArray = np.ndarray([len(auxIds)])
                for idx2, auxId in enumerate(auxIds):
                    radDiffArray[idx2] = np.abs(segmentsArray[linkedPairs[auxId, 1], 1][0] - segmentsArray[linkedPairs[auxId, 1], 1][int(len(segmentsArray[linkedPairs[auxId, 1], 1]) / 2)])
                for idx2 in range(len(auxIds)):
                    if radDiffArray[idx2] != np.amin(radDiffArray):
                        deleteIdx.append(auxIds[idx2])

    finalGroupIds = np.delete(groupIds, deleteIdx)
    finalVesselTypes = np.delete(vesselTypes, deleteIdx)

    finalGroupIds = finalGroupIds[np.argsort(finalVesselTypes)]
    finalVesselTypes = np.sort(finalVesselTypes)

    if not os.path.isdir(os.path.join(caseDir, "labeledSegments")): 
        os.mkdir(os.path.join(caseDir, "labeledSegments"))
    else:
        for filename in os.listdir(os.path.join(caseDir, "labeledSegments")):
            os.remove(os.path.join(caseDir, "labeledSegments", filename))

    filenames = np.array(finalVesselTypes, dtype=str)

    # For repeated vesselTypes, we add a number at the end to differentiate between them
    for idx, vesselType in enumerate(finalVesselTypes):
        filenames[idx] = edgeTypes[vesselType]
        if len(finalVesselTypes[finalVesselTypes == vesselType]) > 1:
            identifier = 0
            for idx2, vesselType2 in enumerate(finalVesselTypes):
                if vesselType == vesselType2:
                    filenames[idx2] = edgeTypes[vesselType] + str(identifier)
                    identifier += 1

    for idx, filename in enumerate(filenames):
        if os.path.isfile(os.path.join(caseDir, "surfaceSegments", f"segment{finalGroupIds[idx]}.vtk")):
            shutil.copyfile(os.path.join(caseDir, "surfaceSegments", f"segment{finalGroupIds[idx]}.vtk"), os.path.join(caseDir, "labeledSegments", f"{filename}.vtk"))

    groupIdsToVesselTypesDict = {}
    groupIdsToVesselTypesDict["groupIdsToVesselTypes"] = {}
    groupIdsToVesselTypesDict["groupIdsToVesselFilenames"] = {}
    for idx, _ in enumerate(finalGroupIds):
         groupIdsToVesselTypesDict["groupIdsToVesselTypes"][f"{finalGroupIds[idx]}"] = int(finalVesselTypes[idx])
         groupIdsToVesselTypesDict["groupIdsToVesselFilenames"][f"{finalGroupIds[idx]}"] = filenames[idx]

    with open(os.path.join(caseDir, "groupIdsToVesselTypesDict.json"), "w") as outfile:
        json.dump(groupIdsToVesselTypesDict, outfile, indent=4)


def searchSequence(segmentsArray, branchCell):
    ''' Find a given sequence in a larger array.

    Arguments:
        - segmentsArray <numpy array>: segmentsArray containing all segments between bifurcations.
        - branchCell <numpy array>: smaller array from the branch model that we want to identify in segmentsArray.
        Only arrays with blanking == 0 should be input.
        
    Returns:
        - np.argmin(minimumsArray) <int>: returns the index of the segmentsArray containing the branchCell.
    
    '''

    lenCell = len(branchCell)
    minimumsArray = np.ones([len(segmentsArray)]) * 10000.
    for idx, segment in enumerate(segmentsArray):
        lenSegment = len(segment[0])
        if lenSegment >= lenCell:
            normArray = np.ndarray([lenSegment - lenCell + 1])
            for idx2 in range(lenSegment - lenCell + 1):
                normArray[idx2] = np.linalg.norm(segment[0][idx2:idx2 + lenCell] - branchCell)
                
            minimumsArray[idx] = np.min(normArray)
            
    if np.amin(minimumsArray) < 10:
        return np.argmin(minimumsArray)
    else:
        return None

def makeDicts():

    edgeTypes = {
        0: "other",
        1: "AA",
        2: "BT",
        3: "RCCA",
        4: "LCCA",
        5: "RSA",
        6: "LSA",
        7: "RVA",
        8: "LVA",
        9: "RICA",
        10: "LICA",
        11: "RECA",
        12: "LECA",
        13: "BA"
    }

    return edgeTypes