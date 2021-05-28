import os
import shutil
import vtk
import numpy as np
import networkx as nx
import nibabel as nib

from vtk.util.numpy_support import vtk_to_numpy

def graphBranchModelLink(caseDir):

    graphLabel = nx.read_gpickle(os.path.join(caseDir, "graph_label.pickle"))
    segmentsArray = np.load(os.path.join(caseDir, "segmentsArray.npy"), allow_pickle=True)
    branchModelPath = os.path.join(caseDir, "branchModel.vtk")

    edgeTypes, _ = makeDicts()

    nodeTypesGraph = {}
    edgeTypesGraph = {}

    for node in graphLabel.nodes:
        nodeType = int(graphLabel.nodes(data=True)[node]["nodetype"])
        nodeTypesGraph[node] = nodeType

    for n0, n1 in graphLabel.edges:
        cellId = int(graphLabel[n0][n1]["CellID"])
        edgeType = int(graphLabel[n0][n1]["edgetype"])
        edgeTypesGraph[cellId] = edgeType

    # Load branch model
    vtkPolyDataReader = vtk.vtkPolyDataReader()
    vtkPolyDataReader.SetFileName(branchModelPath)
    vtkPolyDataReader.Update()

    branchModel = vtkPolyDataReader.GetOutput()

    cellData = branchModel.GetCellData()
    cellDataArray = np.ndarray([2, branchModel.GetNumberOfCells()], dtype=np.int64)
    cellDataArray[0] = vtk_to_numpy(cellData.GetArray(2)) # blanking -> transition to a new branch
    cellDataArray[1] = vtk_to_numpy(cellData.GetArray(3)) # groupId -> indicates is the centerline is inside of the tract

    branchModelSegments = np.ndarray([branchModel.GetNumberOfCells()], dtype=object)
    branchModelSegmentsIds = np.arange(branchModel.GetNumberOfCells())

    for idx in range(branchModel.GetNumberOfCells()):
        cell = branchModel.GetCell(idx)
        branchModelSegments[idx] = np.ndarray([cell.GetNumberOfPoints(), 3])
        
        for idx2 in range(cell.GetNumberOfPoints()):
            branchModelSegments[idx][idx2] = cell.GetPoints().GetPoint(idx2)

    removeRepeats = []

    for idx in range(branchModel.GetNumberOfCells())[1:]:    
        for idx2 in range(idx):
            if len(branchModelSegments[idx]) == len(branchModelSegments[idx2]):
                # 0.5 is hard coded, but it is essentially a measure of similarity
                if np.sum(np.abs(branchModelSegments[idx] - branchModelSegments[idx2])) < 0.5 and idx not in removeRepeats:
                    removeRepeats.append(idx)
                    break
            
    uniqueBranchModelSegments = np.delete(branchModelSegments, removeRepeats)
    uniqueBranchModelSegmentsIds = np.delete(branchModelSegmentsIds, removeRepeats)

    # Primer hem de trobar les bifurcacions
    containsBifurcations = []

    for idx in range(len(uniqueBranchModelSegments))[1:]:
        for idx2 in range(idx):
            if np.sum(np.abs(uniqueBranchModelSegments[idx][0] - uniqueBranchModelSegments[idx2][0])) < 0.1:
                containsBifurcations.append([uniqueBranchModelSegmentsIds[idx2], uniqueBranchModelSegmentsIds[idx]])

    # For each bifurcation, we generate 3 new cells with non-over overlapping segments (parent and childs)
    branchModelSegmentsWithBifurcations = np.ndarray([branchModel.GetNumberOfCells() + 3 * len(containsBifurcations)], dtype=object)
    branchModelSegmentsWithBifurcations[:branchModel.GetNumberOfCells()] = branchModelSegments
    branchModelSegmentsIdsWithBifurcations = branchModelSegmentsIds
    cellDataArrayWithBifurcations = cellDataArray

    for idxAux, pairId in enumerate(containsBifurcations):
        for idx in range(len(branchModelSegments[pairId[0]])):
            if not (branchModelSegments[pairId[0]][idx] == branchModelSegments[pairId[1]][idx]).all():
                branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 0] = branchModelSegments[pairId[0]][:idx]
                branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 1] = branchModelSegments[pairId[0]][idx:]
                branchModelSegmentsWithBifurcations[branchModel.GetNumberOfCells() + 3 * idxAux + 2] = branchModelSegments[pairId[1]][idx:]
                branchModelSegmentsIdsWithBifurcations = np.append(branchModelSegmentsIdsWithBifurcations, [pairId[0], pairId[0], pairId[1]])
                cellDataArrayWithBifurcations = np.append(cellDataArrayWithBifurcations, np.transpose(np.array([cellDataArray[:, pairId[0]], cellDataArray[:, pairId[0]], cellDataArray[:, pairId[1]]])), axis=1)
                # Blanking for child cells set to 1 (unless no overlapping). If no overlapping, len(parent) = 0 and only child vessels have blanking = 0
                if len(branchModelSegments[pairId[0]][:idx]) == 0:
                    cellDataArrayWithBifurcations[0, -3] = 1 
                    cellDataArrayWithBifurcations[0, -2] = 0
                    cellDataArrayWithBifurcations[0, -1] = 0 
                else:
                    cellDataArrayWithBifurcations[0, -3] = 0 
                    cellDataArrayWithBifurcations[0, -2] = 1 
                    cellDataArrayWithBifurcations[0, -1] = 1
                break
                
    removeRepeats2 = []
                
    for idx in range(branchModel.GetNumberOfCells() + 3 * len(containsBifurcations))[1:]:    
        for idx2 in range(idx):
            if len(branchModelSegmentsWithBifurcations[idx]) == len(branchModelSegmentsWithBifurcations[idx2]):
                # 0.5 is hard coded, but it is essentially a measure of similarity
                if np.sum(np.abs(branchModelSegmentsWithBifurcations[idx] - branchModelSegmentsWithBifurcations[idx2])) < 0.5 and idx not in removeRepeats2:
                    removeRepeats2.append(idx)
                    break

    # We delete the repeated segments as well as the original ones containing the bifurcations (these get subbed for the three new ones generated in the previous step)
    uniqueBranchModelSegmentsWithBifurcations = np.delete(branchModelSegmentsWithBifurcations, np.append(removeRepeats2, np.array(containsBifurcations).flatten()))
    uniqueCellDataArrayWithBifurcations = np.delete(cellDataArrayWithBifurcations, np.append(removeRepeats2, np.array(containsBifurcations).flatten()), axis=1)

    # segmentsArray segments are converted through the affine matric to a unified coordinate system (not in mm, but in voxels)
    # We repeat the transformation for the branchModel cells to compare both families of cells
    aff = np.linalg.inv(nib.load(os.path.join(caseDir, os.path.basename(caseDir) + ".nii.gz")).affine)

    uniqueBranchModelSegmentsWithBifurcationsAff = np.empty_like(uniqueBranchModelSegmentsWithBifurcations)

    for idx in range(len(uniqueBranchModelSegmentsWithBifurcationsAff)):
        uniqueBranchModelSegmentsWithBifurcationsAff[idx] = np.empty_like(uniqueBranchModelSegmentsWithBifurcations[idx])
        for idx2 in range(len(uniqueBranchModelSegmentsWithBifurcations[idx])):
            uniqueBranchModelSegmentsWithBifurcationsAff[idx][idx2] = np.matmul(aff, np.append(uniqueBranchModelSegmentsWithBifurcations[idx][idx2], 1.0))[:3]

    linkedPairsList = []

    for idx, branchCell in enumerate(uniqueBranchModelSegmentsWithBifurcationsAff):
        if uniqueCellDataArrayWithBifurcations[2, idx] == 0: # Blanking equal to 0
            if searchSequence(segmentsArray, branchCell) is not None:
                linkedPairsList.append([idx, searchSequence(segmentsArray, branchCell)])

    linkedPairs = np.array(linkedPairsList)
    groupIds = uniqueCellDataArrayWithBifurcations[1, linkedPairs[:, 0]]
    linkedUniqueIds = np.arange(len(groupIds))
    lengthLinkedUniqueBranchModelSegmentsWithBifurcationsAff = np.array([len(x) for x in uniqueBranchModelSegmentsWithBifurcationsAff[linkedPairs[:, 0]]])
    uniqueGroupIds, countsGroupIds = np.unique(groupIds, return_counts=True)

    vesselTypes = np.empty_like(linkedPairs[:, 1])
    for idx, cellID in enumerate(linkedPairs[:, 1]):
        vesselTypes[idx] = edgeTypesGraph[cellID]

    deleteIdx = []

    for idx, groupId in enumerate(uniqueGroupIds):
        if countsGroupIds[idx] > 1:
            lengthsAux = lengthLinkedUniqueBranchModelSegmentsWithBifurcationsAff[groupIds == groupId]
            auxIds = linkedUniqueIds[groupIds == groupId]
            for idx2 in range(len(auxIds)):
                if idx2 != np.argmax(lengthsAux):
                    deleteIdx.append(auxIds[idx2])

    finalGroupIds = np.delete(groupIds, deleteIdx)
    finalVesselTypes = np.delete(vesselTypes, deleteIdx)

    finalGroupIds = finalGroupIds[np.argsort(finalVesselTypes)]
    finalVesselTypes = np.sort(finalVesselTypes)

    if not os.path.isdir(os.path.join(caseDir, "labeledSegments")): os.mkdir(os.path.join(caseDir, "labeledSegments"))

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
        shutil.copyfile(os.path.join(caseDir, "surfaceSegments", f"segment{finalGroupIds[idx]}.vtk"), os.path.join(caseDir, "labeledSegments", f"{filename}.vtk"))

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

    nodeTypes = {
        0: "other",
        1: "endpoint",
        2: "AA_BT",
        3: "AA-LCCA",
        4: "AA-LSA",
        5: "AA-RSA",
        6: "BT-LCCA",
        7: "BT-RCCA/RSA",
        8: "RSA-RVA",
        9: "LSA-LVA",
        10: "RCCA-RICA/RECA",
        11: "LCCA-LICA/LECA",
        12: "RVA/LVA-BA",
        13: "BT-LSA",
        14: "AA-BT/LCCA",
        15: "AA-LVA",
        16: "BT-LSA",
        17: "AA-RCCA",
        18: "LCCA-LSA"
    }

    return edgeTypes, nodeTypes