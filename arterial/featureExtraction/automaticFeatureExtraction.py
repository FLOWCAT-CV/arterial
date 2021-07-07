import os
import json

import numpy as np
import networkx as nx
import nibabel as nib

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import math

vesselNameDict = {}
vesselNameDict["AA"] = [1]
vesselNameDict["BT"] = [2, 14] # BT transition (bovine arch) present if BT-LCCA bifurcation present (instead of AA/LCCA) and blanking != 0 somewhere before LCCA
vesselNameDict["RCCA"] = [3]
vesselNameDict["RSA"] = [5]
vesselNameDict["RVA"] = [7, 15] # 15 needs further tuning
vesselNameDict["LCCA"] = [4]
vesselNameDict["LSA"] = [6]
vesselNameDict["LVA"] = [8, 15]

# Number of different bifurcation types
BIFTYPENUM = 19
# Number of different vessel types
VESTYPENUM = 16

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
    13: "BA",
    14: "AA+BT",
    15: "RVA+LVA"  
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

class featureExtractor:

    def __init__(self, caseDir):

        # Set caseDir as attribute
        self.caseDir = caseDir
        # Get affine matrix
        self.aff = nib.load(os.path.join(self.caseDir, os.path.basename(self.caseDir) + ".nii.gz")).affine
        # Load segmentsArray
        self.segmentsArray = np.load(os.path.join(caseDir, "segmentsArray.npy"), allow_pickle=True)
        self.segmentsArrayAff = np.ndarray([len(self.segmentsArray)], dtype = object)
        for idx in range(len(self.segmentsArray)):
            self.segmentsArrayAff[idx] = np.ndarray([len(self.segmentsArray[idx][0]), 3])
            for idx2 in range(len(self.segmentsArray[idx][0])):
                self.segmentsArrayAff[idx][idx2] = np.matmul(self.aff, np.append(self.segmentsArray[idx][0][idx2], 1.0))[:3]
        # Load graph
        self.graph = nx.read_gpickle(os.path.join(caseDir, "graph_pred.pickle"))
        # Get cellId to segmentId dict
        self.cellIdToVesselType = {}
        for n0, n1 in self.graph.edges:
            self.cellIdToVesselType[self.graph[n0][n1]["CellID"]] = self.graph[n0][n1]["edgetype"]
        # Load branchModel
        vtkPolyDataReader = vtk.vtkPolyDataReader()
        vtkPolyDataReader.SetFileName(os.path.join(caseDir, "branchModel.vtk"))
        vtkPolyDataReader.Update()
        self.branchModel = vtkPolyDataReader.GetOutput()
        # Pool branchModel point coordinates. Get Blanking and GroupId for each point
        self.branchModelCoordinates = np.ndarray([self.branchModel.GetNumberOfPoints(), 3])
        self.blanking = np.ndarray([self.branchModel.GetNumberOfPoints()])
        self.groupIdBranchModel = np.ndarray([self.branchModel.GetNumberOfPoints()], dtype = int)
        accumulatedNumberOfPoints = 0
        for idx in range(self.branchModel.GetNumberOfCells()):         
            for idx2 in range(self.branchModel.GetCell(idx).GetNumberOfPoints()):
                self.branchModelCoordinates[idx2 + accumulatedNumberOfPoints] = self.branchModel.GetCell(idx).GetPoints().GetPoint(idx2)
                self.blanking[idx2 + accumulatedNumberOfPoints] = vtk_to_numpy(self.branchModel.GetCellData().GetArray("Blanking"))[idx]
                self.groupIdBranchModel[idx2 + accumulatedNumberOfPoints] = vtk_to_numpy(self.branchModel.GetCellData().GetArray("GroupIds"))[idx]
            accumulatedNumberOfPoints += self.branchModel.GetCell(idx).GetNumberOfPoints()
        # Get radius
        self.radius = vtk_to_numpy(self.branchModel.GetPointData().GetArray("Radius"))
        # Load clippedModel
        vtkPolyDataReader = vtk.vtkPolyDataReader()
        vtkPolyDataReader.SetFileName(os.path.join(caseDir, "clippedModel.vtk"))
        vtkPolyDataReader.Update()
        self.clippedModel = vtkPolyDataReader.GetOutput()
        # Pool clippedModel point coordinates
        self.clippedModelCoordinates = np.ndarray([self.clippedModel.GetNumberOfPoints(), 3])
        accumulatedNumberOfPoints = 0
        for idx in range(self.clippedModel.GetNumberOfPoints()):         
            self.clippedModelCoordinates[idx] = self.clippedModel.GetPoints().GetPoint(idx)
        # Get GroupId array for clippedModel
        self.groupIdArrayClippedModel = vtk_to_numpy(self.clippedModel.GetPointData().GetArray("GroupIds"))
        # Load groupIdsToVesselTypesDict
        with open(os.path.join(caseDir, "groupIdsToVesselTypesDict.json"),) as jsonFile:
            self.groupIdsToVesselTypesDict = json.load(jsonFile)["groupIdsToVesselTypes"]

        # Create dict for the results
        self.featureExtractorDict = {}
        # Create second dict for a broader inclusion of results
        self.featureExtractorExtendedDict = {}

    def extractFeatures(self):
        # Get presence of bovine AA
        self.featureExtractorDict["presence of bovine arch"] = self.getBovineArch()
        self.featureExtractorExtendedDict["presence of bovine arch"] = self.getBovineArch()
        # Get presence of ARSA
        self.featureExtractorDict["presence of ARSA"] = self.getARSA()
        self.featureExtractorExtendedDict["presence of ARSA"] = self.getARSA()

        # Get proximal diameters
        for vesselName in vesselNameDict.keys():
            self.featureExtractorDict[f"{vesselName} proximal diameter"] = 2 * self.findProximalDiameter(vesselName)
            self.featureExtractorExtendedDict[f"{vesselName} proximal diameter"] = 2 * self.findProximalDiameter(vesselName)

        for vesselName in vesselNameDict.keys():
            if vesselName != "AA":
            # Get relative lenghts
                self.featureExtractorDict[f"{vesselName} relative length"] = self.computeRelativeLength(vesselName)
                self.featureExtractorExtendedDict[f"{vesselName} relative length"] = self.computeRelativeLength(vesselName)
                # Get absolute departure angles
                polar, azimuth = self.computeAbsoluteAngle(vesselName)
                self.featureExtractorDict[f"{vesselName} abs polar angle"] = polar
                self.featureExtractorDict[f"{vesselName} abs azmiuth angle"] = azimuth
                self.featureExtractorExtendedDict[f"{vesselName} abs polar angle"] = polar
                self.featureExtractorExtendedDict[f"{vesselName} abs azimuth angle"] = azimuth
                # Get relative departure angles
                polarRel, azimuthRel = self.computeRelativeAngle(vesselName)
                self.featureExtractorDict[f"{vesselName} rel polar angle"] = polarRel
                self.featureExtractorDict[f"{vesselName} rel azmiuth angle"] = azimuthRel
                self.featureExtractorExtendedDict[f"{vesselName} rel polar angle"] = polarRel
                self.featureExtractorExtendedDict[f"{vesselName} rel azimuth angle"] = azimuthRel
        
        # Get AA type
        self.featureExtractorDict["AA type"] = self.findAAType()
        self.featureExtractorExtendedDict["AA type"] = self.findAAType()

        with open(os.path.join(self.caseDir, f"features.json"), "w") as outfile:
            json.dump(self.featureExtractorDict, outfile, indent=4)
        
        with open(os.path.join(self.caseDir, f"extendedFeatures.json"), "w") as outfile:
            json.dump(self.featureExtractorExtendedDict, outfile, indent=4)

    def getBovineArch(self):
        ''' Returns True if bovine arch is detected, which is found when LCCA bifurcates from BT.
        It is frequent for the LCCA to depart from the segmentsArray's detected BT, so the computation is
        based around detecting if a BT transition segment is found.

        Arguments:

        Returns:
            self.bovineArch <bool>: True if bovine arch if found. False otherwise.

        '''
        # Select all those segments Ids from segmentsArray that make up the LCCA
        segmentIds = []
        for idx in self.cellIdToVesselType:
            if self.cellIdToVesselType[idx] in vesselNameDict["LCCA"]:
                segmentIds.append(idx)
        # If no segments are detected, output False for self.bovineArch
        if len(segmentIds) == 0:
            return False
        # Otherwise, perform computation
        else:
            # Get singleSegment, coordinatesArray and segmentsOrder
            singleSegment, _ = self.getSingleSegment("LCCA", segmentIds)
            # We use bifurcation point to find 
            bifurcationPoint = singleSegment[0]
            closestSegmentsPoints = np.ndarray([len(self.segmentsArrayAff) - len(segmentIds), 3])
            closestSegmentsCellIds = np.ndarray([len(self.segmentsArrayAff) - len(segmentIds)], dtype = int)
            idxAux = 0
            # Get closest points from all other segments from segmentsArray to bifurcation point of vesselName segment
            for idx in range(len(self.segmentsArrayAff)):
                if idx not in segmentIds:
                    closestSegmentsPoints[idxAux] = self.segmentsArrayAff[idx][np.argmin(np.linalg.norm(self.segmentsArrayAff[idx] - bifurcationPoint, axis = 1))]
                    closestSegmentsCellIds[idxAux] = idx
                    idxAux += 1
            # Get segments in contact with bifurcation node
            segmentsInContactIdx = []
            for idx, point in enumerate(closestSegmentsCellIds):
                if np.linalg.norm(point - bifurcationPoint) < 0.1:
                    segmentsInContactIdx.append(idx)

            # Get cellId (position in segmentsArray), and vesselTypes
            segmentsInContactCellIds = list(closestSegmentsCellIds[segmentsInContactIdx])
            segmentsInContactVesselTypes = [self.cellIdToVesselType[cellId] for cellId in segmentsInContactCellIds]

            # If LCCA is connected to AA, no bovine arch is found
            if 1 in segmentsInContactVesselTypes:
                return False
            # If LCCA in contact with BT (2 or 14) and not with AA (1), we have to determine wether a BT transition segment is present or not
            elif 2 in segmentsInContactVesselTypes or 14 in segmentsInContactVesselTypes:
                # THIS IS PROVISIONAL
                return True
            # In all other cases, return False
            else:
                return False

    def getARSA(self):
        ''' Returns True if ARSA is detected, which is found when RSA bifurcates from AA.

        Arguments:

        Returns:
            self.ARSA <bool>: True if ARSA if found. False otherwise.

        '''
        # Select all those segments Ids from segmentsArray that make up the RSA
        segmentIds = []
        for idx in featureExtractor.cellIdToVesselType:
            if featureExtractor.cellIdToVesselType[idx] in vesselNameDict["RSA"]:
                segmentIds.append(idx)
        # If no segments are detected, output False for self.ARSA
        if len(segmentIds) == 0:
            return False
        # Otherwise, perform computation
        else:
            # Get singleSegment, coordinatesArray and segmentsOrder
            singleSegment, _ = featureExtractor.getSingleSegment("RSA", segmentIds)
            # We use bifurcation point to find 
            bifurcationPoint = singleSegment[0]
            closestSegmentsPoints = np.ndarray([len(self.segmentsArrayAff) - len(segmentIds), 3])
            closestSegmentsCellIds = np.ndarray([len(self.segmentsArrayAff) - len(segmentIds)], dtype = int)
            idxAux = 0
            # Get closest points from all other segments from segmentsArray to bifurcation point of vesselName segment
            for idx in range(len(self.segmentsArrayAff)):
                if idx not in segmentIds:
                    closestSegmentsPoints[idxAux] = self.segmentsArrayAff[idx][np.argmin(np.linalg.norm(self.segmentsArrayAff[idx] - bifurcationPoint, axis = 1))]
                    closestSegmentsCellIds[idxAux] = idx
                    idxAux += 1
            # Get segments in contact with bifurcation node
            segmentsInContactIdx = []
            for idx, point in enumerate(closestSegmentsCellIds):
                if np.linalg.norm(point - bifurcationPoint) < 0.1:
                    segmentsInContactIdx.append(idx)

            # Get cellId (position in segmentsArray) and vesselTypes
            segmentsInContactCellIds = list(closestSegmentsCellIds[segmentsInContactIdx])
            segmentsInContactVesselTypes = [self.cellIdToVesselType[cellId] for cellId in segmentsInContactCellIds]

            # If one of the segments in the contact node with RSA is AA (only type 1) return True for self.ARSA
            if 1 in segmentsInContactVesselTypes:
                return True
            # Otherwise, return False
            else:
                return False

    def findAAType(self):
        ''' Find AA type by computing the ratio between the difference in the S (axial) coordinate of the highest
        point of the AA and the BT origin, and the LCCA diameter. There are three possible types: 1 (0 > ratio >= 1), 2 
        (1 > ratio >= 2) and 3 (ratio > 2).

        Arguments:

        Returns:
            - AA type (int or math.nan).

        '''
        # Pool all segments with type 1 or 14
        AAsegmentIds = []
        for idx in self.cellIdToVesselType: 
            if self.cellIdToVesselType[idx] in [1, 14]:
                AAsegmentIds.append(idx)
        # If AA, BT and LCCA exist, compute AA type
        if not math.isnan(self.featureExtractorDict[f"BT origin"][0]) and not math.isnan(self.featureExtractorDict[f"LCCA proximal diameter"]) and not len(AAsegmentIds) == 0:
            # Get BT origin S coordinate
            BTOriginS = self.featureExtractorDict[f"BT origin"][2]
            # Get LCCA proximal diameter
            LCCADiameter = self.featureExtractorDict[f"LCCA proximal diameter"]
            # Pool all clippedModel AA points into one array (regardless of order) (corresponding groupId has to correspond to type 1)
            AAClippedModelCoordinatesArray = np.array([self.clippedModelCoordinates[idx] for idx in range(len(self.groupIdArrayClippedModel)) if self.groupIdsToVesselTypesDict[str(self.groupIdArrayClippedModel[idx])] == 1])
            # Find AA clippedModel` point with highest S coordinate
            highestAAPointS = np.amax(AAClippedModelCoordinatesArray[:, 2])
            # Compute ratio
            if (highestAAPointS - BTOriginS) / float(LCCADiameter) <= 1.0:
                return 1
            elif (highestAAPointS - BTOriginS) / float(LCCADiameter) > 1.0 and (highestAAPointS - BTOriginS) / LCCADiameter <= 2.0:
                return 2
            else:
                return 3
        # Otherwise, return nan
        else:
            return math.nan
    
    def findProximalDiameter(self, vesselName):
        ''' Finds proximal diameter of the given vesselName (if present). For the AA, the diameter return corresponds
        to the highest AA point (same criteria as done for manual measurements).

        Arguments:
            - vesselName <str>: name of vessel as in vesselNameDict.

        Returns:
            - Proximal diameter corresponding to vesselName (float, nan if not found).

        '''
        if vesselName == "AA":
            # For AA, just pool all segments with type 1 or 14
            AAsegmentIds = []
            numberOfPoints = 0
            for idx in self.cellIdToVesselType: 
                if self.cellIdToVesselType[idx] in [1, 14]:
                    AAsegmentIds.append(idx)
                    numberOfPoints += len(self.segmentsArray[idx][0])
            # If AA not found, output nan
            if len(AAsegmentIds) == 0:
                print(f"   {vesselName} not present")
                return math.nan
            # Otherwise, find point with highest S coordinate (vertical/axial axis) and record associates dimateter   
            else:
                # Pool all AA coordinates into one array (regardless of order)
                AACoordinatesArray = np.ndarray([numberOfPoints, 3], dtype=float)
                auxIdx = 0
                for idx, AAsegmentId in enumerate(AAsegmentIds):
                    for idx2 in range(len(self.segmentsArray[AAsegmentId][0])):
                        AACoordinatesArray[idx2 + auxIdx] = np.matmul(self.aff, np.append(self.segmentsArray[AAsegmentId][0][idx2], 1.0))[:3]
                    auxIdx += idx2
                # Find point with highest S coordinate
                highestAAPointId = findPointId(AACoordinatesArray[np.argmax(AACoordinatesArray[:, 2])], self.branchModelCoordinates)
                # Return radius of that point
                return self.radius[highestAAPointId]
        else:
            # Select all those segments Ids from segmentsArray that make up the vessel
            segmentIds = []
            for idx in self.cellIdToVesselType:
                if self.cellIdToVesselType[idx] in vesselNameDict[vesselName]:
                    segmentIds.append(idx)
            # If no segments are detected, output a nan value
            if len(segmentIds) == 0:
                self.featureExtractorDict[f"{vesselName} origin"] = math.nan
                self.featureExtractorExtendedDict[f"{vesselName} origin"] = math.nan
                return math.nan
            # Otherwise, perform computation
            else:
                # Get singleSegment
                singleSegment, _ = self.getSingleSegment(vesselName, segmentIds)
                # Start interating from the most proximal point until a point is found where:
                    # 1) Point is in branchModel
                    # 2) Point has blanking 0
                    # 3) vesselName exists for this case
                    # 4) groupId in branch model is the same as found by graphBranchModelLink
                for idx in range(len(singleSegment)):
                    proximalEndId = findPointId(singleSegment[idx], self.branchModelCoordinates)
                    if self.blanking[proximalEndId] == 0 and str(self.groupIdBranchModel[proximalEndId]) in self.groupIdsToVesselTypesDict.keys() and self.groupIdsToVesselTypesDict[str(self.groupIdBranchModel[proximalEndId])] in vesselNameDict[vesselName]:
                        break
                # Record coordinates of origin
                self.featureExtractorDict[f"{vesselName} origin"] = list(self.branchModelCoordinates[proximalEndId])
                self.featureExtractorExtendedDict[f"{vesselName} origin"] = list(self.branchModelCoordinates[proximalEndId])
                # Return radius of the most proximal point found
                return self.radius[proximalEndId]

    def computeRelativeLength(self, vesselName):
        ''' Computes relative length (RL) of vesselName segment (AA not included).

        Arguments:
            - vesselName <str>: name of vessel as in vesselNameDict.

        Returns:
            - RL corresponding to vesselName (nan if not found).

        '''

        def distanceAlongCenterline(centerline):
            distance = 0
            for idx in range(1, len(centerline)):
                distance += np.linalg.norm(centerline[idx] - centerline[idx - 1])
                
            return distance

        # Select all those segments Ids from segmentsArray that make up the vessel
        segmentIds = []
        for idx in self.cellIdToVesselType:
            if self.cellIdToVesselType[idx] in vesselNameDict[vesselName]:
                segmentIds.append(idx)
        # If no segments are detected, output a nan value
        if len(segmentIds) == 0:
            print(f"   {vesselName} not present")
            self.featureExtractorExtendedDict[f"{vesselName} distal bifurcation"] = math.nan
            return math.nan

        # Otherwise, perform computation
        else:
            # Get singleSegment, coordinatesArray and segmentsOrder
            singleSegment, orderedCoordinatesArray = self.getSingleSegment(vesselName, segmentIds)
            # Get start and endpoints from dict
            startpoint = np.array(self.featureExtractorDict[f"{vesselName} origin"])
            # For the R/LSA, just use closest point to R/LVA origin
            if vesselName == "RSA" and not math.isnan(self.featureExtractorDict[f"RVA origin"][0]):
                VAOrigin = np.array(self.featureExtractorDict[f"RVA origin"])
                endpoint = singleSegment[np.argmin(np.linalg.norm(singleSegment - VAOrigin, axis = 1))]
            elif vesselName == "LSA" and not math.isnan(self.featureExtractorDict[f"LVA origin"][0]):
                VAOrigin = np.array(self.featureExtractorDict[f"LVA origin"])
                endpoint = singleSegment[np.argmin(np.linalg.norm(singleSegment - VAOrigin, axis = 1))]
            # For the rest of vessel types, use all segments found
            else:
                endpoint = singleSegment[-1]
            # Record distal bifurcation coordinates
            self.featureExtractorExtendedDict[f"{vesselName} distal bifurcation"] = list(endpoint)
            # Get signeSegments arguments for closest points to start- and endpoints
            startSingleSegmentArg = np.argmin(np.linalg.norm(singleSegment - startpoint, axis = 1))
            endSingleSegmentArg = np.argmin(np.linalg.norm(singleSegment - endpoint, axis = 1))
            # Compute euclidean distance and distance along centerline
            euclideanDistance = np.linalg.norm(singleSegment[startSingleSegmentArg] - singleSegment[endSingleSegmentArg])
            centerlineDistance = distanceAlongCenterline(singleSegment[startSingleSegmentArg:endSingleSegmentArg + 1])
            # Return RL
            return euclideanDistance / centerlineDistance

    def computeAbsoluteAngle(self, vesselName):
        ''' Computes absolute departure angle of vesselName segment (AA not included).

        Arguments:
            - vesselName <str>: name of vessel as in vesselNameDict.

        Returns:
            - polarRel <float>: polar component in spherical coordinates of absolute departure angle.
            - azimuthRel <float>: azimuthal component in spherical coordinates of absolute departure angle.

        '''
        # Select all those segments Ids from segmentsArray that make up the vessel
        segmentIds = []
        for idx in self.cellIdToVesselType:
            if self.cellIdToVesselType[idx] in vesselNameDict[vesselName]:
                segmentIds.append(idx)
        # If no segments are detected, output a nan value
        if len(segmentIds) == 0:
            print(f"   {vesselName} not present")
            self.featureExtractorExtendedDict[f"{vesselName} abs angle point"] = math.nan
            return math.nan, math.nan

        # Otherwise, perform computation
        else:
            # Get singleSegment, coordinatesArray and segmentsOrder
            singleSegment, _ = self.getSingleSegment(vesselName, segmentIds)
            # Get origin
            origin =  np.array(self.featureExtractorDict[f"{vesselName} origin"])
            # Get origin point id
            originId = findPointId(origin, singleSegment)
            # Get diameter
            diameter = np.array(self.featureExtractorDict[f"{vesselName} proximal diameter"])
            # Search for distal point on the centerline at a distance closest to diameter
            absPoint = singleSegment[originId + np.argmin(np.abs(np.linalg.norm(singleSegment[originId:] - origin, axis = 1) - diameter))]
            self.featureExtractorExtendedDict[f"{vesselName} abs angle point"] = list(absPoint)
            # Compute polar and azimuth 
            polar, azimuth = absSphericalAnglesFrom3DCartesian(absPoint - origin)

            return polar, azimuth

    def computeRelativeAngle(self, vesselName):
        ''' Computes relative departure angle of vesselName segment (AA not included) with respect to the
        immediately preceeding vessel.

        Arguments:
            - vesselName <str>: name of vessel as in vesselNameDict.

        Returns:
            - polar <float>: polar component in spherical coordinates of relative departure angle.
            - azimuth <float>: azimuthal component in spherical coordinates of relative departure angle.

        '''
       # Select all those segments Ids from segmentsArray that make up the vessel
        segmentIds = []
        for idx in self.cellIdToVesselType:
            if self.cellIdToVesselType[idx] in vesselNameDict[vesselName]:
                segmentIds.append(idx)
        # If no segments are detected, output a nan value
        if len(segmentIds) == 0:
            self.featureExtractorExtendedDict[f"{vesselName} rel angle point"] = math.nan
            return math.nan, math.nan

        # Otherwise, perform computation
        else:
            # Get singleSegment, coordinatesArray and segmentsOrder
            singleSegment, _ = self.getSingleSegment(vesselName, segmentIds)
            # Get origin
            origin =  np.array(self.featureExtractorDict[f"{vesselName} origin"])

            # We use bifurcation point to find connected segments
            bifurcationPoint = singleSegment[0]
            closestSegmentsPoints = np.ndarray([len(self.segmentsArrayAff) - len(segmentIds), 3])
            closestSegmentsCellIds = np.ndarray([len(self.segmentsArrayAff) - len(segmentIds)], dtype = int)
            idxAux = 0
            # Get closest points from all other segments from segmentsArray to bifurcation point of vesselName segment
            for idx in range(len(self.segmentsArrayAff)):
                if idx not in segmentIds:
                    closestSegmentsPoints[idxAux] = self.segmentsArrayAff[idx][np.argmin(np.linalg.norm(self.segmentsArrayAff[idx] - bifurcationPoint, axis = 1))]
                    closestSegmentsCellIds[idxAux] = idx
                    idxAux += 1
            # Get segments in contact with bifurcation node and AA segments
            potentialRelPointsIdx = []
            for idx, cellId in enumerate(closestSegmentsCellIds):
                if np.linalg.norm(closestSegmentsPoints[idx] - bifurcationPoint) < 0.1 or self.cellIdToVesselType[cellId] == 1:
                    potentialRelPointsIdx.append(idx)

            # Get cellId (position in segmentsArray), vesselTypes and diameters for corresponding segments
            potentialRelPointsCellIds = list(closestSegmentsCellIds[potentialRelPointsIdx])
            potentialRelPointsVesselTypes = [self.cellIdToVesselType[cellId] for cellId in potentialRelPointsCellIds]
            potentialRelPointsDiameters = [self.featureExtractorDict[f"{edgeTypes[vesselType]} proximal diameter"] for vesselType in potentialRelPointsVesselTypes]
            # Filter out those that do not have a point within a diameter from the origin
            deleteFarSegments = []
            potentialRelPoints = []
            for idx, cellId in enumerate(potentialRelPointsCellIds):
                if np.amin(np.abs(np.linalg.norm(self.segmentsArrayAff[cellId] - origin, axis = 1) - potentialRelPointsDiameters[idx])) > 1.:
                    deleteFarSegments.append(idx)
                else:
                    potentialRelPoints.append(self.segmentsArrayAff[cellId][np.argmin(np.abs(np.linalg.norm(self.segmentsArrayAff[cellId] - origin, axis = 1) - potentialRelPointsDiameters[idx]))])
            potentialRelPointsCellIds = np.delete(potentialRelPointsCellIds, deleteFarSegments)
            potentialRelPointsVesselTypes = np.delete(potentialRelPointsVesselTypes, deleteFarSegments)
            potentialRelPointsDiameters = np.delete(potentialRelPointsDiameters, deleteFarSegments)
            potentialRelPoints = np.array(potentialRelPoints)

            # Now, for each vesselType, different rules are set to select the point to compute the relative angle point. This point should be placed on the centerline coming from the catheter path, in most cases on the parent vessel
            # For the BT, the point is chosen on the AA, on the centerline point closest to the coordinates origin of the image (the [0, 0, 0] in voxel (RAS) coordinates, the lowmost, posterior-most and leftmost point)
            if vesselName == "BT":
                # Check only type 1 vessels, ignore type 14
                if 1 in list(potentialRelPointsVesselTypes):
                    relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 1][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 1] - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                # If AA not found, return nan
                else:
                    relPoint = math.nan
            # For RCCA, check if ARSA
            if vesselName == "RCCA":
                # If ARSA, RCCA should depart from the AA (1 or 14) (no BT)
                if self.ARSA:
                    if 1 in list(potentialRelPointsVesselTypes) or 14 in list(potentialRelPointsVesselTypes):
                        potentialRelPointsAux = [potentialRelPoints[idx] for idx in range(len(potentialRelPointsVesselTypes)) if potentialRelPointsVesselTypes[idx] in [1, 14]]
                        relPoint = potentialRelPointsAux[np.argmin(np.linalg.norm(potentialRelPointsAux - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                    # If AA not found, return nan
                    else:
                        relPoint = math.nan
                # If not ARSA, search for BT points. Select closest to AA
                else:
                    if 2 in list(potentialRelPointsVesselTypes) or 14 in list(potentialRelPointsVesselTypes):
                        potentialRelPointsAux = [potentialRelPoints[idx] for idx in range(len(potentialRelPointsVesselTypes)) if potentialRelPointsVesselTypes[idx] in [2, 14]]
                        relPoint = potentialRelPointsAux[np.argmin(np.linalg.norm(potentialRelPointsAux - self.getClosestAAPoint(self, origin), axis = 1))]
                    # If BT not found, return nan
                    else:
                        relPoint = math.nan
            # For RSA, check if ARSA
            if vesselName == "RSA":
                # If ARSA, RSA should depart from AA (only type 1)
                if self.ARSA:
                    if 1 in list(potentialRelPointsVesselTypes):
                        relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 1][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 1] - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                    # If AA not found, return nan
                    else:
                        relPoint = math.nan
                # If not ARSA, search for BT points. Select closest to AA
                else:
                    if 2 in list(potentialRelPointsVesselTypes) or 14 in list(potentialRelPointsVesselTypes):
                        potentialRelPointsAux = [potentialRelPoints[idx] for idx in range(len(potentialRelPointsVesselTypes)) if potentialRelPointsVesselTypes[idx] in [2, 14]]
                        relPoint = potentialRelPointsAux[np.argmin(np.linalg.norm(potentialRelPointsAux - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                    # If BT not found, return nan
                    else:
                        relPoint = math.nan
            # For RVA, search for points in RSA and select closest to AA       
            if vesselName == "RVA":
                if 5 in list(potentialRelPointsVesselTypes):
                    relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 5][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 5] - self.getClosestAAPoint(self, origin), axis = 1))]
                # If AA not found, return nan
                else:
                    relPoint = math.nan
            # For LCCA, check if bovine arch     
            if vesselName == "LCCA":
                # If bovine arch, first check if any points fall into BT transition (types 2, 14) and select closest to AA
                if self.bovineArch:
                    if 2 in list(potentialRelPointsVesselTypes) or 14 in list(potentialRelPointsVesselTypes):
                        potentialRelPointsAux = [potentialRelPoints[idx] for idx in range(len(potentialRelPointsVesselTypes)) if potentialRelPointsVesselTypes[idx] in [2, 14]]
                        relPoint = potentialRelPointsAux[np.argmin(np.linalg.norm(potentialRelPointsAux - self.getClosestAAPoint(self, origin), axis = 1))]
                    # If no points fall into BT transition (this could be because BT transition is too short, for example), select points over AA centerline. Select point closest to coordinates origin
                    elif 1 in list(potentialRelPointsVesselTypes):
                        relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 1][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 1] - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                    # If BT and AA are not found, return nan
                    else:
                        relPoint = math.nan
                # In not bovine arch, search for AA points (1 or 14) and select closest to coordinates origin
                else:
                    if 1 in list(potentialRelPointsVesselTypes) or 14 in list(potentialRelPointsVesselTypes):
                        potentialRelPointsAux = [potentialRelPoints[idx] for idx in range(len(potentialRelPointsVesselTypes)) if potentialRelPointsVesselTypes[idx] in [1, 14]]
                        relPoint = potentialRelPointsAux[np.argmin(np.linalg.norm(potentialRelPointsAux - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                    # If AA not found, return nan
                    else:
                        relPoint = math.nan
            # For LSA, search for AA points (1) and return closest to origin
            if vesselName == "LSA":
                if 1 in list(potentialRelPointsVesselTypes):
                    relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 1][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 1] - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                # If AA not found, return nan
                else:
                    relPoint = math.nan
            # For LVA, first check if it departs from AA (rare case, but possible)    
            if vesselName == "LVA":
                # If this is the case (distance from bifurcation point and closest AA point within 0.1 mm), search for AA points and select closest to origin
                if np.linalg.norm(self.getClosestAAPoint(self, bifurcationPoint) - bifurcationPoint) < 0.1:
                    if 1 in list(potentialRelPointsVesselTypes):
                        relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 1][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 1] - np.matmul(self.aff, [0.0, 0.0, 0.0, 1.0])[:3], axis = 1))]
                    # If AA not found, return nan
                    else:
                        relPoint = math.nan
                # If LVA departs from LSA (standard), check for LSA points and select closest to AA
                elif 6 in list(potentialRelPointsVesselTypes):
                    relPoint = potentialRelPoints[potentialRelPointsVesselTypes == 5][np.argmin(np.linalg.norm(potentialRelPoints[potentialRelPointsVesselTypes == 5] - self.getClosestAAPoint(self, origin), axis = 1))]
                # If LSA not found and LVA does not depart from AA, return nan
                else:
                    relPoint = math.nan

            # Check if relPoint found has three coordinates (otherwise it will be nan)
            if len(relPoint) == 3:
                self.featureExtractorExtendedDict[f"{vesselName} rel angle point"] = list(relPoint)
                # Compute relative polar and azimuth 
                polarRel, azimuthRel = absSphericalAnglesFrom3DCartesian(relPoint - origin)
                # Get absolute polar and azimuth
                polarAbs = self.featureExtractorDict[f"{vesselName} abs polar angle"]
                azimuthAbs = self.featureExtractorDict[f"{vesselName} abs azmiuth angle"]
                # Get difference between both (this is the actual relative angle)
                polarRel -= polarAbs
                azimuthRel -= azimuthAbs
                # Polar has to be comprised between 0 and pi
                if polarRel > math.pi:
                    polarRel -= math.pi
                # Azimuth has to be comprised between -pi and pi
                if azimuthRel < 0.0 - math.pi:
                    azimuthRel += 2 * math.pi
                elif azimuthRel > math.pi:
                    azimuthRel -= 2 * math.pi

                return polarRel, azimuthRel
            # If relPoint is nan, return nan and nan 
            else:
                self.featureExtractorExtendedDict[f"{vesselName} rel angle point"] = math.nan
                return math.nan, math.nan

    def getSingleSegment(self, vesselName, segmentIds):
        ''' Using the collected segmentIds from a given vesselName,
        returns single numpy array with all coordinates from segmentsArray of
        the corresponding vesselName, as well as these divided into different arrays 
        and their orientation.

        Arguments:
            - vesselName <str>: name of vessel as in vesselNameDict.
            - segmentIds <list>: list with all cellIds from segmentsArray that correspond to vesselName.

        Returns:
            - singleSegment <numpy array>: array with all coordinates from segmentsArray, oriented 
            with respect to AA>
            - orderedCoordinatesArray <numpy array>: array with dtype=object with alla ordered arrays 
            from segmentsArray separated and flipped if necessary.

        '''
        def orderCoordinatesArray(coordinatesArray, AACoordinatesArray):
            # Define number of segments
            numberOfSegments = len(coordinatesArray)
            # Initialize arrays
            startPoints = np.ndarray([numberOfSegments, 3])
            endPoints = np.ndarray([numberOfSegments, 3])
            possibleReferences = np.ndarray([2 * numberOfSegments, 3])
            possibleReferencesDistances = np.ndarray([2 * numberOfSegments])
            for idx in range(numberOfSegments):
                # Get start and enpoint coordinates from segmentsArray
                startPoints[idx] = coordinatesArray[idx][0]
                endPoints[idx] = coordinatesArray[idx][-1]
                # Get closest AA point from each endpoint for each segment with the target vesselType and the corrsponding distances
                possibleReferences[2 * idx] = AACoordinatesArray[np.argmin(np.linalg.norm(AACoordinatesArray - startPoints[idx], axis = 1))]
                possibleReferences[2 * idx + 1] = AACoordinatesArray[np.argmin(np.linalg.norm(AACoordinatesArray - endPoints[idx], axis = 1))]
                possibleReferencesDistances[2 * idx] = np.amin(np.linalg.norm(AACoordinatesArray - startPoints[idx], axis = 1))
                possibleReferencesDistances[2 * idx + 1] = np.amin(np.linalg.norm(AACoordinatesArray - endPoints[idx], axis = 1))
            
            # Select closest AA point as reference (this is used for orientation)
            reference = possibleReferences[np.argmin(possibleReferencesDistances)]
            # Initialize ordered coordinatesArray
            orderedCoordinatesArray = np.ndarray([numberOfSegments], dtype=object)
            # auxIds is needed
            auxIds = np.arange(numberOfSegments)
            # We want to store order of coordinatesArray
            segmentsOrder = np.empty_like(auxIds)
            # Order and check if any segments need flipping
            for idx in range(numberOfSegments):
                # First, select which has to be the first segment (with start- or endpoint closest to reference)
                startPointsRefDistance = np.linalg.norm(startPoints - reference, axis = 1)
                endPointsRefDistance = np.linalg.norm(endPoints - reference, axis = 1)
                startArgmin = np.argmin(startPointsRefDistance)
                endArgmin = np.argmin(endPointsRefDistance)
                # Check if flipping is needed (if endpoint is closer to AA than startpoint)
                if startPointsRefDistance[startArgmin] <= endPointsRefDistance[endArgmin]: # No flipping needed
                    orderedCoordinatesArray[idx] = coordinatesArray[auxIds[startArgmin]]
                    refArgmin = startArgmin
                else:
                    orderedCoordinatesArray[idx] = np.flip(coordinatesArray[auxIds[endArgmin]], axis=0) # Flipping needed
                    refArgmin = endArgmin
                # Delete already found segments
                startPoints = np.delete(startPoints, refArgmin, axis = 0)
                endPoints = np.delete(endPoints, refArgmin, axis = 0)
                auxIds = np.delete(auxIds, refArgmin)
                # Store order of coordinatesArray
                segmentsOrder[idx] = refArgmin
                
                reference = orderedCoordinatesArray[idx][-1]
                
            singleSegment = np.ndarray([0, 3])

            for idx in range(len(coordinatesArray)):
                singleSegment = np.concatenate((singleSegment, orderedCoordinatesArray[idx]), axis = 0)
            
            return singleSegment, orderedCoordinatesArray

        # Get AA points as reference to determine vessel origin
        AAsegmentIds = []
        numberOfPoints = 0
        for idx in self.cellIdToVesselType: 
            if vesselName == "BT": # If BT, we consider only type 1 as AA
                if self.cellIdToVesselType[idx] == 1:
                    AAsegmentIds.append(idx)
                    numberOfPoints += len(self.segmentsArray[idx][0])
            else:                  # Otherwise, we also consider type 14
                if self.cellIdToVesselType[idx] in [1, 14]:
                    AAsegmentIds.append(idx)
                    numberOfPoints += len(self.segmentsArray[idx][0])
        # Pool all AA coordinates in one array
        AACoordinatesArray = np.ndarray([numberOfPoints, 3], dtype=float)
        auxIdx = 0
        for idx, AAsegmentId in enumerate(AAsegmentIds):
            for idx2 in range(len(self.segmentsArray[AAsegmentId][0])):
                AACoordinatesArray[idx2 + auxIdx] = np.matmul(self.aff, np.append(self.segmentsArray[AAsegmentId][0][idx2], 1.0))[:3]
            auxIdx += idx2
        coordinatesArray = np.ndarray([len(segmentIds)], dtype=object)
        for idx, segmentId in enumerate(segmentIds):
            coordinatesArray[idx] = np.ndarray([len(self.segmentsArray[segmentId][0]), 3])
            for idx2 in range(len(self.segmentsArray[segmentId][0])):
                coordinatesArray[idx][idx2] = np.matmul(self.aff, np.append(self.segmentsArray[segmentId][0][idx2], 1.0))[:3]
        # Order segments and return a single ordered array with all segments' coordinates, and segments order from coordinatesArray
        singleSegment, orderedCoordinatesArray = orderCoordinatesArray(coordinatesArray, AACoordinatesArray)

        return singleSegment, orderedCoordinatesArray

    def getClosestAAPoint(self, point):
        # For AA, just pool all segments with type 1 or 14
        AAsegmentIds = []
        numberOfPoints = 0
        for idx in self.cellIdToVesselType: 
            if self.cellIdToVesselType[idx] in [1]:
                AAsegmentIds.append(idx)
                numberOfPoints += len(self.segmentsArray[idx][0])
        # If AA not found, output nan
        if len(AAsegmentIds) == 0:
            return math.nan
        # Otherwise, find point with highest S coordinate (vertical/axial axis) and record associates dimateter   
        else:
            # Pool all AA coordinates into one array (regardless of order)
            AACoordinatesArray = np.ndarray([numberOfPoints, 3], dtype=float)
            auxIdx = 0
            for idx, AAsegmentId in enumerate(AAsegmentIds):
                for idx2 in range(len(self.segmentsArray[AAsegmentId][0])):
                    AACoordinatesArray[idx2 + auxIdx] = np.matmul(self.aff, np.append(self.segmentsArray[AAsegmentId][0][idx2], 1.0))[:3]
                auxIdx += idx2
        
        return AACoordinatesArray[np.argmin(np.linalg.norm(AACoordinatesArray - point, axis = 1))]

def findPointId(point, modelCoordinates):
    return np.argmin(np.linalg.norm(modelCoordinates - point, axis = 1))

def absSphericalAnglesFrom3DCartesian(vec):
    ''' Returns spherical angles of a vector in cartesian coordinates
    with rescpect to a reference. If not specified, ref = [1, 0, 0].

    Arguments:
        vec: numpy array of shape [3] or equivalent.

    Returns:
        polar: polar angle in spherical coordinates.
        azimuth: azimuthal angle in spherical coordinates.

    '''
    x, y, z = vec
    
    if x == 0:
        return math.nan, math.nan
    else:
        polar = math.atan((x ** 2 + y ** 2) ** 0.5 / z)
        azimuth = math.atan(y / x)

        if polar < 0:
            polar += math.pi

        return polar, azimuth