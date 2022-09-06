import os
import json
import shutil

import numpy as np
import networkx as nx

import nibabel as nib

import vtk

from .centerlineGraphUtils import *
from .vesselLabelling.predictGraph import predictGraph

import warnings
warnings.filterwarnings('ignore')

class centerlineGraphOperator:
    def __init__(self, caseDir):
        # Case directory
        self.caseDir = caseDir
        # Get caseId    
        self.caseId = os.path.basename(self.caseDir)

        # Initialize graph with networkx
        self.centerlineGraph = nx.Graph()
        # The rightmost node can be used for hierarchical indexing from radial access
        self.rightmostNode = None
        # Define subgraph list
        self.subgraphs = []

        # Get segmentsArray
        self.segmentsArray = np.load(os.path.join(caseDir, "segmentsArray.npy"), allow_pickle = True)
        # Get affine matrix from nifti
        self.aff = nib.load(os.path.join(self.caseDir, f"{self.caseId}.nii.gz")).affine
        # Change sign of first component (change transformation from LAS to RAS)
        self.aff[0, 0] = - self.aff[0, 0]
        # Set translation from affine matrix to 0
        self.aff[:3, 3] = 0
        # Get segmentsArray in RAS coordinates (without considering translation)
        self.segmentsArrayRAS = np.ndarray([len(self.segmentsArray)], dtype = object)
        for idx in range(len(self.segmentsArray)):
            self.segmentsArrayRAS[idx] = np.ndarray([len(self.segmentsArray[idx][0]), 3])
            for idx2 in range(len(self.segmentsArray[idx][0])):
                self.segmentsArrayRAS[idx][idx2] = np.matmul(self.aff, np.append(self.segmentsArray[idx][0][idx2], 1.0))[:3]
        # Get coordinates array from segmentsArray
        self.segmentsCoordinateArray = self.segmentsArrayRAS
        a = np.ndarray([0, 3])
        for segment in self.segmentsCoordinateArray:
            a = np.append(a, segment, axis = 0)
        # Get radius array from segmentsArray
        self.segmentsRadiusArray = self.segmentsArray[:, 1]

        # Initialize simplified graph with networkx
        self.simpleCenterlineGraph = nx.Graph()
        
        # Set reference scale for node sampling
        self.sampleNodeEvery = 5

        # Get CTA data from nifti
        self.niftiCTA = nib.load(os.path.join(self.caseDir, f"{os.path.basename(self.caseDir)}.nii.gz"))

        # Load branchModel
        vtkPolyDataReader = vtk.vtkPolyDataReader()
        vtkPolyDataReader.SetFileName(os.path.join(self.caseDir, "branchModel.vtk"))
        vtkPolyDataReader.Update()
        self.branchModel = vtkPolyDataReader.GetOutput()

        # Define accesses
        self.accesses = ["femoral", "radial"]

        # Define cellId to vesselType dict
        self.cellIdToVesselType = {}
        # Define cellId to vesselTypeName dict
        self.cellIdToVesselTypeName = {}
        # Define subgraphUnionEdges
        self.subgraphsUnionEdges = []

        # Build standard reference configurations
        self.configurations = {}
        # Configurations from femoral access
        self.configurations["femoral"] = []
        # Configuration 0: femoral + right + anterior
        priorityOrder = ["AA", "BT", "RCCA", "RICA"]
        highlights = ["RCCA", "RICA"]
        counterHighlights = ["RSA"]
        self.configurations["femoral"].append([priorityOrder, highlights, counterHighlights, "0_FemoralRightAnterior"])
        # Configuration 1: femoral + right + posterior
        priorityOrder = ["AA", "BT", "RSA", "RVA", "BA"]
        highlights = ["RVA"]
        counterHighlights = ["LVA"]
        self.configurations["femoral"].append([priorityOrder, highlights, counterHighlights, "1_FemoralRightPosterior"])
        # Configuration 2: femoral + left + anterior
        priorityOrder = ["AA", "LCCA", "LICA"]
        highlights = ["LCCA", "LICA"]
        counterHighlights = []
        self.configurations["femoral"].append([priorityOrder, highlights, counterHighlights, "2_FemoralLeftAnterior"])
        # Configuration 3: femoral + left + posterior
        priorityOrder = ["AA", "LSA", "LVA", "BA"]
        highlights = ["LVA"]
        counterHighlights = ["RVA"]
        self.configurations["femoral"].append([priorityOrder, highlights, counterHighlights, "3_FemoralLeftPosterior"])
        # Configurations from radial access
        self.configurations["radial"] = []
        # Configuration 4: radial + right + anterior
        priorityOrder = ["RSA", "RCCA", "RICA"]
        highlights = ["RCCA", "RICA"]
        counterHighlights = ["BT"]
        self.configurations["radial"].append([priorityOrder, highlights, counterHighlights, "4_RadialRightAnterior"])
        # Configuration 5: radial + right + posterior
        priorityOrder = ["RSA", "RVA", "BA"]
        highlights = ["RVA"]
        counterHighlights = ["LVA"]
        self.configurations["radial"].append([priorityOrder, highlights, counterHighlights, "5_RadialRightPosterior"])
        # Configuration 6: radial + left + anterior
        priorityOrder = ["RSA", "BT", "LCCA", "LICA"]
        highlights = ["LCCA", "LICA"]
        counterHighlights = []
        self.configurations["radial"].append([priorityOrder, highlights, counterHighlights, "6_RadialLeftAnterior"])
        # Configuration 7: radial + left + posterior
        priorityOrder = ["RSA", "BT", "LSA", "LVA", "BA"]
        highlights = ["LVA"]
        counterHighlights = ["RVA"]
        self.configurations["radial"].append([priorityOrder, highlights, counterHighlights, "7_RadialLeftPosterior"])

        # We also have to pass the reference configurations to one-hot encoding
        self.configurationsOneHot = {}
        for access in self.accesses:
            self.configurationsOneHot[access] = []
            for configuration in self.configurations[access]:
                configSequence, highlights, counterHighlights, _ = configuration
                self.configurationsOneHot[access].append(vesselTypeSequenceToOneHot(configSequence, highlights, counterHighlights))

        # Define predicted configurations (cellIds sequences)
        self.predictedConfigurations = {}
        # Define supersegments dict
        self.supersegments = {}
        # Define simpleSupersegments dict
        self.simpleSupersegments = {}

    def makeSimpleCenterlineGraph(self):
        ''' Elaborate simplified graph (each segmnet from the segmentsArray corresponds to each graph edge).

        Creates self.simpleCenterlineGraph.

        Stores simpleGraph.pickle and simpleGraph.png.
        
        '''
        # Initialize simplified graph with networkx
        self.simpleCenterlineGraph = nx.Graph()

        # Only taking first and last positions of the curves arrays
        totalNodes = 0 # We only link nodes from the same centerline
        for cellId in range(len(self.segmentsCoordinateArray)):
            if len(self.segmentsCoordinateArray[cellId]) > 1 and len(self.segmentsRadiusArray[cellId]) > 1:
                # Add nodes
                # First node of cell (startpoint)
                self.simpleCenterlineGraph.add_node(totalNodes + 0, pos=self.segmentsCoordinateArray[cellId][0])
                self.simpleCenterlineGraph.nodes[totalNodes + 0]["Radius"] = self.segmentsRadiusArray[cellId][0]
                # Last node of cell (endpoint)
                self.simpleCenterlineGraph.add_node(totalNodes + 1, pos=self.segmentsCoordinateArray[cellId][-1])
                self.simpleCenterlineGraph.nodes[totalNodes + 1]["Radius"] = self.segmentsRadiusArray[cellId][-1]

                # Add edges
                self.simpleCenterlineGraph.add_edge(totalNodes, totalNodes + 1, cellId = cellId)
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["segmentsCoordinateArray"] = self.segmentsCoordinateArray[cellId]
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["segmentsRadiusArray"] = self.segmentsRadiusArray[cellId]

                totalNodes += 2

        # Merge nodes that share the same RAS coordinate (bifurcation spots)
        # First get all nodes that have a degree of 1 (start- and endpoints)
        deg1Nodes = []
        for node, deg in self.simpleCenterlineGraph.degree:
            if deg == 1:
                deg1Nodes.append(node)
                
        # For all degree 1 nodes, we check position to join corresponding start- and enpoints, as well as bifurcations
        removedNodes = []
        for _, node in enumerate(deg1Nodes):
            if node not in removedNodes:
                aux = deg1Nodes.copy()
                aux.remove(node)
                for auxNodes in removedNodes:
                    aux.remove(auxNodes)
                for _, node2 in enumerate(aux):
                    C1 = self.simpleCenterlineGraph.nodes[node]["pos"]
                    C2 = self.simpleCenterlineGraph.nodes[node2]["pos"]
                    if C1[0] == C2[0] and C1[1] == C2[1] and C1[2] == C2[2]:
                        self.simpleCenterlineGraph = nx.contracted_nodes(self.simpleCenterlineGraph, node, node2)
                        removedNodes.append(node2)
                        self.simpleCenterlineGraph.nodes[node].pop("contraction")

        # Relabel nodes as sequential labels
        mapping = {}
        newNode = 0
        for oldNode in self.simpleCenterlineGraph.nodes():
            mapping[oldNode] = newNode
            newNode += 1
        self.simpleCenterlineGraph = nx.relabel.relabel_nodes(self.simpleCenterlineGraph, mapping)

        # Featurizes self.simpleCenterlineGraph
        if len(self.cellIdToVesselType) == 0:
            self.simpleCenterlineGraph = extractSegmentLevelFeatures(self, forLabelling = True)
            # Save simplified graph
            nx.readwrite.gpickle.write_gpickle(self.simpleCenterlineGraph, os.path.join(self.caseDir, "simpleGraph.pickle"), protocol = 4)
            # Make quick plot for easy visualization
            makeGraphPlot(self.caseDir, self.simpleCenterlineGraph, "simpleGraph.png", label = "cellId")
        else:
            self.getSingleSegments()

            self.simpleCenterlineGraph = extractSegmentLevelFeatures(self, forLabelling = False)
            # Save simplified graph
            nx.readwrite.gpickle.write_gpickle(self.simpleCenterlineGraph, os.path.join(self.caseDir, "unifiedSmpleGraph.pickle"), protocol = 4)
            # Make quick plot for easy visualization
            makeGraphPlot(self.caseDir, self.simpleCenterlineGraph, "unifiedsSmpleGraph.png", label = "cellId")

    def predictVesselTypes(self):
        ''' Performs graph U-Net inference with simplified graph and creates self.cellIdToVesselTypeName.

        Creates self.predictedSimpleCenterlineGraph and self.cellIdToVesselTypeName.
        
        '''
        # Perform vesselType prediction
        self.predictedSimpleCenterlineGraph = predictGraph(self.caseDir)
        # self.predictedSimpleCenterlineGraph = nx.readwrite.gpickle.read_gpickle(os.path.join(self.caseDir, "graph_pred.pickle"))
        # If manually labelled are to be used for labelling of dense graphs
        # self.predictedSimpleCenterlineGraph = nx.readwrite.gpickle.read_gpickle(os.path.join(self.caseDir, "graph_label.pickle"))
        # Get cellId to vesselType dict from predicted graph
        for n0, n1 in self.predictedSimpleCenterlineGraph.edges:
            self.cellIdToVesselType[self.predictedSimpleCenterlineGraph[n0][n1]["cellId"]] = self.predictedSimpleCenterlineGraph[n0][n1]["Vessel type"]
            self.cellIdToVesselTypeName[self.predictedSimpleCenterlineGraph[n0][n1]["cellId"]] = self.predictedSimpleCenterlineGraph[n0][n1]["Vessel type name"]

        # for node in self.centerlineGraph:
        #     self.centerlineGraph.nodes[node]["Vessel type"] = self.cellIdToVesselType[self.centerlineGraph.nodes[node]["cellId"]]
        #     self.centerlineGraph.nodes[node]["Vessel type name"] = self.cellIdToVesselTypeName[self.centerlineGraph.nodes[node]["cellId"]]

        # for src, dst in self.centerlineGraph.edges:
        #     self.centerlineGraph[src][dst]["Vessel type"] = self.cellIdToVesselType[self.centerlineGraph[src][dst]["cellId"]]
        #     self.centerlineGraph[src][dst]["Vessel type name"] = self.cellIdToVesselTypeName[self.centerlineGraph[src][dst]["cellId"]]

        # Save graph with predicted edge types
        nx.readwrite.gpickle.write_gpickle(self.centerlineGraph, os.path.join(self.caseDir, "centerlineGraph.pickle"), protocol = 4)

    def sanityCheckForRandomIslands(self):
        # Search for potentially randomly segmented islands depending on the distance between centers of mass 
        # Compute the center of mass of each of the subgraphs, as well as the overall center of mass
        globalCenterOfMass = np.ndarray([0, 3])
        centersOfMass = []
        for subgraph in self.subgraphs:
            centerOfMass = np.ndarray([0, 3])
            for node in subgraph:
                centerOfMass = np.append(centerOfMass, [subgraph.nodes[node]["pos"]], axis = 0)
                globalCenterOfMass = np.append(globalCenterOfMass, [subgraph.nodes[node]["pos"]], axis = 0)
            centerOfMass = np.mean(centerOfMass, axis = 0)
            # Store the center of mass of each subgraph
            centersOfMass.append(centerOfMass)
        # Compute global center of mass
        globalCenterOfMass = np.mean(globalCenterOfMass, axis = 0)
        # Compute distance from all subgraph's center of mass to the global center of mass
        distances = np.linalg.norm(np.array(centersOfMass) - np.array([globalCenterOfMass]), axis = 1)
        
        deleteSubgraphs = []
        deleteCellIds = []
        for idx, distance in enumerate(distances):
            if distance > 2.5 * np.mean(distances):
                deleteSubgraphs.append(idx)
                for node in self.subgraphs[idx]:
                    if self.subgraphs[idx].nodes[node]["cellId"] not in deleteCellIds:
                        deleteCellIds.append(self.subgraphs[idx].nodes[node]["cellId"])

        if len(deleteCellIds) > 0:
            print("     Found random islands containing the following cellId:", deleteCellIds)
            # Delete subgraphs from random islands
            self.subgraphs = list(np.delete(self.subgraphs, deleteSubgraphs))
            # Correcting position shift due to presence of random island
            newCenterOfMass = np.mean(np.delete(centersOfMass, deleteSubgraphs), axis = 0)
            # Compute translation correction
            correctionRAS = globalCenterOfMass - newCenterOfMass
            # Pass to IJK coordinates
            correction = np.matmul(np.linalg.inv(self.aff), np.append(correctionRAS, 1.0))[:3]
            # Apply correction to segmentsArray
            self.segmentsArray = np.delete(self.segmentsArray, deleteCellIds, axis = 0)
            for idx in range(len(self.segmentsArray)):
                for idx2 in range(len(self.segmentsArray[idx][0])):
                    self.segmentsArray[idx][0][idx2] = self.segmentsArray[idx][0][idx2] - 1.25 * correction
            # Saving new segmentsArray
            np.save(os.path.join(self.caseDir, "segmentsArray.npy"), self.segmentsArray)
            # Reinitializing class and generating new dense graph
            self.__init__(self.caseDir) 
            self.makeCenterlineGraph()
        else:
            pass

    def makeCenterlineGraph(self):
        ''' Elaborate dense graph from segmentsArray (dense node sampling).

        Creates self.centerlineGraph, self.rightmostNode and self.subgraphs.

        Stores denseGraph.pickle and denseGraph.png.
        
        '''
        # Renitialize graph with networkx
        self.centerlineGraph = nx.Graph()
        # Building dense graph
        totalNodes = 0 
        # We only link nodes from the same centerline first, and afterwards we contract nodes with the same position
        for cellId, curve in enumerate(self.segmentsCoordinateArray):
            # Initialize distance for node sampling
            distance = 0
            # We keep track of last node 
            previousIdx = 0
            # Now we loop to every centerline point in each segmentsArray cell
            for idx, position in enumerate(curve):
                # For the first node in every cell, we add just a node with its corresponding cellId
                if idx == 0:
                    self.centerlineGraph.add_node(totalNodes, pos=position)
                    self.centerlineGraph.nodes[totalNodes]["radius"] = self.segmentsRadiusArray[cellId][idx]
                    self.centerlineGraph.nodes[totalNodes]["cellId"] = cellId
                    totalNodes += 1
                    previousPosition = position
                # If distance from last sampled node is larger than selected sampleNodeEvery, add a node with cellId and an edge to the previous sample 
                # node, keeping cellId and the indices of the centerline points in the segmentsArray
                elif distance > self.sampleNodeEvery:
                    self.centerlineGraph.add_node(totalNodes, pos=position)
                    self.centerlineGraph.nodes[totalNodes]["radius"] = self.segmentsRadiusArray[cellId][idx]
                    self.centerlineGraph.nodes[totalNodes]["cellId"] = cellId
                    self.centerlineGraph.add_edge(totalNodes - 1, totalNodes)
                    self.centerlineGraph[totalNodes - 1][totalNodes]["cellId"] = cellId
                    self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayIndices"] = np.arange(previousIdx, idx)
                    self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsCoordinateArray"] = self.segmentsCoordinateArray[cellId][self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayIndices"]]
                    self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsRadiusArray"] = self.segmentsRadiusArray[cellId][self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayIndices"]]
                    totalNodes += 1
                    previousPosition = position
                    # We alse reinitialize the accumulated distance and update the prior node previousIdx
                    distance = 0
                    previousIdx = idx
                # For the last point of every centerline cell (presumably at a distance smaller than sampleNodeEvery) we differentiate between two possible cases
                elif idx == len(curve) - 1:
                    # If the number of nodes of the previous node of the cell is 0 (which means that the segment's length is smaller than sampleNodeEvery), we add an additional node
                    if len([n for n in self.centerlineGraph.neighbors(totalNodes - 1)]) == 0:
                        self.centerlineGraph.add_node(totalNodes, pos=position)
                        self.centerlineGraph.nodes[totalNodes]["radius"] = self.segmentsRadiusArray[cellId][idx]
                        self.centerlineGraph.nodes[totalNodes]["cellId"] = cellId
                        self.centerlineGraph.add_edge(totalNodes - 1, totalNodes)
                        self.centerlineGraph[totalNodes - 1][totalNodes]["cellId"] = cellId
                        self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayIndices"] = np.arange(previousIdx, idx)
                        self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsCoordinateArray"] = self.segmentsCoordinateArray[cellId][self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayIndices"]]
                        self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsRadiusArray"] = self.segmentsRadiusArray[cellId][self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayIndices"]]
                        totalNodes += 1
                        previousPosition = position
                    # If it is not, which will be the general case, we do not add a new node, but instead we transform the last added node and change its position to be placed at the bifurcation/endpoint
                    else:
                        self.centerlineGraph.nodes[totalNodes - 1]["pos"]= position
                        self.centerlineGraph.nodes[totalNodes - 1]["radius"] = self.segmentsRadiusArray[cellId][idx]
                        self.centerlineGraph.nodes[totalNodes - 1]["cellId"] = cellId
                        self.centerlineGraph[totalNodes - 2][totalNodes - 1]["cellId"] = cellId
                        self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsArrayIndices"] = np.append(self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsArrayIndices"], np.arange(previousIdx, idx))
                        self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsCoordinateArray"] = self.segmentsCoordinateArray[cellId][self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsArrayIndices"]]
                        self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsRadiusArray"] = self.segmentsRadiusArray[cellId][self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsArrayIndices"]]
                        previousPosition = position
                # If the accumulated distance from the last sample node is smaller than sampleNodeEvery, and we are not in either the first or last nodes of the segmentsArray cell, just update the distance
                else:
                    distance += np.linalg.norm(previousPosition - position)
                    previousPosition = position
                    
        # Merge nodes that share the same coordinate (bifurcation spots)
        # First get all nodes that have a degree of 1 (start- and endpoints)
        deg1Nodes = []
        for node, deg in self.centerlineGraph.degree:
            if deg == 1:
                deg1Nodes.append(node)

        # For all degree 1 nodes, we check position to join corresponding startpoints and endpoints
        contractedNodes = []
        for _, node in enumerate(deg1Nodes):
            if node not in contractedNodes:
                aux = deg1Nodes.copy()
                aux.remove(node)
                for auxNodes in contractedNodes:
                    aux.remove(auxNodes)
                for _, node2 in enumerate(aux):
                    C1 = self.centerlineGraph.nodes[node]["pos"]
                    C2 = self.centerlineGraph.nodes[node2]["pos"]
                    if C1[0] == C2[0] and C1[1] == C2[1] and C1[2] == C2[2]:
                        self.centerlineGraph = nx.contracted_nodes(self.centerlineGraph, node, node2)
                        contractedNodes.append(node2)
                        self.centerlineGraph.nodes[node].pop("contraction")           
                        

        # If any nodes with degree == 0 are present, remove them
        removeNodes = []
        for node in self.centerlineGraph:
            if self.centerlineGraph.degree(node) == 0:
                removeNodes.append(node)
        for node in removeNodes:
            self.centerlineGraph.remove_node(node)

        # Relabel nodes as sequential labels
        mapping = {}
        newNode = 0
        for oldNode in self.centerlineGraph.nodes():
            mapping[oldNode] = newNode
            newNode += 1
        self.centerlineGraph = nx.relabel.relabel_nodes(self.centerlineGraph, mapping)

        # The self.rightmostNode node can be used for hierarchical indexing from radial access
        rightmostNodePosition = self.segmentsCoordinateArray[0][0]
        for node in self.centerlineGraph:
            if self.centerlineGraph.nodes[node]["pos"][0] < rightmostNodePosition[0] and self.centerlineGraph.degree(node) == 1:
                self.rightmostNode = node
                rightmostNodePosition = self.centerlineGraph.nodes[node]["pos"]

        for node in self.centerlineGraph:
            self.centerlineGraph.nodes[node]["Vessel type"] = self.cellIdToVesselType[self.centerlineGraph.nodes[node]["cellId"]]
            self.centerlineGraph.nodes[node]["Vessel type name"] = self.cellIdToVesselTypeName[self.centerlineGraph.nodes[node]["cellId"]]

        for src, dst in self.centerlineGraph.edges:
            self.centerlineGraph[src][dst]["Vessel type"] = self.cellIdToVesselType[self.centerlineGraph[src][dst]["cellId"]]
            self.centerlineGraph[src][dst]["Vessel type name"] = self.cellIdToVesselTypeName[self.centerlineGraph[src][dst]["cellId"]]

        # Divide the dense graph into disconnected subgraphs
        self.subgraphs = [self.centerlineGraph.subgraph(components) for components in nx.connected_components(self.centerlineGraph)]

        # Remove short (less than 5 nodes) subgraphs (do not introduce much information and can easily corrupt feature extraction)
        if len(self.subgraphs) > 1:
            deleteSubgraphs = []
            for idx, subgraph in enumerate(self.subgraphs):
                if len(subgraph) < 5:
                    print(f"     Removing subgraph {idx} with length", len(subgraph))
                    deleteSubgraphs.append(idx)
                    # self.subgraphs.remove(subgraph)

            self.subgraphs = list(np.delete(self.subgraphs, deleteSubgraphs))

        # Loop over the subgraphs and get hierarchical indexing for each (separately) and extract features
        # We have to divide the graphs into subgraphs for the hierarchical indexing to be applied properly,
        # so that feature extraction can be easlily performed
        for idx, subgraph in enumerate(self.subgraphs):
            startNode = [node for node in subgraph][0]
            # # For the startNode of disconnected subgraphs, get the node with the lowest S coordinate for hierarchical ordering (it is only an approximation for vessel labelling, not very relevant)
            if idx > 0:
                minS = subgraph.nodes[startNode]["pos"][2]
                for node in self.subgraphs[idx]:
                    if subgraph.nodes[node]["pos"][2] < minS and subgraph.degree(node) == 1:
                        minS = subgraph.nodes[node]["pos"][2]
                        startNode = node
            subgraph = getHierarchicalOrderingDense(subgraph, "femoral", startNode)
            self.subgraphs[idx] = subgraph

        # Now join all subgraphs for labelling
        self.centerlineGraph = None
        self.centerlineGraph = self.subgraphs[0].copy()
        for subgraph in self.subgraphs[1:]:
            self.centerlineGraph = nx.compose(self.centerlineGraph, subgraph)

        # Save resulting graph
        nx.readwrite.gpickle.write_gpickle(self.centerlineGraph, os.path.join(self.caseDir, "centerlineGraph.pickle"), protocol = 4)
        # Generate plot of dense graph for quick visualization
        makeGraphPlot(self.caseDir, self.centerlineGraph, "centerlineGraph.png")

        self.sanityCheckForRandomIslands()

    def unifySubgraphs(self):
        ''' The goal of this method is make one large connected graph resulting from the union between all subgraphs.
        To do that, the candidate points for graph union are identified and then a union node is searched in the main graph.
        
        Candidate points are taken as nodes with degree 1 from secondary subgraphs, that are more proximal than the alternative
        extremal point of the segment. VA points from the main subgraph are also considered. 

        Then, once candidate points for union form secondary subgraphs are gathered, candidate points for artificial edge connections to
        the main subgraph (or larger subgraphs) are searched. A preferential sequence is then followed. First, a candidate from a segment 
        with the same vessel type from all larger subgraphs (excluding the same one, except for the VA candidate search of the main subgraph)
        is searched. If this is not available, then a preferred vessel type for each vessel type is defined. This preferred vessel type is then 
        searched in all smaller subgraphs (same criteria than before). If this is not found either, then the candidate nodes are joint to the
        closest node from the main subgraph. 

        Creates self.subgraphsUnionEdges.

        Stores unifiedGraph.pickle and unifiedGraph.png.

        '''
        # Preferred vessel types for subgraph union
        preferredVesselTypes = {"AA": None,
                                "BT": "AA",
                                "RCCA": "BT",
                                "RSA": "BT",
                                "RVA": "RSA",
                                "RICA": "RCCA",
                                "RECA": "RCCA",
                                "LCCA": "AA",
                                "LSA": "AA",
                                "LVA": "LSA", 
                                "LICA": "LCCA",
                                "LECA": "LCCA", 
                                "BA": "RVA",
                                "other": None}

        # Compute the center of mass of each of the subgraphs
        if len(self.subgraphs) > 1:
            centersOfMassS = []
            for subgraph in self.subgraphs[1:]:
                centerOfMass = np.ndarray([0, 3])
                for node in subgraph:
                    centerOfMass = np.append(centerOfMass, [subgraph.nodes[node]["pos"]], axis = 0)
                centerOfMass = np.mean(centerOfMass, axis = 0)
                # Store the S coordinates of each subgraph's center of mass
                centersOfMassS.append(centerOfMass[2])
            # Reorder subgraphs according to the S coordinate of their center of mass in ascending order. Maintain subgraph at 0 position
            self.subgraphs = list(np.array(self.subgraphs)[np.insert(np.argsort(centersOfMassS) + 1, 0, 0)])

        # We can get the cellId for all segments that form the subgraph 
        subgraphsCellIds = []
        for subgraph in self.subgraphs:
            subgraphCellIds = []
            for node in subgraph:
                if subgraph.nodes[node]["cellId"] not in subgraphCellIds:
                    subgraphCellIds.append(subgraph.nodes[node]["cellId"])      
            subgraphsCellIds.append(subgraphCellIds)
       
        # Pool all AA points for proximal/distal orientation
        AACoordinatesArray = np.ndarray([0, 3])
        for cellId in self.cellIdToVesselTypeName.keys():
            if cellId in subgraphsCellIds[0] and self.cellIdToVesselTypeName[cellId] == "AA":
                for point in self.segmentsCoordinateArray[cellId]:
                    AACoordinatesArray = np.append(AACoordinatesArray, [point], axis = 0)

        # Now, we can use the segmentsArray to get the extremal points for all cells
        candidatesForSubgraphsUnion = []
        oppositeNodesForSubgraphsUnion = []
        for idx, subgraphCellIds in enumerate(subgraphsCellIds):
            candidatesForSubgraphUnion = []
            oppositeNodesForSubgraphUnion = []
            for cellId in subgraphCellIds:
                # For all other subgraphs, we search for all vesselTypes
                if self.cellIdToVesselTypeName[cellId] not in ["AA", "other"]:
                    # Gather both extremal nodes from each segment
                    extremalNode0 = None
                    extremalNode1 = None
                    for node in self.subgraphs[idx]:
                        if (self.subgraphs[idx].nodes[node]["pos"] == self.segmentsCoordinateArray[cellId][0]).all() and extremalNode0 not in candidatesForSubgraphUnion: 
                            extremalNode0 = node
                        if (self.subgraphs[idx].nodes[node]["pos"] == self.segmentsCoordinateArray[cellId][-1]).all() and extremalNode1 not in candidatesForSubgraphUnion: 
                            extremalNode1 = node
                    # 0 is more proximal than 1 and 0 has degree 1, store node
                    if np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[extremalNode0]["pos"], axis = 1)) < np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[extremalNode1]["pos"], axis = 1)):
                        if self.subgraphs[idx].degree(extremalNode0) == 1:
                            candidatesForSubgraphUnion.append(extremalNode0)
                            oppositeNodesForSubgraphUnion.append(extremalNode1)
                    # 1 is more proximal than 0 and 1 has degree 1, store node
                    elif np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[extremalNode0]["pos"], axis = 1)) >= np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[extremalNode1]["pos"], axis = 1)):
                        if self.subgraphs[idx].degree(extremalNode1) == 1:
                            candidatesForSubgraphUnion.append(extremalNode1)
                            oppositeNodesForSubgraphUnion.append(extremalNode0)
                # If two nodes are found close by (within 20 mm) and they share the same vesselType, only proximalest (to AA) will be chosen (rarely happens)
                deleteCloseNodes = []
                for idxNodeA, nodeA in enumerate(candidatesForSubgraphUnion):
                    for idxNodeB, nodeB in enumerate(candidatesForSubgraphUnion[:idxNodeA]):
                        if nodeA != nodeB:
                            if np.linalg.norm(self.subgraphs[idx].nodes[nodeA]["pos"] - self.subgraphs[idx].nodes[nodeB]["pos"]) < 20 and self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[nodeA]["cellId"]] == self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[nodeB]["cellId"]]:
                                # We delete the more distal of the two
                                if np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[nodeA]["pos"], axis = 1)) < np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[nodeB]["pos"], axis = 1)):
                                    deleteCloseNodes.append(idxNodeB)
                                elif np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[nodeA]["pos"], axis = 1)) >= np.amin(np.linalg.norm(AACoordinatesArray - self.subgraphs[idx].nodes[nodeB]["pos"], axis = 1)):
                                    deleteCloseNodes.append(idxNodeA)
                candidatesForSubgraphUnion = list(np.delete(candidatesForSubgraphUnion, deleteCloseNodes))
                oppositeNodesForSubgraphUnion = list(np.delete(oppositeNodesForSubgraphUnion, deleteCloseNodes))
            # Append each candidate union node separately depending on the subgraph
            candidatesForSubgraphsUnion.append(candidatesForSubgraphUnion)
            oppositeNodesForSubgraphsUnion.append(oppositeNodesForSubgraphUnion)

        # Auxiliar list to store already joint subgraphs
        composedSubgraphs = []

        # Search for alternate union points in main subgraph or larger subgraphs following a preference system
        for idx, candidatesForSubgraphUnion in enumerate(candidatesForSubgraphsUnion):
            for candidateIdx, candidateNode in enumerate(candidatesForSubgraphUnion):
                # Pool all node and positions from the main graph with the same vesselType as the candidate node
                candidateMainGraphCoordinates = np.ndarray([0, 3])
                candidateMainGraphNodes = []
                # Boolean variable to end search for union node
                foundUnion = False
                # Auxiliar variable to store subgraph index to perform composition
                candidateSubgraphIdx = None
                if self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]] != "other":
                    for subgraphIdx in range(max([idx, 1])):
                        if self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]] in [self.cellIdToVesselTypeName[cellId] for cellId in subgraphsCellIds[subgraphIdx]] and not foundUnion:
                        # Check if the same vessel type exists in the larger subgraphs
                            # If it exists and has a different cellId, store all node coordinates
                            for nodeSubgraphIdx in self.subgraphs[subgraphIdx]:
                                if self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]] == self.cellIdToVesselTypeName[self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["cellId"]] and self.subgraphs[idx].nodes[candidateNode]["cellId"] != self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["cellId"]:
                                    # In the rare event that the connection node is found to be the self.rightmostNode, then choose its neighbor
                                    if nodeSubgraphIdx == self.rightmostNode:
                                        nodeSubgraphIdx = self.subgraphs[subgraphIdx].neighbors(self.rightmostNode).__next__()
                                    # In the case that we are looking at the same subgraph as the candidate node, forbid union with segments in contact
                                    if subgraphIdx == idx:
                                        cellIdsInContact = []
                                        for neighbor in self.subgraphs[subgraphIdx].neighbors(oppositeNodesForSubgraphsUnion[idx][candidateIdx]):
                                            cellIdsInContact.append(self.subgraphs[subgraphIdx].nodes[neighbor]["cellId"])
                                        if self.subgraphs[idx].nodes[nodeSubgraphIdx]["cellId"] in cellIdsInContact:
                                            pass
                                        else:
                                            candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
                                            candidateMainGraphNodes.append(nodeSubgraphIdx)
                                            candidateSubgraphIdx = subgraphIdx
                                            foundUnion = True
                                    # This tries to forbid union in the scenario where the same vessel type is found in the main subgraph and a secondary subgraph 
                                    # and the one in the main subgraph is above the secondarty subgraph by a large margin (30 mm). This cound create weird union, 
                                    # so it is better not to have this union at all and search for an alternative union
                                    elif self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"][2] - self.subgraphs[idx].nodes[candidateNode]["pos"][2] > 30:
                                        pass
                                    # Otherwise, go on with analysis
                                    else:
                                        candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
                                        candidateMainGraphNodes.append(nodeSubgraphIdx)
                                        candidateSubgraphIdx = subgraphIdx
                                        foundUnion = True
                    # Check if the preferred vessel type exists in the larger subgraphs
                    # Look at all graphs larger than the one we are analyzing (except for main subgraph, where we only look at itself)
                    for subgraphIdx in range(max([idx, 1])):
                        if preferredVesselTypes[self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]]] in [self.cellIdToVesselTypeName[cellId] for cellId in subgraphsCellIds[subgraphIdx]] and not foundUnion:
                            # If it exists, store all node coordinates
                            for nodeSubgraphIdx in self.subgraphs[subgraphIdx]:
                                if preferredVesselTypes[self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]]] == self.cellIdToVesselTypeName[self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["cellId"]]:
                                    # In the rare event that the connection node is found to be the self.rightmostNode, then choose its neighbor
                                    if nodeSubgraphIdx == self.rightmostNode:
                                        nodeSubgraphIdx = self.subgraphs[subgraphIdx].neighbors(self.rightmostNode).__next__()
                                    # In the case that we are looking at the same subgraph as the candidate node, group cellIds in contact with candidate node. Forbid union with segments in contact
                                    if subgraphIdx == idx:
                                        cellIdsInContact = []
                                        for neighbor in self.subgraphs[subgraphIdx].neighbors(oppositeNodesForSubgraphsUnion[idx][candidateIdx]):
                                            cellIdsInContact.append(self.subgraphs[subgraphIdx].nodes[neighbor]["cellId"])
                                        if self.subgraphs[idx].nodes[nodeSubgraphIdx]["cellId"] in cellIdsInContact:
                                            pass
                                        else:
                                            candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
                                            candidateMainGraphNodes.append(nodeSubgraphIdx)
                                            candidateSubgraphIdx = subgraphIdx
                                            foundUnion = True
                                    # Otherwise, go on with analysis
                                    else:
                                        candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
                                        candidateMainGraphNodes.append(nodeSubgraphIdx)
                                        candidateSubgraphIdx = subgraphIdx
                                        foundUnion = True
                # If none of the above have worked, pool all node coordinates from main subgraph
                if not foundUnion:
                    for nodeMain in self.subgraphs[0]:
                        if self.cellIdToVesselTypeName[self.subgraphs[0].nodes[nodeMain]["cellId"]]:
                            candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[0].nodes[nodeMain]["pos"]], axis = 0)
                            candidateMainGraphNodes.append(nodeMain)
                            candidateSubgraphIdx = 0
                            foundUnion = True

                # Compose graphs into largest one. Update subgraphsCellIds for next candidate node union search
                if foundUnion:
                    # Choose closest node from the candidate pool
                    mainGraphNode = candidateMainGraphNodes[np.argmin(np.linalg.norm(candidateMainGraphCoordinates - self.subgraphs[idx].nodes[candidateNode]["pos"], axis = 1))]
                    # Append node pairs altogether
                    self.subgraphsUnionEdges.append([mainGraphNode, candidateNode])
                    # If the the union is found within the same subgraph, or both subgraphs have already been composed, just add the edge (or don't, this is just an intermediate result which is left unused)
                    if candidateSubgraphIdx == idx or [candidateSubgraphIdx, idx] in composedSubgraphs:
                        pass
                    # Otherwise, compose both graphs. This will be handy for union search of future candidate nodes
                    else:
                        # Compose subgraphs onto larger subgraph (candidateSubgraphIdx will be smaller than idx, by design)
                        self.subgraphs[candidateSubgraphIdx] = nx.compose(self.subgraphs[idx], self.subgraphs[candidateSubgraphIdx])
                        self.subgraphs[candidateSubgraphIdx].add_edge(mainGraphNode, candidateNode)
                        for cellId in subgraphsCellIds[idx]:
                            subgraphsCellIds[candidateSubgraphIdx].append(cellId)
                        # Add to already composed list
                        composedSubgraphs.append([candidateSubgraphIdx, idx])
                else:
                    # If no union is found, remove candidates and opposite nodes from lists
                    candidatesForSubgraphsUnion[idx].remove(candidateNode)
                    oppositeNodesForSubgraphsUnion[idx].remove(oppositeNodesForSubgraphsUnion[idx][candidateIdx])

        # Prepare nextCellId for segment splitting
        nextCellId = len(self.segmentsCoordinateArray)
        # In order to deliver a more complete supersegment visualization, and accurately deliver supersegments with their bifurcating segments, 
        # we will include artificial edges to the graphs, and we will divide segments (segmentsCoordinatesArray and segmentsRadiusArray) into new cellIds
        # We analyze each artificial union
        for mainGraphNode, candidateNode in self.subgraphsUnionEdges:
            candidateCellId = self.centerlineGraph.nodes[candidateNode]["cellId"]
            # We add the position and radius of the mainGraphNode to the candidateCellId segment to the first (or last) position of the segments arrays
            if np.linalg.norm(self.centerlineGraph.nodes[mainGraphNode]["pos"] - self.segmentsCoordinateArray[candidateCellId][0]) < np.linalg.norm(self.centerlineGraph.nodes[mainGraphNode]["pos"] - self.segmentsCoordinateArray[candidateCellId][-1]):
                self.segmentsCoordinateArray[candidateCellId] = np.insert(self.segmentsCoordinateArray[candidateCellId], 0, [self.centerlineGraph.nodes[mainGraphNode]["pos"]], axis = 0)
                self.segmentsRadiusArray[candidateCellId] = np.insert(self.segmentsRadiusArray[candidateCellId], 0, self.centerlineGraph.nodes[mainGraphNode]["radius"]) 
            else:
                self.segmentsCoordinateArray[candidateCellId] = np.append(self.segmentsCoordinateArray[candidateCellId], [self.centerlineGraph.nodes[mainGraphNode]["pos"]], axis = 0)
                self.segmentsRadiusArray[candidateCellId] = np.append(self.segmentsRadiusArray[candidateCellId], self.centerlineGraph.nodes[mainGraphNode]["radius"]) 
            
            # We add the edge to the graph
            self.centerlineGraph.add_edge(mainGraphNode, candidateNode, cellId = self.centerlineGraph.nodes[candidateNode]["cellId"])
            self.centerlineGraph[mainGraphNode][candidateNode]["Vessel type"] = self.cellIdToVesselType[self.centerlineGraph[mainGraphNode][candidateNode]["cellId"]]
            self.centerlineGraph[mainGraphNode][candidateNode]["Vessel type name"] = self.cellIdToVesselTypeName[self.centerlineGraph[mainGraphNode][candidateNode]["cellId"]]
            self.centerlineGraph[mainGraphNode][candidateNode]["segmentsArrayIndices"] = np.array([])
            self.centerlineGraph[mainGraphNode][candidateNode]["segmentsCoordinateArray"] = np.ndarray([0, 3])
            self.centerlineGraph[mainGraphNode][candidateNode]["segmentsRadiusArray"] = np.array([])

            
            # Now, for the modification of the segments arrays and the cellIds and segmentsArrayIndices of nodes and edges, we perform an indepth analysis.
            # First of all, it only makes sense to split the segment if the node has degree 2 (otherwise it will already be a border between different segments)
            if self.centerlineGraph.degree(mainGraphNode) == 3:
                # The position of the mainGraphNode will be the division point between segments
                cutOffIdx = np.argmin(np.linalg.norm(self.segmentsCoordinateArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]] - self.centerlineGraph.nodes[mainGraphNode]["pos"], axis = 1))
                # Check if segment goes downstream with respect to hierarchy. If it is, go against hierarchy. If it is not (normal case), go with hierarchy
                downstream = False
                for neighbor in self.centerlineGraph.neighbors(mainGraphNode):
                    # If neighbor with higher hierarchy has smaller segmentsArrayIndices indices than cutOffIdx, the segment is downstream. Otherwise it is not
                    if self.centerlineGraph.nodes[mainGraphNode]["cellId"] == self.centerlineGraph.nodes[neighbor]["cellId"] and self.centerlineGraph.nodes[neighbor]["hierarchy femoral"] > self.centerlineGraph.nodes[mainGraphNode]["hierarchy femoral"] and np.mean(self.centerlineGraph[neighbor][mainGraphNode]["segmentsArrayIndices"]) < cutOffIdx:
                        downstream = True
                # We keep mainGraphNode as initial previousNode for recursive node analysis
                previousNode = mainGraphNode
                # Boolean variable to stop the neighbor sweeping
                cellIdChangeCompleted = False
                while not cellIdChangeCompleted:
                    # Auxiliar boolean variable to check if a neighbor fulfilling the conditions has been found
                    neighborFound = False
                    # Sweep through neighbors of previousNode
                    for neighbor in self.centerlineGraph.neighbors(previousNode):
                        # If not downstream and there is a node with the same cellId as the mainGraphNode and a higher hierarchy
                        if not downstream and self.centerlineGraph.nodes[neighbor]["cellId"] == self.centerlineGraph.nodes[mainGraphNode]["cellId"] and self.centerlineGraph.nodes[neighbor]["hierarchy femoral"] > self.centerlineGraph.nodes[previousNode]["hierarchy femoral"]:
                            # Update node cellId
                            self.centerlineGraph.nodes[neighbor]["cellId"] = nextCellId
                            # Update edge cellId
                            self.centerlineGraph[previousNode][neighbor]["cellId"] = nextCellId
                            # Update edge segmentsArrayIndices with cutOffIdx
                            self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"] = self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"] - cutOffIdx
                            # Eliminate negative edges if found (this)
                            while self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"][0] < 0 and len(self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"]) > 1:
                                self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"], 0)
                                self.centerlineGraph[previousNode][neighbor]["segmentsCoordinateArray"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsCoordinateArray"], 0, axis = 0)
                                self.centerlineGraph[previousNode][neighbor]["segmentsRadiusArray"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsRadiusArray"], 0)
                            # Update previousNode
                            previousNode = neighbor
                            # Check found neighbor
                            neighborFound = True
                        # If downstream and there is a node with the same cellId as the mainGraphNode and a lower hierarchy
                        elif downstream and self.centerlineGraph.nodes[neighbor]["cellId"] == self.centerlineGraph.nodes[mainGraphNode]["cellId"] and self.centerlineGraph.nodes[neighbor]["hierarchy femoral"] < self.centerlineGraph.nodes[previousNode]["hierarchy femoral"]:
                            # Update node cellId
                            self.centerlineGraph.nodes[neighbor]["cellId"] = nextCellId
                            # Update edge cellId
                            self.centerlineGraph[previousNode][neighbor]["cellId"] = nextCellId
                            # Update edge segmentsArrayIndices with cutOffIdx
                            self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"] = self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"] - cutOffIdx
                            while self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"][0] < 0 and len(self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"]) > 1:
                                self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsArrayIndices"], 0)
                                self.centerlineGraph[previousNode][neighbor]["segmentsCoordinateArray"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsCoordinateArray"], 0, axis = 0)
                                self.centerlineGraph[previousNode][neighbor]["segmentsRadiusArray"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsRadiusArray"], 0)
                            # Update previousNode
                            previousNode = neighbor
                            # Check found neighbor
                            neighborFound = True
                    # When a neighbor fulfilling the conditions is not found, the graph update is complete
                    if not neighborFound:
                        cellIdChangeCompleted = True
                # Now update the segmentsCoordinateArray and the segmentsRadiusArray
                # Create a new object at the end of the array
                self.segmentsCoordinateArray = np.hstack((self.segmentsCoordinateArray, np.empty(1)))
                # The new cell will contain all points from the original cellId from cutOffIdx onwards
                self.segmentsCoordinateArray[nextCellId] = self.segmentsCoordinateArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]][cutOffIdx:]
                # The original cell will only keep points up until cutOffIdx (included)
                self.segmentsCoordinateArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]] = self.segmentsCoordinateArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]][:cutOffIdx + 1]
                # Create a new object at the end of the array
                self.segmentsRadiusArray = np.hstack((self.segmentsRadiusArray, np.empty(1)))
                # The new cell will contain all points from the original cellId from cutOffIdx onwards
                self.segmentsRadiusArray[nextCellId] = self.segmentsRadiusArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]][cutOffIdx:]
                # The original cell will only keep points up until cutOffIdx (included)
                self.segmentsRadiusArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]] = self.segmentsRadiusArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]][:cutOffIdx + 1]
                # Also, add new cellId to label dicts
                self.cellIdToVesselType[nextCellId] = self.cellIdToVesselType[self.centerlineGraph.nodes[mainGraphNode]["cellId"]]
                self.cellIdToVesselTypeName[nextCellId] = self.cellIdToVesselTypeName[self.centerlineGraph.nodes[mainGraphNode]["cellId"]]
                # Update nextCellId
                nextCellId += 1

        # Check for separate subgraphs after graph unification
        subgraphsAux = [self.centerlineGraph.subgraph(components) for components in nx.connected_components(self.centerlineGraph)]

        # If more than one subgraph is found, it is probably a problematic one. We remove all nodes from all remaining secondary subgraphs
        if len(subgraphsAux) > 1:
            positionsSubgraphs = []
            for subgraph in subgraphsAux[1:]:
                for subgraphNode in subgraph:
                    positionsSubgraphs.append(subgraph.nodes[subgraphNode]["pos"])

            removeNodes = []
            for node in self.centerlineGraph:
                if np.amin(np.linalg.norm(self.centerlineGraph.nodes[node]["pos"] - positionsSubgraphs, axis = 1)) < 1e-5:
                    removeNodes.append(node)

            for node in removeNodes:
                self.centerlineGraph.remove_node(node)

        # Save unified graph
        nx.readwrite.gpickle.write_gpickle(self.simpleCenterlineGraph, os.path.join(self.caseDir, "centerlineGraph.pickle"), protocol = 4)
        # Generate plot of dense graph for quick visualization
        makeGraphPlot(self.caseDir, self.centerlineGraph, "centerlineGraph.png")
        
    def getSingleSegments(self):

        def rescaleHierarchy(graph):
            minHierarchy = 1000
            maxHierarchy = 0
            for node in graph:
                if graph.nodes[node]["hierarchy femoral"] < minHierarchy:
                    minHierarchy = graph.nodes[node]["hierarchy femoral"]
                if graph.nodes[node]["hierarchy femoral"] > maxHierarchy:
                    maxHierarchy = graph.nodes[node]["hierarchy femoral"]

            for node in graph:
                graph.nodes[node]["hierarchy femoral"] = graph.nodes[node]["hierarchy femoral"] - minHierarchy

            return graph, maxHierarchy - minHierarchy

        def getCellIds(self):
            # This should be extended to work with cellIds, so that
            self.cellIds = []

            for cellId in self.cellIdToVesselTypeName.keys():
                self.cellIds.append(cellId)
                
            return self

        def getOrientationReference(self):
            AAcoordinatesArray = np.ndarray([0, 3])
            for node in self.centerlineGraph:
                if self.centerlineGraph.nodes[node]["Vessel type name"] == "AA":
                    AAcoordinatesArray = np.append(AAcoordinatesArray, [self.centerlineGraph.nodes[node]["pos"]], axis = 0)

            self.orientationReference = np.mean(AAcoordinatesArray, axis = 0)

            return self

        self = getCellIds(self)

        self = getOrientationReference(self)
        
        self.singleSegments = {}

        for cellId in self.cellIds:
            vesselType = self.cellIdToVesselTypeName[cellId]
            maskedCenterlineGraph = self.centerlineGraph.copy()

            # We want to remove all nodes not connected to at least one edge of the explored vessel type
            removeNodes = []
            for node in maskedCenterlineGraph:
                isVesselType = False
                for neighbor in maskedCenterlineGraph.neighbors(node):
                    if maskedCenterlineGraph[node][neighbor]["cellId"] == cellId:
                        isVesselType = True
                if not isVesselType:
                    removeNodes.append(node)

            for node in removeNodes:
                maskedCenterlineGraph.remove_node(node)

            subgraphs = [maskedCenterlineGraph.subgraph(components) for components in nx.connected_components(maskedCenterlineGraph)]

            # Compute all 1D paths present in all subgraphs
            segments1D = []
            for subgraph in subgraphs:
                subgraph, maxHierarchy = rescaleHierarchy(subgraph)
                # Initialize list for supersegment depending on access
                subgraphSegments1D = []
                # This is used to avoid advancing over finished supersegments segments (supersegment candidates are added when an endnode is reached)
                finishedPaths = []
                for hierarchy in range(maxHierarchy + 1):
                    for node in subgraph:
                        if subgraph.nodes[node]["hierarchy femoral"] == hierarchy:
                            # Store neighbor nodes
                            neighborNodes = []
                            for neighbor in subgraph.neighbors(node):
                                neighborNodes.append(neighbor)
                            # For startNode, just start a new supersegment candidate
                            if len(subgraphSegments1D) == 0:
                                subgraphSegments1D.append([node])
                            # All other nodes
                            else:
                                # In every iteration, search for new segments
                                newSegments = []
                                for idx, pathAux in enumerate(subgraphSegments1D):
                                    path = pathAux.copy()
                                    if idx not in finishedPaths:
                                        # Endpoints
                                        # Since the startNode is treated differently, all nodes with degree == 0 are endpoints
                                        if subgraph.degree(node) == 1 and path[-1] in neighborNodes:
                                            subgraphSegments1D[idx].append(node)
                                            # When an endpoint is reached, add the sequence to finishedPaths to discontinue attention over it
                                            finishedPaths.append(idx)
                                        # Normal node
                                        # For nodes with degree == 2, just add to every active sequence
                                        elif subgraph.degree(node) == 2 and path[-1] in neighborNodes:
                                            subgraphSegments1D[idx].append(node)
                                        # Multifurcations
                                        # For multifurcations, create new segments for all bifurcations except for one (which can continue previously existing segment)
                                        elif subgraph.degree(node) > 2 and path[-1] in neighborNodes:
                                            # Auxiliar boolean variable
                                            firstBifurcation = True
                                            subgraphSegments1D[idx].append(node)
                                            for _, neighbor in enumerate(neighborNodes):
                                                if neighbor != path[-1] and neighbor not in subgraphSegments1D[idx]:
                                                    # For the first neighbor, we add it to the the supersegment candidate
                                                    if firstBifurcation:
                                                        subgraphSegments1D[idx].append(neighbor)
                                                        firstBifurcation = False
                                                    # For the first neighbor, we add it to the the supersegment candidate
                                                    # For all other neighbors, we create new segments
                                                    else:
                                                        newSegment = subgraphSegments1D[idx][:-1].copy()
                                                        newSegment.append(neighbor)
                                                        newSegments.append(newSegment)
                                        # Special case: when two bifurcations come in consecutive nodes
                                        # The second bifurcation will share hierarchy with the neighbors from the first bifurcation,
                                        # and it will have already been added to one of the paths
                                        elif subgraph.degree(node) > 2 and node == path[-1]:
                                            firstBifurcation = True
                                            for _, neighbor in enumerate(neighborNodes):
                                                if neighbor != path[-1] and neighbor not in subgraphSegments1D[idx]:
                                                    # For the first neighbor, we add it to the the supersegment candidate
                                                    if firstBifurcation:
                                                        subgraphSegments1D[idx].append(neighbor)
                                                        firstBifurcation = False
                                                    # For all other neighbors, we create new segments
                                                    else:
                                                        newSegment = subgraphSegments1D[idx][:-1].copy()
                                                        newSegment.append(neighbor)
                                                        newSegments.append(newSegment)
                                # Once a hierarchy index is fully covered, add new segments to the supersegmentCandidates list
                                if len(newSegments) > 0:
                                    for segment in newSegments:
                                        subgraphSegments1D.append(segment)
                for subgraphSegment1D in subgraphSegments1D: 
                    segments1D.append(subgraphSegment1D)

            # Select longest segment (now it is computed from the number of nodes, but could really be computed by the overall actual distance of each segment)
            lengthSegments1D = [len(segment1D) for segment1D in segments1D]
            # Store all nodes not in longest segment
            removeNodes = []
            for node in maskedCenterlineGraph:
                if node not in segments1D[np.argmax(lengthSegments1D)]:
                    removeNodes.append(node)
            # Remove all nodes not in longest segment
            for node in removeNodes:
                maskedCenterlineGraph.remove_node(node)

            # Finally, orient segment. If it is AA, compare position of both endpoints and start hierarchy = 0 in the one closest to the RAS origin 
            # (careful consideration of the LPS/RAS/ijk coordinates!)
            # It it is not AA, compare distance between both endpoints

            if vesselType == "AA":
                endNodes = []
                distanceEndNodes = []
                for node in maskedCenterlineGraph:
                    if maskedCenterlineGraph.degree(node) == 1:
                        endNodes.append(node)
                        # Compute distance to RAS origin
                        distanceEndNodes.append(np.linalg.norm(maskedCenterlineGraph.nodes[node]["pos"] - [0., 0., 0.]))
            else:
                endNodes = []
                distanceEndNodes = []
                for node in maskedCenterlineGraph:
                    if maskedCenterlineGraph.degree(node) == 1:
                        endNodes.append(node)
                        # Compute distance to AA center of mass
                        distanceEndNodes.append(np.linalg.norm(maskedCenterlineGraph.nodes[node]["pos"] - self.orientationReference))

            # Select closest node to reference
            if len(distanceEndNodes) > 0:
                startNode = endNodes[np.argmin(distanceEndNodes)]

                maskedCenterlineGraph = getHierarchicalOrderingDense(maskedCenterlineGraph, startNode = startNode)

                # Ensure right ordering of the segmentsArray values in the edges that might be used for feature computation
                # To do that, compare the direction of the hierarchy values between nodes and the direction of the segmentsArrayIndices values
                # If they are coherent (same direction), do nothing. If they are incoherent, flip the segmentsArray arrays
                for src, dst in maskedCenterlineGraph.edges:
                    if len(maskedCenterlineGraph[src][dst]["segmentsArrayIndices"]) > 0:
                        if np.sign(maskedCenterlineGraph.nodes[src]["hierarchy femoral"] - maskedCenterlineGraph.nodes[dst]["hierarchy femoral"]) == np.sign(maskedCenterlineGraph[src][dst]["segmentsArrayIndices"][-1] - maskedCenterlineGraph[src][dst]["segmentsArrayIndices"][0]):
                            pass
                        else:
                            maskedCenterlineGraph[src][dst]["segmentsArrayIndices"] = np.flip(maskedCenterlineGraph[src][dst]["segmentsArrayIndices"])
                            maskedCenterlineGraph[src][dst]["segmentsCoordinateArray"] = np.flip(maskedCenterlineGraph[src][dst]["segmentsCoordinateArray"], axis = 0)
                            maskedCenterlineGraph[src][dst]["segmentsRadiusArray"] = np.flip(maskedCenterlineGraph[src][dst]["segmentsRadiusArray"])

#             if len(maskedCenterlineGraph) > 2:
            self.singleSegments[cellId] = maskedCenterlineGraph

    def performFeatureExtraction(self):
        ''' Computes hierarchization and performs feature extraction for both accesses for the centerline graph.
        Featurizes self.centerlineGraph.

        Stores centerlineGraph.pickle. 
        
        '''
        # Hierarchization of self.centerlineGraph for femoral access (hierarchy femoral)
        self.centerlineGraph = getHierarchicalOrderingDense(self.centerlineGraph, access = "femoral", startNode = 0)
        # Featurizes self.centerlineGraph (features femoral)
        self.centerlineGraph = extractFeatures(self.centerlineGraph, segmentsCoordinateArray = self.segmentsCoordinateArray, segmentsRadiusArray = self.segmentsRadiusArray, niftiCTA = self.niftiCTA, branchModel = self.branchModel, access = "femoral", edgesToRemove = self.subgraphsUnionEdges, cellIdToVesselType = self.cellIdToVesselType)
        # Hierarchization of self.centerlineGraph for radial access (hierarchy radial)
        self.centerlineGraph = getHierarchicalOrderingDense(self.centerlineGraph, access = "radial", startNode = self.rightmostNode)
        # Featurizes self.centerlineGraph (features radial)
        self.centerlineGraph = extractFeatures(self.centerlineGraph, segmentsCoordinateArray = self.segmentsCoordinateArray, segmentsRadiusArray = self.segmentsRadiusArray, niftiCTA = self.niftiCTA, branchModel = self.branchModel, access = "radial", edgesToRemove = self.subgraphsUnionEdges, cellIdToVesselType = self.cellIdToVesselType)

        # Save featurized graph (with artificial edges)
        nx.readwrite.gpickle.write_gpickle(self.centerlineGraph, os.path.join(self.caseDir, "centerlineGraph.pickle"), protocol = 4)

    def supersegmentExtraction(self):
        ''' Performs supersegment search, reference configuration comparison, and builds featurized supersegment for each
        configuration.

        Creates self.predictedConfigurations and self.supersegments.

        Stores supersegments/*.pickle for each configuration and supersegments.png.
        
        '''

        def supersegmentPrediction(self):
            ''' Performs supersegment search and associates all possible supersegment configurations to all
            reference configurations.

            For each access, starts iterative search at the startNode (hierarchy == 0) and stores all cellIds sequences for the supersegment candidate
            segments as well as the bifurcating segments.

            Then, a one-hot encoded version of the supersegments is generated, encoded for the present vesselTypes in the cellId sequence. This is compared
            to the reference thrombectomy configurations (also one-hot encoded, with enhanced weights for all unique vesselTypes of each configuration) and 
            the cosine similarity between encoded sequences is used as a similarity measure. The supersegment candidate sequence for each configuration is
            selected and stored in self.predictedConfigurations, keeping the cellId sequences of the supersegment and the bifurcating segments.

            Creates self.predictedConfigurations.
            
            '''
            # Create supersegmentCandidates, storing all nodes for all possible paths for each access startNode (hierarchy == 0)
            supersegmentCandidates = {}
            # Initialize dicts for supersegment and bifurcating segments cellIds and vesselTypes
            supersegmentCandidatesCellIds = {}
            supersegmentCandidatesVesselTypes = {}
            bifurcatingSegmentsCandidatesCellIds = {}
            bifurcatingSegmentsCandidatesVesselTypes = {}
            # For both access, perform supersegment search
            for idx, access in enumerate(self.accesses):
                maxHierarchy = getMaxHierarchy(self.centerlineGraph, access)
                # Initialize list for supersegment depending on access
                supersegmentCandidates[access] = []
                # This is used to avoid advancing over finished supersegments segments (supersegment candidates are added when an endnode is reached)
                finishedPaths = []
                # This is necessary to include segments with only one node in the path computations (sometimes happens when two bifurcations are very close)
                numberOfNodesForCellId = {}
                for cellId in range(len(self.segmentsCoordinateArray)):
                    numberOfNodes = 0
                    for node in self.centerlineGraph:
                        if self.centerlineGraph.nodes[node]["cellId"] == cellId:
                            numberOfNodes += 1
                    numberOfNodesForCellId[cellId] = numberOfNodes

                # Now, we loop over all nodes and stack them depending on the hierarchy indices for each access
                # Every time we find a bifurcation, we create a new path. From here, we can extract a cellId sequence for each of the paths    
                # To detect bifurcations, we check the node degree        
                for currentHierarchy in range(maxHierarchy + 1):
                    for node in self.centerlineGraph:
                        if self.centerlineGraph.nodes[node][f"hierarchy {access}"] == currentHierarchy:
                            # Store neighbor nodes
                            neighborNodes = []
                            for neighbor in self.centerlineGraph.neighbors(node):
                                neighborNodes.append(neighbor)
                            # For startNode, just start a new supersegment candidate
                            if len(supersegmentCandidates[access]) == 0:
                                supersegmentCandidates[access].append([node])
                            # All other nodes
                            else:
                                # In every iteration, search for new segments
                                newSegments = []
                                for idx, pathAux in enumerate(supersegmentCandidates[access]):
                                    path = pathAux.copy()
                                    if idx not in finishedPaths:
                                        # Endpoints
                                        # Since the startNode is treated differently, all nodes with degree == 0 are endpoints
                                        if self.centerlineGraph.degree(node) == 1 and path[-1] in neighborNodes:
                                            supersegmentCandidates[access][idx].append(node)
                                            # When an endpoint is reached, add the sequence to finishedPaths to discontinue attention over it
                                            finishedPaths.append(idx)
                                        # Normal node
                                        # For nodes with degree == 2, just add to every active sequence
                                        elif self.centerlineGraph.degree(node) == 2 and path[-1] in neighborNodes:
                                            supersegmentCandidates[access][idx].append(node)
                                        # Multifurcations
                                        # For multifurcations, create new segments for all bifurcations except for one (which can continue previously existing segment)
                                        elif self.centerlineGraph.degree(node) > 2 and path[-1] in neighborNodes:
                                            # Auxiliar boolean variable
                                            firstBifurcation = True
                                            supersegmentCandidates[access][idx].append(node)
                                            for _, neighbor in enumerate(neighborNodes):
                                                if neighbor != path[-1] and neighbor not in supersegmentCandidates[access][idx]:
                                                    # For the first neighbor, we add it to the the supersegment candidate
                                                    if firstBifurcation:
                                                        supersegmentCandidates[access][idx].append(neighbor)
                                                        firstBifurcation = False
                                                    # For the first neighbor, we add it to the the supersegment candidate
                                                    # For all other neighbors, we create new segments
                                                    else:
                                                        newSegment = supersegmentCandidates[access][idx][:-1].copy()
                                                        newSegment.append(neighbor)
                                                        newSegments.append(newSegment)
                                        # Special case: when two bifurcations come in consecutive nodes
                                        # The second bifurcation will share hierarchy with the neighbors from the first bifurcation,
                                        # and it will have already been added to one of the paths
                                        elif self.centerlineGraph.degree(node) > 2 and node == path[-1]:
                                            firstBifurcation = True
                                            for _, neighbor in enumerate(neighborNodes):
                                                if neighbor != path[-1] and neighbor not in supersegmentCandidates[access][idx]:
                                                    # For the first neighbor, we add it to the the supersegment candidate
                                                    if firstBifurcation:
                                                        supersegmentCandidates[access][idx].append(neighbor)
                                                        firstBifurcation = False
                                                    # For all other neighbors, we create new segments
                                                    else:
                                                        newSegment = supersegmentCandidates[access][idx][:-1].copy()
                                                        newSegment.append(neighbor)
                                                        newSegments.append(newSegment)
                                # Once a hierarchy index is fully covered, add new segments to the supersegmentCandidates list
                                if len(newSegments) > 0:
                                    for segment in newSegments:
                                        supersegmentCandidates[access].append(segment)

                # Define an empty list for each access for cellIds and vessel types for supersegment and bifurcating segments
                supersegmentCandidatesCellIds[access] = []   
                bifurcatingSegmentsCandidatesCellIds[access] = []
                supersegmentCandidatesVesselTypes[access] = []
                bifurcatingSegmentsCandidatesVesselTypes[access] = []
                # Add all cellIds for the whole node sequence for every possible path from each access startNode
                for idx, supersegmentCandidate in enumerate(supersegmentCandidates[access]):
                    supersegmentCandidateCellIds = []
                    bifurcatingSegmentsCandidateCellIds = []
                    # Add cellId from first node
                    supersegmentCandidateCellIds.append(self.centerlineGraph.nodes[supersegmentCandidate[0]]["cellId"])
                    # Keep track of previous node
                    previousNode = None
                    # Iterate over all nodes searching for the cellId sequence
                    for node in supersegmentCandidate:
                        # First node
                        if previousNode is None:
                            previousNode = node
                            # Every other node
                        else:
                            # If cellId from current node corresponds to the last added cellId from the supersegment canidate cellId list, skip node
                            if self.centerlineGraph[previousNode][node]["cellId"] != supersegmentCandidateCellIds[-1]:
                                # Else, add cellId from edge, not from node. This helps avoid problems when covering segments that advance downstream
                                supersegmentCandidateCellIds.append(self.centerlineGraph[previousNode][node]["cellId"])
                                # If previous node is a bifurcation, add all other cellIds to bifurcating segments list
                                if self.centerlineGraph.degree(previousNode) > 2:
                                    for neighbor in self.centerlineGraph.neighbors(previousNode):
                                        if self.centerlineGraph[previousNode][neighbor]["cellId"] not in supersegmentCandidateCellIds:
                                            bifurcatingSegmentsCandidateCellIds.append(self.centerlineGraph[previousNode][neighbor]["cellId"])
                            # Update previous node
                            previousNode = node
                    # Add supersegment and bifuracating segment cellId sequences to candidate list
                    supersegmentCandidatesCellIds[access].append(supersegmentCandidateCellIds)
                    bifurcatingSegmentsCandidatesCellIds[access].append(bifurcatingSegmentsCandidateCellIds)
                # Pass cellId sequences to vessel type sequences
                for supersegmentCandidateCellIds in supersegmentCandidatesCellIds[access]:
                    supersegmentCandidateVesselTypes = []
                    for cellId in supersegmentCandidateCellIds:
                        if self.cellIdToVesselTypeName[cellId] not in supersegmentCandidateVesselTypes:
                            supersegmentCandidateVesselTypes.append(self.cellIdToVesselTypeName[cellId])
                    supersegmentCandidatesVesselTypes[access].append(supersegmentCandidateVesselTypes)
                for bifurcatingSegmentsCandidateCellIds in bifurcatingSegmentsCandidatesCellIds[access]:
                    bifurcatingSegmentsCandidateVesselTypes = []
                    for cellId in bifurcatingSegmentsCandidateCellIds:
                        if self.cellIdToVesselTypeName[cellId] not in bifurcatingSegmentsCandidateVesselTypes:
                            bifurcatingSegmentsCandidateVesselTypes.append(self.cellIdToVesselTypeName[cellId])
                    bifurcatingSegmentsCandidatesVesselTypes[access].append(bifurcatingSegmentsCandidateVesselTypes)
                
                # If two segments are exactly the same in terms of vesselTypes but end in two different cellIds, we choose the longer one
                deleteIdx = []
                for idxA, supersegmentCandidateCellIdsA in enumerate(supersegmentCandidatesCellIds[access]):
                    for idxB, supersegmentCandidateCellIdsB in enumerate(supersegmentCandidatesCellIds[access][:idxA]):
                        if supersegmentCandidateCellIdsA[:-1] == supersegmentCandidateCellIdsB[:-1] and self.cellIdToVesselTypeName[supersegmentCandidateCellIdsA[-1]] == self.cellIdToVesselTypeName[supersegmentCandidateCellIdsB[-1]]:
                            distanceA = 0
                            distanceB = 0
                            for n0, n1 in self.centerlineGraph.edges:
                                if self.centerlineGraph[n0][n1]["cellId"] == supersegmentCandidateCellIdsA[-1] and len(self.centerlineGraph[n0][n1]["segmentsArrayIndices"]) != 0:
                                    distanceA += np.linalg.norm(self.centerlineGraph.nodes[n0]["pos"] - self.centerlineGraph.nodes[n1]["pos"])
                                elif self.centerlineGraph[n0][n1]["cellId"] == supersegmentCandidateCellIdsB[-1] and len(self.centerlineGraph[n0][n1]["segmentsArrayIndices"]) != 0:
                                    distanceB += np.linalg.norm(self.centerlineGraph.nodes[n0]["pos"] - self.centerlineGraph.nodes[n1]["pos"])
                            if distanceA > distanceB:
                                deleteIdx.append(idxB)
                            else:
                                deleteIdx.append(idxA)

                # Delete discarded sequences
                supersegmentCandidatesCellIds[access] = list(np.delete(supersegmentCandidatesCellIds[access], deleteIdx))
                supersegmentCandidatesVesselTypes[access] = list(np.delete(supersegmentCandidatesVesselTypes[access], deleteIdx))
                bifurcatingSegmentsCandidatesCellIds[access] = list(np.delete(bifurcatingSegmentsCandidatesCellIds[access], deleteIdx))
                bifurcatingSegmentsCandidatesVesselTypes[access] = list(np.delete(bifurcatingSegmentsCandidatesVesselTypes[access], deleteIdx))

            # Build a one-hot encoded version of the paths (encoding vesselType). We use the same name convention as in the vessel labelling problem
            supersegmentCandidatesOneHot = {}
            for access in self.accesses:
                supersegmentCandidatesOneHot[access] = []
                for sequence in supersegmentCandidatesVesselTypes[access]:
                    if type(sequence) is not list:
                        sequence = [sequence]
                    supersegmentCandidatesOneHot[access].append(vesselTypeSequenceToOneHot(sequence))
                
            # Now we select the closest candidate to each of the reference configurations, using the cosine similarity between the one-hot
            # encoded version of the vesselType sequences and the enhanced one-hot encoded version of the reference configurations
            for access in supersegmentCandidatesOneHot.keys():
                self.predictedConfigurations[access] = []
                for idx, configurationOneHot in enumerate(self.configurationsOneHot[access]):
                    cosineSimilarities = []
                    for supersegmentCandidateOneHot in supersegmentCandidatesOneHot[access]:
                        cosineSimilarities.append(cosineSimilarity(configurationOneHot, supersegmentCandidateOneHot))
                    # For the most similar configuration, we store the cellId sequences for the supersegment and the bifurcating segments
                    self.predictedConfigurations[access].append([supersegmentCandidatesCellIds[access][np.argmax(cosineSimilarities)], bifurcatingSegmentsCandidatesCellIds[access][np.argmax(cosineSimilarities)]])

        def supersegmentBuilt(self):
            ''' Using the self.predictedConfigurations dictionary, creates featurized supersegments for each of the possible
            thrombectomy configurations. Uses the same process as for the creation of the self.centerlineGraph, but including nodes from 
            the cellId sequences stored in self.predictedConfigurations. Nodes from the bifurcating segment cellId list are only included
            up to a distance equal to limiBifurcationLength.

            Creates self.predictedConfigurations and self.supersegments.

            Stores supersegments/*.pickle for each configuration and supersegments.png.
            
            '''
            # Make supersegment directory in case it is missing
            if not os.path.isdir(os.path.join(self.caseDir, "supersegments")): os.mkdir(os.path.join(self.caseDir, "supersegments"))
            if not os.path.isdir(os.path.join(self.caseDir, "supersegments", "dense")): os.mkdir(os.path.join(self.caseDir, "supersegments", "dense"))
            if not os.path.isdir(os.path.join(self.caseDir, "supersegments", "simple")): os.mkdir(os.path.join(self.caseDir, "supersegments", "simple"))

            # Specify the maximum length for a bifurcating segment
            limitBifurcationLength = 15

            # Extract sequences for both accesses
            for access in self.accesses:
                # Initiaize supersegment list for both accesses
                self.supersegments[access] = []
                # Initiaize simpleSupersegment list for both accesses
                self.simpleSupersegments[access] = []
                # Build supersegment for each of the existing reference configurations
                for configIdx, predictedConfiguration in enumerate(self.predictedConfigurations[access]):
                    _, _, _, configurationName = self.configurations[access][configIdx]
                    # Get predicted confirguration
                    supersegmentSegments, bifurcatingSegments = predictedConfiguration
                    # Initialize a graph with networkx for each supersegment 
                    supersegment = self.centerlineGraph.copy()
                    # Define empty list to store self.centerlineGraph nodes connected by edges with cellId in supersegmentSegments
                    supersegmentNodes = []
                    # Define empty list to store self.centerlineGraph edges with cellId in supersegmentSegments
                    supersegmentEdges = []
                    # Define empty list to store self.centerlineGraph nodes connected by edges with cellId in bifurcatingSegments up to the limitBifurcationLength
                    bifurcatingNodes = []
                    # Define empty list to store first self.centerlineGraph edges with cellId in bifurcatingSegments
                    bifurcatingEdges = []
                    
                    # Store all corresponding nodes and edges in supersegmentNodes and supersegmentEdges
                    for src, dst in supersegment.edges:
                        if supersegment[src][dst]["cellId"] in supersegmentSegments:
                            if src not in supersegmentNodes:
                                supersegmentNodes.append(src)
                            if dst not in supersegmentNodes:
                                supersegmentNodes.append(dst)
                            supersegmentEdges.append((src, dst))
                            
                    # Store all corresponding nodes and edges from immediate bfiurcating edges in bifurcatingNodes and bifurcatingEdges
                    for src, dst in supersegment.edges:
                        if (src, dst) not in supersegmentEdges and supersegment[src][dst]["cellId"] in bifurcatingSegments:
                            if src in supersegmentNodes and dst not in supersegmentNodes:
                                bifurcatingNodes.append(dst)
                                bifurcatingEdges.append((src, dst))
                            elif dst in supersegmentNodes and src not in supersegmentNodes:
                                bifurcatingNodes.append(src)
                                bifurcatingEdges.append((src, dst))
                                
                    # Store all corresponding nodes left in bifurcatingNodes and bifurcatingEdges
                    for src, dst in bifurcatingEdges:
                        if src in bifurcatingNodes:
                            currentDst = src
                        else:
                            currentDst = dst
                        distance = 0
                        continueSearch = True
                        check = True
                        # We only include cases where first bifurcating edge is connected to a deg = 2 node. Else, we only include the first bifurcating node
                        if supersegment.degree(currentDst) == 2:
                            while distance < limitBifurcationLength and continueSearch and check:
                                check = False
                                for neighbor in supersegment.neighbors(currentDst):
                                    if neighbor not in supersegmentNodes and neighbor not in bifurcatingNodes:
                                        check = True
                                        if supersegment.degree(neighbor) == 2:
                                            bifurcatingNodes.append(neighbor)
                                            # distance += supersegment[currentDst][neighbor][f"features {access}"]["Segment length"]
                                            distance += np.linalg.norm(supersegment.nodes[currentDst]["pos"] - supersegment.nodes[neighbor]["pos"])
                                            currentDst = neighbor
                                        else:
                                            bifurcatingNodes.append(neighbor)
                                            continueSearch = False

                    # We now search for all non-included nodes from hierarchicDenseG, which will be masked out
                    removeNodes = []
                    for node in supersegment:
                        if node not in supersegmentNodes and node not in bifurcatingNodes:
                            removeNodes.append(node)
                        else:
                            if node in supersegmentNodes:
                                supersegment.nodes[node]["isSupersegment"] = 1
                            else:
                                supersegment.nodes[node]["isSupersegment"] = 0
                            supersegment.nodes[node]["hierarchy"] = supersegment.nodes[node][f"hierarchy {access}"]
                            supersegment.nodes[node]["features"] = supersegment.nodes[node][f"features {access}"]
                            supersegment.nodes[node]["features"]["isSupersegment"] = supersegment.nodes[node]["isSupersegment"]
                            supersegment.nodes[node].pop("features femoral")
                            supersegment.nodes[node].pop("features radial")
                            supersegment.nodes[node].pop("hierarchy femoral")
                            supersegment.nodes[node].pop("hierarchy radial")

                    for src, dst in supersegment.edges:
                        if (src, dst) in supersegmentEdges:
                            supersegment[src][dst]["isSupersegment"] = 1
                        else:
                            supersegment[src][dst]["isSupersegment"] = 0

                    # Perform masking (remove non-included nodes)
                    for node in removeNodes:
                        supersegment.remove_node(node)
                            
                    # Relabel nodes as sequential labels
                    mapping = {}
                    newNode = 0
                    for oldNode in supersegment.nodes():
                        mapping[oldNode] = newNode
                        newNode += 1
                    supersegment = nx.relabel.relabel_nodes(supersegment, mapping)

                    # Add to the supersegments dict
                    self.supersegments[access].append(supersegment)
                    # Save supersegment as pickle
                    nx.readwrite.gpickle.write_gpickle(supersegment, os.path.join(self.caseDir, "supersegments", "dense", f"{configurationName}.pickle"), protocol = 4)

                    # Initialize a graph with networkx for each supersegment 
                    simpleSupersegment = self.simpleCenterlineGraph.copy()
                    # Define empty list to store self.centerlineGraph nodes connected by edges with cellId in supersegmentSegments
                    supersegmentNodes = []
                    # Define empty list to store self.centerlineGraph edges with cellId in supersegmentSegments
                    supersegmentEdges = []
                    # Define empty list to store self.centerlineGraph nodes connected by edges with cellId in bifurcatingSegments up to the limitBifurcationLength
                    bifurcatingNodes = []
                    # Define empty list to store first self.centerlineGraph edges with cellId in bifurcatingSegments
                    bifurcatingEdges = []
                    
                    # Store all corresponding nodes and edges in supersegmentNodes and supersegmentEdges
                    for src, dst in simpleSupersegment.edges:
                        if simpleSupersegment[src][dst]["cellId"] in supersegmentSegments:
                            if src not in supersegmentNodes:
                                supersegmentNodes.append(src)
                            if dst not in supersegmentNodes:
                                supersegmentNodes.append(dst)
                            supersegmentEdges.append((src, dst))
                            
                    # Store all corresponding nodes and edges from immediate bfiurcating edges in bifurcatingNodes and bifurcatingEdges
                    for src, dst in simpleSupersegment.edges:
                        if (src, dst) not in supersegmentEdges and simpleSupersegment[src][dst]["cellId"] in bifurcatingSegments:
                            if src in supersegmentNodes and dst not in supersegmentNodes:
                                bifurcatingNodes.append(dst)
                                bifurcatingEdges.append((src, dst))
                            elif dst in supersegmentNodes and src not in supersegmentNodes:
                                bifurcatingNodes.append(src)
                                bifurcatingEdges.append((src, dst))
                                
                    # Store all corresponding nodes left in bifurcatingNodes and bifurcatingEdges
                    for src, dst in bifurcatingEdges:
                        if src in bifurcatingNodes:
                            currentDst = src
                        else:
                            currentDst = dst
                        distance = 0
                        continueSearch = True
                        check = True
                        # We only include cases where first bifurcating edge is connected to a deg = 2 node. Else, we only include the first bifurcating node
                        if simpleSupersegment.degree(currentDst) == 2:
                            while distance < limitBifurcationLength and continueSearch and check:
                                check = False
                                for neighbor in simpleSupersegment.neighbors(currentDst):
                                    if neighbor not in supersegmentNodes and neighbor not in bifurcatingNodes:
                                        check = True
                                        if simpleSupersegment.degree(neighbor) == 2:
                                            bifurcatingNodes.append(neighbor)
                                            distance += np.linalg.norm(simpleSupersegment.nodes[currentDst]["pos"] - simpleSupersegment.nodes[neighbor]["pos"])
                                            currentDst = neighbor
                                        else:
                                            bifurcatingNodes.append(neighbor)
                                            continueSearch = False

                    # We now search for all non-included nodes from hierarchicDenseG, which will be masked out
                    removeNodes = []
                    for node in simpleSupersegment:
                        if node not in supersegmentNodes and node not in bifurcatingNodes:
                            removeNodes.append(node)
                        else:
                            if node in supersegmentNodes:
                                simpleSupersegment.nodes[node]["isSupersegment"] = 1
                            else:
                                simpleSupersegment.nodes[node]["isSupersegment"] = 0

                    for src, dst in simpleSupersegment.edges:
                        if (src, dst) in supersegmentEdges:
                            simpleSupersegment[src][dst]["isSupersegment"] = 1
                        else:
                            simpleSupersegment[src][dst]["isSupersegment"] = 0

                    # Perform masking (remove non-included nodes)
                    for node in removeNodes:
                        simpleSupersegment.remove_node(node)
                            
                    # Relabel nodes as sequential labels
                    mapping = {}
                    newNode = 0
                    for oldNode in simpleSupersegment.nodes():
                        mapping[oldNode] = newNode
                        newNode += 1
                    simpleSupersegment = nx.relabel.relabel_nodes(simpleSupersegment, mapping)

                    # Add to the supersegments dict
                    self.simpleSupersegments[access].append(simpleSupersegment)
                    # Save simpleSupersegment as pickle
                    nx.readwrite.gpickle.write_gpickle(simpleSupersegment, os.path.join(self.caseDir, "supersegments", "simple", f"{configurationName}.pickle"), protocol = 4)
            # Make plot with all supersegments
            makeSupersegmentPlots(self.caseDir, self.supersegments, isSimple = False)
            # Make plot with all supersegments
            makeSupersegmentPlots(self.caseDir, self.simpleSupersegments, isSimple = True)

        def selectConfiguration(self):
            ''' Selects supersegment configuration if patientConfiguration.json is present in self.caseDir.

            Creats new dir self.caseDir/thrombectomyConfiguration, and stores selected supersegment and image.
            
            '''

            with open(os.path.join(self.caseDir, "patientConfiguration.json")) as jsonFile:
                self.patientConfiguration = json.load(jsonFile)[self.caseId]

            # If laterality for thrombectomy is undetermined but it is known that occlusion was vertebrobasilar, choose side with larger VA (mean radius)
            if "Vertebrobasilar" in self.patientConfiguration["Laterality"]:
                self.patientConfiguration["Laterality"] = selectVertebrobasilarLaterality(self.centerlineGraph, self.patientConfiguration["Laterality"])

            if self.patientConfiguration["Laterality"] in ["Right", "Left"]:
                configurationId = 0
                if self.patientConfiguration["Access"] == "Femoral":
                    configurationId += 0
                elif self.patientConfiguration["Access"] == "Radial": 
                    configurationId += 4
                    
                if self.patientConfiguration["Laterality"] == "Right":
                    configurationId += 0
                elif self.patientConfiguration["Laterality"] == "Left": 
                    configurationId += 2
                    
                if self.patientConfiguration["Antero-posterior"] == "Anterior":
                    configurationId += 0
                elif self.patientConfiguration["Antero-posterior"] == "Posterior": 
                    configurationId += 1
                    
                print("         Access:", self.patientConfiguration["Access"])
                print("         Laterality:", self.patientConfiguration["Laterality"])
                print("         Antero-posterior:", self.patientConfiguration["Antero-posterior"])
                print("         Configuration selected:", configurationId)

                if not os.path.isdir(os.path.join(self.caseDir, "thrombectomyConfiguration")): os.mkdir(os.path.join(self.caseDir, "thrombectomyConfiguration"))

                for supersegment in [supersegment for supersegment in os.listdir(os.path.join(self.caseDir, "supersegments", "dense")) if supersegment.endswith(".pickle") and supersegment.startswith(str(configurationId))]:
                    print("         Selecting supersegment:", supersegment)
                    shutil.copyfile(os.path.join(self.caseDir, "supersegments", "dense", supersegment), os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.pickle"))
                    makeSupersegmentPlot(self.caseDir, nx.readwrite.gpickle.read_gpickle(os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.pickle")), self.patientConfiguration, isSimple = False)
                for simpleSupersegment in [simpleSupersegment for simpleSupersegment in os.listdir(os.path.join(self.caseDir, "supersegments", "simple")) if simpleSupersegment.endswith(".pickle") and simpleSupersegment.startswith(str(configurationId))]:
                    print("         Selecting supersegment:", simpleSupersegment)
                    shutil.copyfile(os.path.join(self.caseDir, "supersegments", "simple", simpleSupersegment), os.path.join(self.caseDir, "thrombectomyConfiguration", "simpleSupersegment.pickle"))
                    makeSupersegmentPlot(self.caseDir, nx.readwrite.gpickle.read_gpickle(os.path.join(self.caseDir, "thrombectomyConfiguration", "simpleSupersegment.pickle")), self.patientConfiguration, isSimple = True)

                self.supersegment = nx.read_gpickle(os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.pickle"))
                self.simpleSupersegment = nx.read_gpickle(os.path.join(self.caseDir, "thrombectomyConfiguration", "simpleSupersegment.pickle"))
                self.supersegment = addConfigurationFeatures(self.supersegment, self.patientConfiguration)

                self.supersegment.graph["features"] = {}
                for feature in self.supersegment.graph.keys():
                    if feature not in ["Time to first series", "features", "DCP"]:
                        self.supersegment.graph["features"][feature] = self.supersegment.graph[feature]

                # Specially added for database preparation
                self.supersegment.graph["Time to first series"] = self.patientConfiguration["Time first angiography"]
                if self.patientConfiguration["Time first angiography"] <= 15:
                    self.supersegment.graph["Time to first series class"] = 0
                else:
                    self.supersegment.graph["Time to first series class"] = 1

                nx.write_gpickle(self.supersegment, os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.pickle"))
                nx.write_gpickle(self.supersegment, os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/supersegments/dense/graphs", f"{self.caseId}.pickle"))
                nx.write_gpickle(self.simpleSupersegment, os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/supersegments/simple/graphs", f"{self.caseId}.pickle"))
                shutil.copyfile(os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.png"), os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/supersegments/dense/images", f"{self.caseId}.png"))
                shutil.copyfile(os.path.join(self.caseDir, "thrombectomyConfiguration", "simpleSupersegment.png"), os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/supersegments/simple/images", f"{self.caseId}.png"))

            else:
                print("        Laterality is ambiguous:", self.patientConfiguration["Laterality"])
        
        # Perform both methods to perform supersegment extraction
        supersegmentPrediction(self)
        supersegmentBuilt(self)
        selectConfiguration(self)