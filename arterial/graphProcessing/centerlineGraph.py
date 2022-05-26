import os
import json
import shutil

import numpy as np
import networkx as nx

import nibabel as nib

from .centerlineGraphUtils import getHierarchicalOrderingDense, getMaxHierarchy, extractFeatures, vesselTypeSequenceToOneHot, cosineSimilarity, makeGraphPlot, makeSupersegmentPlots, makeSupersegmentPlot
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
        # Get radius array from segmentsArray
        self.segmentsRadiusArray = self.segmentsArray[:, 1]

        # Initialize simplified graph with networkx
        self.simpleCenterlineGraph = nx.Graph()
        
        # Set reference scale for node sampling
        self.sampleNodeEvery = 50

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
                    self.centerlineGraph.nodes[totalNodes]["cellId"] = cellId
                    totalNodes += 1
                    previousPosition = position
                # If distance from last sampled node is larger than selected sampleNodeEvery, add a node with cellId and an edge to the previous sample 
                # node, keeping cellId and the indices of the centerline points in the segmentsArray
                elif distance > self.sampleNodeEvery:
                    self.centerlineGraph.add_node(totalNodes, pos=position)
                    self.centerlineGraph.nodes[totalNodes]["cellId"] = cellId
                    self.centerlineGraph.add_edge(totalNodes - 1, totalNodes)
                    self.centerlineGraph[totalNodes - 1][totalNodes]["cellId"] = cellId
                    self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayPoints"] = np.arange(previousIdx, idx)
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
                        self.centerlineGraph.nodes[totalNodes]["cellId"] = cellId
                        self.centerlineGraph.add_edge(totalNodes - 1, totalNodes)
                        self.centerlineGraph[totalNodes - 1][totalNodes]["cellId"] = cellId
                        self.centerlineGraph[totalNodes - 1][totalNodes]["segmentsArrayPoints"] = np.arange(previousIdx, idx)
                        totalNodes += 1
                    # If it is not, which will be the general case, we do not add a new node, but instead we transform the last added node and change its position to be placed at the bifurcation/endpoint
                    else:
                        self.centerlineGraph.nodes[totalNodes - 1]["pos"]= position
                        self.centerlineGraph.nodes[totalNodes - 1]["cellId"] = cellId
                        self.centerlineGraph[totalNodes - 2][totalNodes - 1]["cellId"] = cellId
                        self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsArrayPoints"] = np.append(self.centerlineGraph[totalNodes - 2][totalNodes - 1]["segmentsArrayPoints"], np.arange(previousIdx, idx))
                # If the accumulated distance from the last sample node is smaller than sampleNodeEvery, and we are not in either the first or last nodes of the segmentsArray cell, just update the distance
                else:
                    distance += np.linalg.norm(previousPosition - position)
                    
        # Merge nodes that share the same coordinate (bifurcation spots)
        # First get all nodes that have a degree of 1 (start- and endpoints)
        deg1Nodes = []
        for node, deg in self.centerlineGraph.degree:
            if deg == 1:
                deg1Nodes.append(node)

        # For all degree 1 nodes, we check position to join corresponding startpoints and endpoints
        removedNodes = []
        for _, node in enumerate(deg1Nodes):
            if node not in removedNodes:
                aux = deg1Nodes.copy()
                aux.remove(node)
                for auxNodes in removedNodes:
                    aux.remove(auxNodes)
                for _, node2 in enumerate(aux):
                    C1 = self.centerlineGraph.nodes[node]["pos"]
                    C2 = self.centerlineGraph.nodes[node2]["pos"]
                    if C1[0] == C2[0] and C1[1] == C2[1] and C1[2] == C2[2]:
                        self.centerlineGraph = nx.contracted_nodes(self.centerlineGraph, node, node2)
                        removedNodes.append(node2)
                        self.centerlineGraph.nodes[node].pop("contraction")           
                        
        # Relabel nodes as sequential labels
        mapping = {}
        newNode = 0
        for oldNode in self.centerlineGraph.nodes():
            mapping[oldNode] = newNode
            newNode += 1
        self.centerlineGraph = nx.relabel.relabel_nodes(self.centerlineGraph, mapping)

        # The self.rightmostNode node can be used for hierarchical indexing from radial access
        removeNodes = []
        rightmostNodePosition = self.segmentsCoordinateArray[0][0]
        for node in self.centerlineGraph:
            if self.centerlineGraph.nodes[node]["pos"][0] < rightmostNodePosition[0] and self.centerlineGraph.degree(node) == 1:
                self.rightmostNode = node
                rightmostNodePosition = self.centerlineGraph.nodes[node]["pos"]
            # If any nodes with degree == 0 are present, remove them
            if self.centerlineGraph.degree(node) == 0:
                removeNodes.append(node)
        for node in removeNodes:
            self.centerlineGraph.remove_node(node)

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
            subgraph = extractFeatures(self.caseDir, subgraph, self.segmentsCoordinateArray, self.segmentsRadiusArray, "femoral", featureExtractionForVesselLabelling = True)
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
            print("     Found random islands containing the follwing cellId:", deleteCellIds)
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

    def makeSimpleCenterlineGraph(self):
        ''' Elaborate simplified graph (each segmnet from the segmentsArray corresponds to each graph edge).

        Creates self.simpleCenterlineGraph.

        Stores graph.pickle and graph.png.
        
        '''
        # # We can make a simplified version of the graph for visualization purposes (cellId easy visualization)

        # # Only taking first and last positions of the curves arrays
        # totalNodes = 0 # We only link nodes from the same centerline
        # for cellId, curve in enumerate(self.segmentsCoordinateArray):
        #     if len(curve) > 1 and len(self.segmentsRadiusArray[cellId]) > 1:
        #         # Add nodes
        #         # First node of cell (startpoint)
        #         self.simpleCenterlineGraph.add_node(totalNodes + 0, pos=curve[0])
        #         # Last node of cell (endpoint)
        #         self.simpleCenterlineGraph.add_node(totalNodes + 1, pos=curve[-1])
        #         # Add edges
        #         self.simpleCenterlineGraph.add_edge(totalNodes, totalNodes + 1, cellId = cellId)
        #         totalNodes += 2   

        # # Merge nodes that share the same RAS coordinate (bifurcation spots)
        # # First get all nodes that have a degree of 1 (start- and endpoints)
        # deg1Nodes = []
        # for node, deg in self.simpleCenterlineGraph.degree:
        #     if deg == 1:
        #         deg1Nodes.append(node)
                
        # # For all degree 1 nodes, we check position to join corresponding start- and enpoints, as well as bifurcations
        # removedNodes = []
        # for _, node in enumerate(deg1Nodes):
        #     if node not in removedNodes:
        #         aux = deg1Nodes.copy()
        #         aux.remove(node)
        #         for auxNodes in removedNodes:
        #             aux.remove(auxNodes)
        #         for _, node2 in enumerate(aux):
        #             C1 = self.simpleCenterlineGraph.nodes[node]["pos"]
        #             C2 = self.simpleCenterlineGraph.nodes[node2]["pos"]
        #             if C1[0] == C2[0] and C1[1] == C2[1] and C1[2] == C2[2]:
        #                 self.simpleCenterlineGraph = nx.contracted_nodes(self.simpleCenterlineGraph, node, node2)
        #                 removedNodes.append(node2)
        #                 self.simpleCenterlineGraph.nodes[node].pop("contraction")

        # # Relabel nodes as sequential labels
        # mapping = {}
        # newNode = 0
        # for oldNode in self.simpleCenterlineGraph.nodes():
        #     mapping[oldNode] = newNode
        #     newNode += 1
        # self.simpleCenterlineGraph = nx.relabel.relabel_nodes(self.simpleCenterlineGraph, mapping)

        # # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
        # node_pos_dict_P = {}
        # for n in self.simpleCenterlineGraph.nodes():
        #     node_pos_dict_P[n] = [self.simpleCenterlineGraph.nodes[n]["pos"][0], self.simpleCenterlineGraph.nodes[n]["pos"][2]]

        # # For a coronal view, we use P and S coordinates (the view will be from the coronal plane, R axis)
        # node_pos_dict_R = {}
        # for n in self.simpleCenterlineGraph.nodes():
        #     node_pos_dict_R[n] = [self.simpleCenterlineGraph.nodes[n]["pos"][1], self.simpleCenterlineGraph.nodes[n]["pos"][2]]

        # # Save simplified graph
        # nx.readwrite.gpickle.write_gpickle(self.simpleCenterlineGraph, os.path.join(self.caseDir, "graph.pickle"), protocol = 4)
        # # Make quick plot for easy visualization
        # makeGraphPlot(self.caseDir, self.simpleCenterlineGraph, "simpleGraph.png")

        ######### This is provisional for vessel labelling

        def directionalEmbeddings(G, node):
            ''' Computes the major directions one-hot vector associated with the inward directions of the 
            edges parting off an node. Serves as a node features for the GNN.

            Arguments:
                - G <networkx graph>: networkx graph derived from segmentsArray.
                - node <int>: node id.

            Returns:
                - projectedMajorDirections <numpy array>: numpy array of dimensions [26] with ones
                at the positions corresponding with the edge directions parting from the node, and 
                zeros in the rest of positions.

            '''
            import math

            projectedMajorDirections = np.zeros([26])
            majorDirections = np.ndarray([0, 3])
            for a in range(8):
                for b in range(-2, 3):
                    majorDirections = np.append(majorDirections, [
                            [math.sin(math.pi * a / 4) * math.cos(math.pi * b / 4), 
                            math.cos(math.pi * a / 4) * math.cos(math.pi * b / 4), 
                            math.sin(math.pi * b / 4)]], axis=0)
            majorDirections[np.abs(majorDirections) < 0.01] = 0.
            majorDirections = np.unique(np.around(majorDirections, 10), axis=0)
            
            for edge in G.edges(node):
                direction = G.edges[edge]["direction"]
                projectedMajorDirections[np.argmax(np.abs(np.matmul(majorDirections, direction)))] += 1
                
            return projectedMajorDirections

        def relativeLength(segmentCoordinates):
            ''' Computes relative length for a given segment.

            Arguments:
                - segmentCoordinates: coordinats for a given centerline segment.

            Returns:
                - RL: relative length.
                
            '''

            def distanceAlongCenterline(centerline):
                    distance = 0
                    for idx in range(1, len(centerline)):
                        distance += np.linalg.norm(centerline[idx] - centerline[idx - 1])
                        
                    return distance
                
            euclideanDistance = np.linalg.norm(segmentCoordinates[-1] - segmentCoordinates[0])
            centerlineDistance = distanceAlongCenterline(segmentCoordinates)

            return euclideanDistance / centerlineDistance

        # We can make a simplified version of the graph for visualization purposes (cellId easy visualization)
       
        # Only taking first and last positions of the curves arrays
        totalNodes = 0 # We only link nodes from the same centerline
        for cellId, curve in enumerate(self.segmentsCoordinateArray):
            if len(curve) > 1 and len(self.segmentsRadiusArray[cellId]) > 1:
                # Add nodes
                # First node of cell (startpoint)
                self.simpleCenterlineGraph.add_node(totalNodes + 0, pos=curve[0])
                self.simpleCenterlineGraph.nodes[totalNodes + 0]["deg"] = self.simpleCenterlineGraph.degree[totalNodes + 0]
                self.simpleCenterlineGraph.nodes[totalNodes + 0]["rad"] = self.segmentsRadiusArray[cellId][0]
                # Build features array
                self.simpleCenterlineGraph.nodes[totalNodes + 0]["features"] = np.array([curve[0][0], curve[0][1], curve[0][2], self.segmentsRadiusArray[cellId][0], self.simpleCenterlineGraph.degree[totalNodes + 0]])

                # Last node of cell (endpoint)
                self.simpleCenterlineGraph.add_node(totalNodes + 1, pos=curve[-1])
                self.simpleCenterlineGraph.nodes[totalNodes + 1]["deg"] = self.simpleCenterlineGraph.degree[totalNodes + 1]
                self.simpleCenterlineGraph.nodes[totalNodes + 1]["rad"] = self.segmentsRadiusArray[cellId][-1]
                # Build node features array
                self.simpleCenterlineGraph.nodes[totalNodes + 1]["features"] = np.array([curve[-1][0], curve[-1][1], curve[-1][2], self.segmentsRadiusArray[cellId][-1], self.simpleCenterlineGraph.degree[totalNodes + 1]])

                # Add edges
                self.simpleCenterlineGraph.add_edge(totalNodes, totalNodes + 1, cellId = cellId)
                distance = np.linalg.norm(curve[-1] - curve[0])
                direction = (curve[-1] - curve[0]) / distance
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["mean rad"] = np.mean(self.segmentsRadiusArray[cellId])
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal radius"] = self.segmentsRadiusArray[cellId][0]
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distal radius"] = self.segmentsRadiusArray[cellId][-1]
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal/distal radius ratio"] = self.segmentsRadiusArray[cellId][0] / self.segmentsRadiusArray[cellId][-1]
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["minimum radius"] = np.amin(self.segmentsRadiusArray[cellId])
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["maximum radius"] = np.amax(self.segmentsRadiusArray[cellId])
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distance"] = distance
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["relative length"] = relativeLength(self.segmentsCoordinateArray[cellId])
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["direction"] = direction
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["departure angle"] = (self.segmentsCoordinateArray[cellId][1] - self.segmentsCoordinateArray[cellId][0]) / np.linalg.norm(self.segmentsCoordinateArray[cellId][1] - self.segmentsCoordinateArray[cellId][0])
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["number of points"] = len(self.segmentsCoordinateArray[cellId])
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal bifurcation position"] = self.segmentsCoordinateArray[cellId][0]
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distal bifurcation position"] = self.segmentsCoordinateArray[cellId][-1]
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["pos"] = np.sum(self.segmentsCoordinateArray[cellId], axis = 0) / len(self.segmentsCoordinateArray[cellId])
                # Build edge feature array
                self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["features"] = np.array([self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["mean rad"], 
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal radius"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distal radius"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal/distal radius ratio"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["minimum radius"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["maximum radius"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distance"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["relative length"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["direction"][0],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["direction"][1],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["direction"][2],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["departure angle"][0],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["departure angle"][1],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["departure angle"][2],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["number of points"],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal bifurcation position"][0],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal bifurcation position"][1],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["proximal bifurcation position"][2],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distal bifurcation position"][0],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distal bifurcation position"][1],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["distal bifurcation position"][2],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["pos"][0],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["pos"][1],
                                                                        self.simpleCenterlineGraph[totalNodes][totalNodes + 1]["pos"][2]
                ])
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

        # Directional embeddings for node features have to be computed after all edge directions for the whole graph are computed
        for node in self.simpleCenterlineGraph.nodes:
            projectedMajorDirections = directionalEmbeddings(self.simpleCenterlineGraph, node)
            self.simpleCenterlineGraph.nodes[node]["dir"] = projectedMajorDirections
            self.simpleCenterlineGraph.nodes[node]["features"] = np.append(self.simpleCenterlineGraph.nodes[node]["features"], projectedMajorDirections)

        # Save simplified graph
        nx.readwrite.gpickle.write_gpickle(self.simpleCenterlineGraph, os.path.join(self.caseDir, "graph.pickle"))
        # Make quick plot for easy visualization
        makeGraphPlot(self.caseDir, self.simpleCenterlineGraph, "simpleGraph.png", label = "cellId")

    def predictVesselTypes(self):
        ''' Performs graph U-Net inference with simplified graph and creates self.cellIdToVesselTypeName.

        Creates self.predictedSimpleCenterlineGraph and self.cellIdToVesselTypeName.
        
        '''
        # Perform vesselType prediction
        self.predictedSimpleCenterlineGraph = predictGraph(self.caseDir)
        # If manually labelled are to be used for labelling of dense graphs
        # self.predictedSimpleCenterlineGraph = nx.readwrite.gpickle.read_gpickle(os.path.join(self.caseDir, "graph_label.pickle"))
        # Get cellId to vesselType dict from predicted graph
        for n0, n1 in self.predictedSimpleCenterlineGraph.edges:
            self.cellIdToVesselType[self.predictedSimpleCenterlineGraph[n0][n1]["cellId"]] = self.predictedSimpleCenterlineGraph[n0][n1]["vessel type"]
            self.cellIdToVesselTypeName[self.predictedSimpleCenterlineGraph[n0][n1]["cellId"]] = self.predictedSimpleCenterlineGraph[n0][n1]["vessel type name"]

        for node in self.centerlineGraph:
            self.centerlineGraph.nodes[node]["vessel type"] = self.cellIdToVesselType[self.centerlineGraph.nodes[node]["cellId"]]
            self.centerlineGraph.nodes[node]["vessel type name"] = self.cellIdToVesselTypeName[self.centerlineGraph.nodes[node]["cellId"]]

        # Save graph with predicted edge types
        nx.readwrite.gpickle.write_gpickle(self.centerlineGraph, os.path.join(self.caseDir, "centerlineGraph.pickle"), protocol = 4)

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

        # subgraphsUnionEdges = []

        # # Search for alternate union points in main subgraph or larger subgraphs following a preference system
        # for idx, candidatesForSubgraphUnion in enumerate(candidatesForSubgraphsUnion):
        #     for candidateIdx, candidateNode in enumerate(candidatesForSubgraphUnion):
        #         # Pool all node and positions from the main graph with the same vesselType as the candidate node
        #         candidateMainGraphCoordinates = np.ndarray([0, 3])
        #         candidateMainGraphNodes = []
        #         # Boolean variable to end search for union node
        #         foundUnion = False
        #         for subgraphIdx in range(max([idx, 1])):
        #             if self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]] in [self.cellIdToVesselTypeName[cellId] for cellId in subgraphsCellIds[subgraphIdx]] and not foundUnion:
        #             # Check if the same vessel type exists in the larger subgraphs
        #                 # If it exists and has a different cellId, store all node coordinates
        #                 for nodeSubgraphIdx in self.subgraphs[subgraphIdx]:
        #                     if self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]] == self.cellIdToVesselTypeName[self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["cellId"]] and self.subgraphs[idx].nodes[candidateNode]["cellId"] != self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["cellId"]:
        #                         # In the rare event that the connection node is found to be the self.rightmostNode, then choose its neighbor
        #                         if nodeSubgraphIdx == self.rightmostNode:
        #                             nodeSubgraphIdx = self.subgraphs[subgraphIdx].neighbors(self.rightmostNode).__next__()
        #                         # In the case that we are looking at the same subgraph as the candidate node, group cellIds in contact with candidate node. Forbid union with segments in contact
        #                         if subgraphIdx == idx:
        #                             cellIdsInContact = []
        #                             for neighbor in self.subgraphs[subgraphIdx].neighbors(oppositeNodesForSubgraphsUnion[idx][candidateIdx]):
        #                                 cellIdsInContact.append(self.subgraphs[subgraphIdx].nodes[neighbor]["cellId"])
        #                             if self.subgraphs[idx].nodes[nodeSubgraphIdx]["cellId"] in cellIdsInContact:
        #                                 pass
        #                             else:
        #                                 candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
        #                                 candidateMainGraphNodes.append(nodeSubgraphIdx)
        #                                 foundUnion = True
        #                         # Otherwise, go on with analysis
        #                         else:
        #                             candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
        #                             candidateMainGraphNodes.append(nodeSubgraphIdx)
        #                             foundUnion = True
        #         # Check if the preferred vessel type exists in the larger subgraphs
        #         # Look at all graphs larger than the one we are analyzing (except for main subgraph, where we only look at itself)
        #         for subgraphIdx in range(max([idx, 1])):
        #             if preferredVesselTypes[self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]]] in [self.cellIdToVesselTypeName[cellId] for cellId in subgraphsCellIds[subgraphIdx]] and not foundUnion:
        #                 # If it exists, store all node coordinates
        #                 for nodeSubgraphIdx in self.subgraphs[subgraphIdx]:
        #                     if preferredVesselTypes[self.cellIdToVesselTypeName[self.subgraphs[idx].nodes[candidateNode]["cellId"]]] == self.cellIdToVesselTypeName[self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["cellId"]]:
        #                         # In the rare event that the connection node is found to be the self.rightmostNode, then choose its neighbor
        #                         if nodeSubgraphIdx == self.rightmostNode:
        #                             nodeSubgraphIdx = self.subgraphs[subgraphIdx].neighbors(self.rightmostNode).__next__()
        #                         # In the case that we are looking at the same subgraph as the candidate node, group cellIds in contact with candidate node. Forbid union with segments in contact
        #                         if subgraphIdx == idx:
        #                             cellIdsInContact = []
        #                             for neighbor in self.subgraphs[subgraphIdx].neighbors(oppositeNodesForSubgraphsUnion[idx][candidateIdx]):
        #                                 cellIdsInContact.append(self.subgraphs[subgraphIdx].nodes[neighbor]["cellId"])
        #                             if self.subgraphs[idx].nodes[nodeSubgraphIdx]["cellId"] in cellIdsInContact:
        #                                 pass
        #                             else:
        #                                 candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
        #                                 candidateMainGraphNodes.append(nodeSubgraphIdx)
        #                                 foundUnion = True
        #                         # Otherwise, go on with analysis
        #                         else:
        #                             candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[subgraphIdx].nodes[nodeSubgraphIdx]["pos"]], axis = 0)
        #                             candidateMainGraphNodes.append(nodeSubgraphIdx)
        #                             foundUnion = True
        #         # If none of the above have worked, pool all node coordinates from main subgraph
        #         if not foundUnion:
        #             for nodeMain in self.subgraphs[0]:
        #                 candidateMainGraphCoordinates = np.append(candidateMainGraphCoordinates, [self.subgraphs[0].nodes[nodeMain]["pos"]], axis = 0)
        #                 candidateMainGraphNodes.append(nodeMain)

        #         # Choose closest node from the candidate pool
        #         mainGraphNode = candidateMainGraphNodes[np.argmin(np.linalg.norm(candidateMainGraphCoordinates - self.subgraphs[idx].nodes[candidateNode]["pos"], axis = 1))]
        #         # Append node pairs altogether
        #         subgraphsUnionEdges.append([mainGraphNode, candidateNode])

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
                self.segmentsRadiusArray[candidateCellId] = np.insert(self.segmentsRadiusArray[candidateCellId], 0, self.centerlineGraph.nodes[mainGraphNode]["features"][6]) # Middle radius
            else:
                self.segmentsCoordinateArray[candidateCellId] = np.append(self.segmentsCoordinateArray[candidateCellId], [self.centerlineGraph.nodes[mainGraphNode]["pos"]], axis = 0)
                self.segmentsRadiusArray[candidateCellId] = np.append(self.segmentsRadiusArray[candidateCellId], self.centerlineGraph.nodes[mainGraphNode]["features"][6]) # Middle radius
            
            # We add the edge to the graph
            self.centerlineGraph.add_edge(mainGraphNode, candidateNode, cellId = self.centerlineGraph.nodes[candidateNode]["cellId"])
            self.centerlineGraph[mainGraphNode][candidateNode]["segmentsArrayPoints"] = np.array([])
            
            # Now, for the modification of the segments arrays and the cellIds and segmentsArrayPoints of nodes and edges, we perform an indepth analysis.
            # First of all, it only makes sense to split the segment if the node has degree 2 (otherwise it will already be a border between different segments)
            if self.centerlineGraph.degree(mainGraphNode) == 3:
                # The position of the mainGraphNode will be the division point between segments
                cutOffIdx = np.argmin(np.linalg.norm(self.segmentsCoordinateArray[self.centerlineGraph.nodes[mainGraphNode]["cellId"]] - self.centerlineGraph.nodes[mainGraphNode]["pos"], axis = 1))
                # Check if segment goes downstream with respect to hierarchy. If it is, go against hierarchy. If it is not (normal case), go with hierarchy
                downstream = False
                for neighbor in self.centerlineGraph.neighbors(mainGraphNode):
                    # If neighbor with higher hierarchy has smaller segmentsArrayPoints indices than cutOffIdx, the segment is downstream. Otherwise it is not
                    if self.centerlineGraph.nodes[mainGraphNode]["cellId"] == self.centerlineGraph.nodes[neighbor]["cellId"] and self.centerlineGraph.nodes[neighbor]["features"][0] > self.centerlineGraph.nodes[mainGraphNode]["features"][0] and np.mean(self.centerlineGraph[neighbor][mainGraphNode]["segmentsArrayPoints"]) < cutOffIdx:
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
                        if not downstream and self.centerlineGraph.nodes[neighbor]["cellId"] == self.centerlineGraph.nodes[mainGraphNode]["cellId"] and self.centerlineGraph.nodes[neighbor]["features"][0] > self.centerlineGraph.nodes[previousNode]["features"][0]:
                            # Update node cellId
                            self.centerlineGraph.nodes[neighbor]["cellId"] = nextCellId
                            # Update edge cellId
                            self.centerlineGraph[previousNode][neighbor]["cellId"] = nextCellId
                            # Update edge segmentsArrayPoints with cutOffIdx
                            self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"] = self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"] - cutOffIdx
                            # Eliminate negative edges if found (this)
                            while self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"][0] < 0 and len(self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"]) > 1:
                                self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"], 0)
                            # Update previousNode
                            previousNode = neighbor
                            # Check found neighbor
                            neighborFound = True
                        # If downstream and there is a node with the same cellId as the mainGraphNode and a lower hierarchy
                        elif downstream and self.centerlineGraph.nodes[neighbor]["cellId"] == self.centerlineGraph.nodes[mainGraphNode]["cellId"] and self.centerlineGraph.nodes[neighbor]["features"][0] < self.centerlineGraph.nodes[previousNode]["features"][0]:
                            # Update node cellId
                            self.centerlineGraph.nodes[neighbor]["cellId"] = nextCellId
                            # Update edge cellId
                            self.centerlineGraph[previousNode][neighbor]["cellId"] = nextCellId
                            # Update edge segmentsArrayPoints with cutOffIdx
                            self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"] = self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"] - cutOffIdx
                            while self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"][0] < 0 and len(self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"]) > 1:
                                self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"] = np.delete(self.centerlineGraph[previousNode][neighbor]["segmentsArrayPoints"], 0)
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


    def performFeatureExtraction(self):
        ''' Computes hierarchization and performs feature extraction for both accesses for the centerline graph.
        Featurizes self.centerlineGraph.

        Stores centerlineGraph.pickle. 
        
        '''
        # Hierarchization of self.centerlineGraph for femoral access (hierarchy femoral)
        self.centerlineGraph = getHierarchicalOrderingDense(self.centerlineGraph, access = "femoral", startNode = 0)
        # Featurizes self.centerlineGraph (features femoral)
        self.centerlineGraph = extractFeatures(self.caseDir, self.centerlineGraph, self.segmentsCoordinateArray, self.segmentsRadiusArray, "femoral", removeEdges = self.subgraphsUnionEdges)
        # Hierarchization of self.centerlineGraph for radial access (hierarchy radial)
        self.centerlineGraph = getHierarchicalOrderingDense(self.centerlineGraph, access = "radial", startNode = self.rightmostNode)
        # Featurizes self.centerlineGraph (features radial)
        self.centerlineGraph = extractFeatures(self.caseDir, self.centerlineGraph, self.segmentsCoordinateArray, self.segmentsRadiusArray, "radial", removeEdges = self.subgraphsUnionEdges)

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
                                if self.centerlineGraph[n0][n1]["cellId"] == supersegmentCandidateCellIdsA[-1] and len(self.centerlineGraph[n0][n1]["segmentsArrayPoints"]) != 0:
                                    distanceA += np.linalg.norm(self.centerlineGraph.nodes[n0]["pos"] - self.centerlineGraph.nodes[n1]["pos"])
                                elif self.centerlineGraph[n0][n1]["cellId"] == supersegmentCandidateCellIdsB[-1] and len(self.centerlineGraph[n0][n1]["segmentsArrayPoints"]) != 0:
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
            print(supersegmentCandidatesVesselTypes[access])
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
                    print(len(supersegmentCandidatesCellIds[access]), np.argmax(cosineSimilarities), len(bifurcatingSegmentsCandidatesCellIds[access]))
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
            # Specify the maximum length for a bifurcating segment
            limitBifurcationLength = 1000
            # Store positions from full main graph
            positionsFullGraph = []
            nodesFullGraph = []
            for node in self.centerlineGraph:
                positionsFullGraph.append(self.centerlineGraph.nodes[node]["pos"])
                nodesFullGraph.append(node)
            # Extract sequences for both accesses
            for access in self.accesses:
                # Initiaize supersegment list for both accesses
                self.supersegments[access] = []
                # Build supersegment for each of the existing reference configurations
                for configIdx, predictedConfiguration in enumerate(self.predictedConfigurations[access]):
                    _, _, _, configurationName = self.configurations[access][configIdx]
                    # Get predicted confirguration
                    supersegmentSegments, bifurcatingSegments = predictedConfiguration
                    # Initialize a graph with networkx for each supersegment 
                    supersegment = None
                    supersegment = nx.Graph()
                    # We store all positions of the supersegment segments to orient all bifurcating segments 
                    positionsMainSupersegment = np.ndarray([0, 3])
                    # Only taking first and last positions of the curves arrays
                    totalNodes = 0 # We only link nodes from the same centerline
                    # We alse reinitialize the accumulated distance and update the prior node previousIdx
                    for cellId, curve in enumerate(self.segmentsCoordinateArray):
                        # Include only those nodes with cellId in the supersegment sequence
                        if cellId in supersegmentSegments:
                            # Initialize distance for node sampling
                            distance = 0
                            # We loop to every centerline point in each segmentsArray cell
                            for idx, position in enumerate(curve):
                                # For the first node in every cell, we add just a node with its corresponding cellId
                                if idx == 0:
                                    supersegment.add_node(totalNodes, pos=position)
                                    supersegment.nodes[totalNodes]["cellId"] = cellId
                                    supersegment.nodes[totalNodes]["isSupersegment"] = True
                                    totalNodes += 1
                                    previousPosition = position
                                # If distance from last sampled node is larger than selected sampleNodeEvery, add a node with cellId and an edge to the previous sample 
                                # node, keeping cellId and the indices of the centerline points in the segmentsArray
                                elif distance > self.sampleNodeEvery:
                                    supersegment.add_node(totalNodes, pos=position)
                                    supersegment.nodes[totalNodes]["cellId"] = cellId
                                    supersegment.nodes[totalNodes]["isSupersegment"] = True
                                    supersegment.add_edge(totalNodes - 1, totalNodes)
                                    supersegment[totalNodes - 1][totalNodes]["cellId"] = cellId
                                    positionsMainSupersegment = np.append(positionsMainSupersegment, [position], axis = 0)
                                    totalNodes += 1
                                    previousPosition = position
                                    distance = 0
                                # For the last point of every centerline cell (presumably at a distance smaller than sampleNodeEvery) we differentiate between two possible cases
                                elif idx == len(curve) - 1:
                                    # If the number of nodes of the previous node of the cell is 0 (which means that the segment's length is smaller than sampleNodeEvery), we add an additional node
                                    if len([n for n in supersegment.neighbors(totalNodes - 1)]) == 0 or np.linalg.norm(supersegment.nodes[totalNodes - 1]["pos"] - curve[idx]) > 10:
                                        supersegment.add_node(totalNodes, pos=position)
                                        supersegment.nodes[totalNodes]["cellId"] = cellId
                                        supersegment.nodes[totalNodes]["isSupersegment"] = True
                                        supersegment.add_edge(totalNodes - 1, totalNodes)
                                        supersegment[totalNodes - 1][totalNodes]["cellId"] = cellId
                                        positionsMainSupersegment = np.append(positionsMainSupersegment, [position], axis = 0)
                                        totalNodes += 1
                                    # If it is not, which will be the general case, we do not add a new node, but instead we transform the last added node and change its position to be placed at the bifurcation/endpoint
                                    else:
                                        supersegment.nodes[totalNodes - 1]["pos"]= position
                                        supersegment.nodes[totalNodes - 1]["cellId"] = cellId
                                        supersegment[totalNodes - 2][totalNodes - 1]["cellId"] = cellId
                                        positionsMainSupersegment[-1, :] = position
                                # If the accumulated distance from the last sample node is smaller than sampleNodeEvery, and we are not in either the first or last nodes of the segmentsArray cell, just update the distance
                                else:
                                    distance += np.linalg.norm(previousPosition - position)
                    # Once the main supersegment is built, we add the bifurcating segments
                    for cellId, curve in enumerate(self.segmentsCoordinateArray):
                        # Include only those nodes with cellId in the bifurcating segment sequence
                        if cellId in bifurcatingSegments:
                            # Initialize distance for node sampling
                            distance = 0
                            # Initialize accumulatedDistance for bifurcation limitation
                            accumulatedDistance = 0
                            # Check if bifurcating segment needs flipping (if starting point is in contact with supersegment or not)
                            if not np.amin(np.linalg.norm(np.expand_dims(curve[0], 0) - positionsMainSupersegment, axis = 1)) < 1 and np.amin(np.linalg.norm(np.expand_dims(curve[0], 0) - positionsMainSupersegment, axis = 1)) > np.amin(np.linalg.norm(np.expand_dims(curve[-1], 0) - positionsMainSupersegment, axis = 1)):
                                curve = np.flip(curve, axis=0)
                            # We loop to every centerline point in each segmentsArray cell
                            for idx, position in enumerate(curve):
                                # Check if accumulatedDistance is over the limit
                                if accumulatedDistance < limitBifurcationLength:
                                    # For the first node in every cell, we add just a node with its corresponding cellId
                                    if idx == 0:
                                        supersegment.add_node(totalNodes, pos=position)
                                        supersegment.nodes[totalNodes]["cellId"] = cellId
                                        supersegment.nodes[totalNodes]["isSupersegment"] = False
                                        totalNodes += 1
                                        previousPosition = position
                                    # If distance from last sampled node is larger than selected sampleNodeEvery, add a node with cellId and an edge to the previous sample 
                                    # node, keeping cellId and the indices of the centerline points in the segmentsArray
                                    elif distance > self.sampleNodeEvery:
                                        supersegment.add_node(totalNodes, pos=position)
                                        supersegment.nodes[totalNodes]["cellId"] = cellId
                                        supersegment.nodes[totalNodes]["isSupersegment"] = False
                                        supersegment.add_edge(totalNodes - 1, totalNodes)
                                        supersegment.add_edge(totalNodes - 1, totalNodes)
                                        supersegment[totalNodes - 1][totalNodes]["cellId"] = cellId
                                        totalNodes += 1
                                        accumulatedDistance += np.linalg.norm(previousPosition - position)
                                        previousPosition = position
                                        distance = 0
                                    # For the last point of every centerline cell (presumably at a distance smaller than sampleNodeEvery) we differentiate between two possible cases
                                    elif idx == len(curve) - 1:
                                        if len([n for n in supersegment.neighbors(totalNodes - 1)]) == 0 or np.linalg.norm(supersegment.nodes[totalNodes - 1]["pos"] - curve[idx]) > 10:
                                            supersegment.add_node(totalNodes, pos=position)
                                            supersegment.nodes[totalNodes]["cellId"] = cellId
                                            supersegment.nodes[totalNodes]["isSupersegment"] = False
                                            supersegment.add_edge(totalNodes - 1, totalNodes)
                                            supersegment[totalNodes - 1][totalNodes]["cellId"] = cellId
                                            totalNodes += 1
                                        # If the number of nodes of the previous node of the cell is 0 (which means that the segment's length is smaller than sampleNodeEvery), we add an additional node
                                        # If it is not, which will be the general case, we do not add a new node, but instead we transform the last added node and change its position to be placed at the bifurcation/endpoint
                                        else:
                                            supersegment.nodes[totalNodes - 1]["pos"]= position
                                            supersegment.nodes[totalNodes - 1]["cellId"] = cellId
                                            supersegment[totalNodes - 2][totalNodes - 1]["cellId"] = cellId
                                    # If the distance from the last sample node is smaller than sampleNodeEvery, and we are not in either the first or last nodes of the segmentsArray cell, just update the distance
                                    else:
                                        distance += np.linalg.norm(previousPosition - position)
                                        accumulatedDistance += distance

                    # Merge nodes that share the same RAS coordinate (bifurcation spots)
                    # First get all nodes that have a degree of 1 (start- and endpoints)
                    deg1Nodes = []
                    for node, deg in supersegment.degree:
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
                                C1 = supersegment.nodes[node]["pos"]
                                C2 = supersegment.nodes[node2]["pos"]
                                if C1[0] == C2[0] and C1[1] == C2[1] and C1[2] == C2[2]:
                                    supersegment = nx.contracted_nodes(supersegment, node, node2)
                                    removedNodes.append(node2)
                                    supersegment.nodes[node].pop("contraction")

                    # Relabel nodes as sequential labels
                    mapping = {}
                    newNode = 0
                    for oldNode in supersegment.nodes():
                        mapping[oldNode] = newNode
                        newNode += 1
                    supersegment = nx.relabel.relabel_nodes(supersegment, mapping)
                    
                    # Transfer hierarchy and features from the corresponding access to the final supersegments
                    for node in supersegment:
                        for nodeFullGraph in self.centerlineGraph:
                            nodeFullGraph = nodesFullGraph[np.argmin(np.linalg.norm(supersegment.nodes[node]["pos"] - positionsFullGraph, axis = 1))]
                            supersegment.nodes[node]["pos"] = self.centerlineGraph.nodes[nodeFullGraph]["pos"]
                            supersegment.nodes[node]["hierarchy"] = self.centerlineGraph.nodes[nodeFullGraph][f"hierarchy {access}"]
                            supersegment.nodes[node]["features"] = self.centerlineGraph.nodes[nodeFullGraph][f"features {access}"]

                    for node in supersegment:
                        if "hierarchy" not in supersegment.nodes[node].keys():
                            print(node, supersegment.nodes[node])

                    # Add to the supersegments dict
                    self.supersegments[access].append(supersegment)
                    # Save supersegment as pickle
                    nx.readwrite.gpickle.write_gpickle(supersegment, os.path.join(self.caseDir, "supersegments", f"{configurationName}.pickle"), protocol = 4)
            # Make plot with all supersegments
            makeSupersegmentPlots(self.caseDir, self.supersegments)

        def selectConfiguration(self):
            ''' Selects supersegment configuration if patientConfiguration.json is present in self.caseDir.

            Creats new dir self.caseDir/thrombectomyConfiguration, and stores selected supersegment and image.
            
            '''

            with open(os.path.join(self.caseDir, "patientConfiguration.json")) as jsonFile:
                self.patientConfiguration = json.load(jsonFile)[self.caseId]

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

                for supersegment in [supersegment for supersegment in os.listdir(os.path.join(self.caseDir, "supersegments")) if supersegment.endswith(".pickle") and supersegment.startswith(str(configurationId))]:
                    print("         Selecting supersegment:", supersegment)
                    shutil.copyfile(os.path.join(self.caseDir, "supersegments", supersegment), os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.pickle"))
                    makeSupersegmentPlot(self.caseDir, nx.readwrite.gpickle.read_gpickle(os.path.join(self.caseDir, "thrombectomyConfiguration", "supersegment.pickle")), self.patientConfiguration)

            else:
                print("        Laterality is ambiguous:", self.patientConfiguration["Laterality"])
        
        # Perform both methods to perform supersegment extraction
        supersegmentPrediction(self)
        supersegmentBuilt(self)
        selectConfiguration(self)