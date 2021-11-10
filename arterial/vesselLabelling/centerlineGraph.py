import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generateCenterlineGraph(segmentsArray, caseDir, make_plot=False):
    ''' Inputs segmentsArray and generates a networkx graph with the features described by 
    Chen et al. [2020] ("Automated Intracranial Artery Labeling using a Graph Neural Network 
    and Hierarchical Refinement"). 

    Saves the graph as a .pickle file.

    Arguments:
        - segmentsArray <numpy array>: numpy array containing the position of the centerline points, 
        as well as the associated radius to each point, and split into individual segments between 
        bifurcations.
        - caseDir <str>: path to the directory containing the binary mask. All 
        segmentations will be saved in this dir.
        - make_plot <bool>: if True, this will display the centerline segments in a 3D plot.
        Set to False if omitted.

    Returns:
        - G <networkx graph>: networkx graph derived from segmentsArray.

    '''
    segmentsCoordinateArray = segmentsArray[:, 0]
    segmentsRadiusArray = segmentsArray[:, 1]

    # Initialize a graph with networkx
    G = 0
    G = nx.Graph()

    # Only taking first and last positions of the curves arrays
    total_nodes = 0 # We only link nodes from the same centerline
    for cellID, curve in enumerate(segmentsCoordinateArray):
        if len(curve) > 1 and len(segmentsRadiusArray[cellID]) > 1:
            # Add nodes
            # First node of cell (startpoint)
            G.add_node(total_nodes + 0, pos=curve[0])
            G.nodes[total_nodes + 0]["deg"] = G.degree[total_nodes + 0]
            G.nodes[total_nodes + 0]["rad"] = segmentsRadiusArray[cellID][0]
            # Build features array
            G.nodes[total_nodes + 0]["features"] = np.array([curve[0][0], curve[0][1], curve[0][2], segmentsRadiusArray[cellID][0], G.degree[total_nodes + 0]])

            # Last node of cell (endpoint)
            G.add_node(total_nodes + 1, pos=curve[-1])
            G.nodes[total_nodes + 1]["deg"] = G.degree[total_nodes + 1]
            G.nodes[total_nodes + 1]["rad"] = segmentsRadiusArray[cellID][-1]
            # Build node features array
            G.nodes[total_nodes + 1]["features"] = np.array([curve[-1][0], curve[-1][1], curve[-1][2], segmentsRadiusArray[cellID][-1], G.degree[total_nodes + 1]])

            # Add edges
            G.add_edge(total_nodes, total_nodes + 1, CellID = cellID)
            distance = np.linalg.norm(curve[-1] - curve[0])
            direction = (curve[-1] - curve[0]) / distance
            G[total_nodes][total_nodes + 1]["mean rad"] = np.mean(segmentsRadiusArray[cellID])
            G[total_nodes][total_nodes + 1]["proximal radius"] = segmentsRadiusArray[cellID][0]
            G[total_nodes][total_nodes + 1]["distal radius"] = segmentsRadiusArray[cellID][-1]
            G[total_nodes][total_nodes + 1]["proximal/distal radius ratio"] = segmentsRadiusArray[cellID][0] / segmentsRadiusArray[cellID][-1]
            G[total_nodes][total_nodes + 1]["minimum radius"] = np.amin(segmentsRadiusArray[cellID])
            G[total_nodes][total_nodes + 1]["maximum radius"] = np.amax(segmentsRadiusArray[cellID])
            G[total_nodes][total_nodes + 1]["distance"] = distance
            G[total_nodes][total_nodes + 1]["relative length"] = relativeLength(segmentsCoordinateArray[cellID])
            G[total_nodes][total_nodes + 1]["direction"] = direction
            G[total_nodes][total_nodes + 1]["departure angle"] = (segmentsCoordinateArray[cellID][1] - segmentsCoordinateArray[cellID][0]) / np.linalg.norm(segmentsCoordinateArray[cellID][1] - segmentsCoordinateArray[cellID][0])
            G[total_nodes][total_nodes + 1]["number of points"] = len(segmentsCoordinateArray[cellID])
            G[total_nodes][total_nodes + 1]["proximal bifurcation position"] = segmentsCoordinateArray[cellID][0]
            G[total_nodes][total_nodes + 1]["distal bifurcation position"] = segmentsCoordinateArray[cellID][-1]
            G[total_nodes][total_nodes + 1]["center of mass"] = np.sum(segmentsCoordinateArray[cellID], axis = 0) / len(segmentsCoordinateArray[cellID])
            # Build edge feature array
            G[total_nodes][total_nodes + 1]["features"] = np.array([G[total_nodes][total_nodes + 1]["mean rad"], 
                                                                    G[total_nodes][total_nodes + 1]["proximal radius"],
                                                                    G[total_nodes][total_nodes + 1]["distal radius"],
                                                                    G[total_nodes][total_nodes + 1]["proximal/distal radius ratio"],
                                                                    G[total_nodes][total_nodes + 1]["minimum radius"],
                                                                    G[total_nodes][total_nodes + 1]["maximum radius"],
                                                                    G[total_nodes][total_nodes + 1]["distance"],
                                                                    G[total_nodes][total_nodes + 1]["relative length"],
                                                                    G[total_nodes][total_nodes + 1]["direction"][0],
                                                                    G[total_nodes][total_nodes + 1]["direction"][1],
                                                                    G[total_nodes][total_nodes + 1]["direction"][2],
                                                                    G[total_nodes][total_nodes + 1]["departure angle"][0],
                                                                    G[total_nodes][total_nodes + 1]["departure angle"][1],
                                                                    G[total_nodes][total_nodes + 1]["departure angle"][2],
                                                                    G[total_nodes][total_nodes + 1]["number of points"],
                                                                    G[total_nodes][total_nodes + 1]["proximal bifurcation position"][0],
                                                                    G[total_nodes][total_nodes + 1]["proximal bifurcation position"][1],
                                                                    G[total_nodes][total_nodes + 1]["proximal bifurcation position"][2],
                                                                    G[total_nodes][total_nodes + 1]["distal bifurcation position"][0],
                                                                    G[total_nodes][total_nodes + 1]["distal bifurcation position"][1],
                                                                    G[total_nodes][total_nodes + 1]["distal bifurcation position"][2],
                                                                    G[total_nodes][total_nodes + 1]["center of mass"][0],
                                                                    G[total_nodes][total_nodes + 1]["center of mass"][1],
                                                                    G[total_nodes][total_nodes + 1]["center of mass"][2]
            ])
            total_nodes += 2

    # Merge nodes that share the same RAS coordinate (bifurcation spots)
    # First get all nodes that have a degree of 1 (start- and endpoints)
    deg1Nodes = []
    for node, deg in G.degree:
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
                C1 = G.nodes(data=True)[node]["pos"]
                C2 = G.nodes(data=True)[node2]["pos"]
                if C1[0] == C2[0] and C1[1] == C2[1] and C1[2] == C2[2]:
                    G = nx.contracted_nodes(G, node, node2)
                    removedNodes.append(node2)
                    G.nodes(data=True)[node].pop("contraction")

    # Relabel nodes as sequential labels
    mapping = {}
    new_node = 0
    for old_node in G.nodes():
        mapping[old_node] = new_node
        new_node += 1
    G = nx.relabel.relabel_nodes(G, mapping)

    # Directional embeddings for node features have to be computed after all edge directions for the whole graph are computed
    for node in G.nodes:
        projectedMajorDirections = directionalEmbeddings(G, node)
        G.nodes(data=True)[node]["dir"] = projectedMajorDirections
        G.nodes(data=True)[node]["features"] = np.append(G.nodes(data=True)[node]["features"], projectedMajorDirections)

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in G.nodes():
        node_pos_dict_P[n] = [G.nodes(data=True)[n]["pos"][0], G.nodes(data=True)[n]["pos"][2]]

    # For a coronal view, we use P and S coordinates (the view will be from the coronal plane, R axis)
    node_pos_dict_R = {}
    for n in G.nodes():
        node_pos_dict_R[n] = [G.nodes(data=True)[n]["pos"][1], G.nodes(data=True)[n]["pos"][2]]

    nx.readwrite.gpickle.write_gpickle(G, os.path.join(caseDir, "graph.pickle"))

    if make_plot:
        edge_labels = nx.get_edge_attributes(G,'CellID')

        nx.draw(G, node_pos_dict_P, node_size=20)
        nx.draw_networkx_edge_labels(G, node_pos_dict_P, edge_labels = edge_labels)

        # nx.draw(G, node_pos_dict_R, node_size=20)
        # nx.draw_networkx_edge_labels(G, node_pos_dict_R, edge_labels = edge_labels)

        plt.savefig(os.path.join(caseDir, "graph.png"))
        plt.close()
        # plt.show()

    # return G


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