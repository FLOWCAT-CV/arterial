import os
import networkx as nx


import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToDevice, BaseTransform

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

def nodeTransform(G):
    ''' Applies node transform to networkx graph.

    Arguments:
        - G <networkx.graph>: graph in edge form.
    
    Returns:
        - Gnew <networkx.graph>: graph in node form.
        
    '''

    newNodes = [] # list with new nodes
    edges2nodes = {} # dict linking former edges to new nodes (undirected)
    newNodes2OldEdges = {} # we wil use this dict to track edge attributes back
    newNode = 0 # auxiliary

    # We store a new node for each edge, and create a dict to pass the edge attributes from
    # the old graph to the nodes of the new graph
    for edge in G.edges:
        newNodes.append(newNode)
        edges2nodes[edge] = newNode
        edges2nodes[(edge[1], edge[0])] = newNode
        newNodes2OldEdges[newNode] = edge
        newNode += 1

    newEdges = [] # list with new edges
    
    # We define edges of the new graph as links between immediately neighbouring vessels
    for node in G.nodes:
        if len(G.edges(node)) > 1:
            # For each node connected to multiple edges, we create an auxiliary list with connected nodes
            edgeListIdx = [edges2nodes[edge] for edge in G.edges(node)]
            # We iterate over all nodes except for the last one to connect them once
            for idx, src in enumerate(edgeListIdx):
                for dst in edgeListIdx[idx + 1:]:
                    newEdges.append([src, dst])

    # We create a new empty graph
    Gnew = nx.Graph()

    # We add nodes, node attributes (former edge attributes) and edges
    for node in newNodes:
        Gnew.add_node(node)
        for attributeKey in G[newNodes2OldEdges[node][0]][newNodes2OldEdges[node][1]].keys():
            Gnew.nodes[node][attributeKey] = G[newNodes2OldEdges[node][0]][newNodes2OldEdges[node][1]][attributeKey]
    for edge in newEdges:
        Gnew.add_edge(edge[0], edge[1])

    return Gnew


def performInferenceGraphUNet(model, nodeFormGraphNx):
    ''' Integrates the predicted vessel types from the inference of the 
    graph U-Net over the nodie form graph on the edge form graph. Saves 
    the predicted final graph as graph_pred.pickle and a png file for quick 
    visualization.

    Arguments:
        - model <PyTorch model>: trained graph U-Net model.
        - nodeFormGraphNx <networkx.graph>: graph in node form.
    
    Returns:
        - cellIDToVesselType <dict>: dictionary with CellIDs as keys and 
        predicted vessel types as values.
    
    '''

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define inference transforms
    inferenceTransforms = Compose([
            RadiusGraph(10, loop=False),
            CustomNormalizeFeatures(),
            ToDevice(device)
        ])
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    inferenceData = ArterialDatasetInference(graph=nodeFormGraphNx, transforms=inferenceTransforms)
    inferenceDataLoader = DataLoader(inferenceData, batch_size=1, shuffle=False) # We need this to perform a preprocessing of the data (transforms)
    # Load model to device
    model = model.to(device)
    model.eval()
    # We have to iterate over the DataLoader (even thogh it will just be one graph at the time)
    for graph in inferenceDataLoader:
        # Inference returns a tensor with the softmax probabilities for the vessel type for each node
        # We perform argmax to obtain the vessel type with the highest probability and pass it to list
        # The result is a 1D list with the predicted vessel types for each node
        predictedNodes = model(graph.x, graph.edge_index).argmax(dim=1).tolist()

    # We create a dict to link the CellIDs from the segmentaArray to the predicted vessel types
    cellIDToVesselType = {}
    for idx, cellID in enumerate(inferenceData.databasePyg[0].cellIDs):
        cellIDToVesselType[cellID] = predictedNodes[idx]
    
    return cellIDToVesselType


def savePredictedGraph(caseDir, edgeFormGraphNx, predictedVessels):
    ''' Integrates the predicted vessel types from the inference of the 
    graph U-Net over the nodie form graph on the edge form graph. Saves 
    the predicted final graph as graph_pred.pickle and a png file for quick 
    visualization.

    Arguments:
        - caseDir <str or path>: path to the caseDir of the case.
        - edgeFormGraphNx <networkx.graph>: graph in edge form.
        - predictedVessels <dict>: dictionary with CellIDs as keys and 
        predicted vessel types as values.
    
    Returns:
    
    '''
    # Define new graph with same nodes and edges and information from the input graph
    predictedGraph = nx.Graph()
    # For nodes, we only keep the position of the original edge form graph nodes. We use these for visualization in the png file
    node_pos_dict_P = {}
    for node in edgeFormGraphNx.nodes:
        predictedGraph.add_node(node)
        predictedGraph.nodes(data=True)[node]["pos"] = edgeFormGraphNx.nodes(data=True)[node]["pos"]
        # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates 
        # (the view will be from the coronal plane, P axis)
        node_pos_dict_P[node] = [edgeFormGraphNx.nodes(data=True)[node]["pos"][0], edgeFormGraphNx.nodes(data=True)[node]["pos"][2]]

    # For edges, we keep all information from the original graph, and in addition we set the vessel type from the predictedVessels dict
    for n0, n1 in edgeFormGraphNx.edges:
        predictedGraph.add_edge(n0, n1)
        predictedGraph[n0][n1]["CellID"] = edgeFormGraphNx[n0][n1]["CellID"]
        predictedGraph[n0][n1]["edgetype"] = predictedVessels[edgeFormGraphNx[n0][n1]["CellID"]]
        predictedGraph[n0][n1]["edgeTypeName"] = edgeTypes[predictedVessels[edgeFormGraphNx[n0][n1]["CellID"]]]
        predictedGraph[n0][n1]["features"] = edgeFormGraphNx[n0][n1]["features"]

    # We save the graph
    nx.write_gpickle(predictedGraph, os.path.join(caseDir, "graph_pred.pickle"))
    # Set the edge labels for visualization in the png file
    edge_labels = nx.get_edge_attributes(predictedGraph, 'edgeTypeName')
    # We draw the png file with the predicted vessel types
    nx.draw(edgeFormGraphNx, node_pos_dict_P, node_size=20)
    nx.draw_networkx_edge_labels(edgeFormGraphNx, node_pos_dict_P, edge_labels = edge_labels)
    plt.savefig(os.path.join(caseDir, "graph_pred.png"))


class ArterialDatasetInference(InMemoryDataset):
    ''' This class is used to load the networkx graph to the suitable form
    for PyTorch Geometric models. Each graph contains the follwoing attributes:
        - pos <torch.tensor of shape [3, num_nodes]>: contains position of nodes.
        - x <torch.tensor of shape [num_nodes, num_attributes]>: contains node attributes for each node.
        - edge_index <torch.tensor of shape [2, num_edges]>: contains edges in the form of connected node pairs.
        - cellIDs <list>: cellIDs from the original graph (from segmentsArray).

    Arguments:
        - graph <networkx.graph>: graph in node form.
        - transforms <PyTorch transform>: composed PyTorch transforms for the graph.
        These define the preprocessing of the graph before inference.
    
    Returns (__getitem__()):
        - data <torch.data.Data()>: data object containing the graph information in 
        PyTorch Geometric form.
    
    '''
    def __init__(self, 
                 graph,
                 transforms=None):
        super().__init__(graph, transforms)
        self.databaseNx = [graph]
        self.databasePyg = []
        for graphNx in self.databaseNx:
            graphPyg = Data()
            pos, x, cellIDs, edge_index = [], [], [], []
            for node in graphNx.nodes:
                pos.append(graphNx.nodes[node]["center of mass"])
                x.append(graphNx.nodes[node]["features"])
                cellIDs.append(graphNx.nodes[node]["CellID"])
            for n0, n1 in graphNx.edges:
                edge_index.append([n0, n1])
            graphPyg.pos = torch.tensor(pos, dtype=torch.float32)
            graphPyg.x = torch.tensor(x, dtype=torch.float32)
            graphPyg.edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.int64), 1, 0)
            graphPyg.cellIDs = cellIDs
            graphPyg.num_nodes = len(graphNx.nodes)
            graphPyg.num_edges = len(graphNx.edges)
            self.databasePyg.append(graphPyg)
        self.numNodeFeatures = graphPyg.x.shape[1]
        self.numNodeClasses = 16
        self.transforms = transforms
        
    def __len__(self) -> int:
        return len(self.databasePyg)

    def __getitem__(self, idx):
        data = self.databasePyg[idx]
        if self.transforms is not None:
            data = self.transforms(data)
            
        return data


class CustomNormalizeFeatures(BaseTransform):
    """Row-normalizes data.x (custom to our case). Normalization constants are
    chosen as the average values for each attribute across the training dataset.
    Some exceptions apply:
        - Relative atributes (e.g., proximal/distal radius ratio, relative length): not normalized.
        - Direction vectors (e.g., direction and departure angle): normalized to norm = 1.

    Arguments:
    
    Returns (__getitem__()):
        - newData <torch.data.Data()>: data object containing the normalized node attributes (x) in 
        PyTorch Geometric form.
        
    """
    def __init__(self):
        # These normalization constants are extracted from the average values of each attribute accross
        # the training dataset
        self.normalizationConstants = torch.tensor([5.273747439888846422e+00, # mean rad
                                                    7.303014327468136280e+00, # proximal radius
                                                    4.322198077560408080e+00, # distal radius
                                                    1.000000000000000000e+00, # proximal/distal radius ratio
                                                    3.727253986260447238e+00, # minimum radius
                                                    7.909319265201797400e+00, # maximum radius
                                                    1.467396407007539381e+02, # distance
                                                    1.000000000000000000e+00, # relative length
                                                    1.000000000000000000e+00, # direction 0
                                                    1.000000000000000000e+00, # direction 1
                                                    1.000000000000000000e+00, # direction 2
                                                    1.000000000000000000e+00, # departure angle 0
                                                    1.000000000000000000e+00, # departure angle 1
                                                    1.000000000000000000e+00, # departure angle 2
                                                    3.407976539589442950e+02, # number of points
                                                    2.582839364498616987e+02, # proximal bifurcation position 0
                                                    2.294162140744778640e+02, # proximal bifurcation position 1
                                                    2.139304154958557262e+02, # proximal bifurcation position 2
                                                    2.621296421762096429e+02, # distal bifurcation position 0
                                                    2.301304323856222709e+02, # distal bifurcation position 1
                                                    3.054477580255216367e+02, # distal bifurcation position 2
                                                    2.606832670113914219e+02, # center of mass 0
                                                    2.257899084802627954e+02, # center of mass 1
                                                    2.672428365375042176e+02] # center of mass 2
                                                    , dtype=torch.float32)

    def __call__(self, data):
        newData = data.__copy__()
        newData.x[..., 8:11] = F.normalize(data.x[..., 8:11], 1)
        newData.x[..., 11:14] = F.normalize(data.x[..., 11:14], 1)
        newData.x[..., 21:] = data.pos[...]
        newData.x = data.x / self.normalizationConstants
        return newData