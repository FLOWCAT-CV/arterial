#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np
import networkx as nx

import pickle

import matplotlib.pyplot as plt

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToDevice, BaseTransform

def node_transform(graph):
    """
    Applies node transform to networkx graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph in original form, where edges encode vessels.
    
    Returns
    -------
    transformed_graph : networkx.Graph
        Graph in node form, where nodes encode vessels.
        
    """
    # Declare empty necessary lists and dicts
    new_nodes = [] # list with new nodes
    edges_to_nodes = {} # dict linking former edges to new nodes (undirected)
    new_nodes_to_old_edges = {} # we wil use this dict to track edge attributes back
    new_node = 0 # auxiliary

    # We store a new node for each edge, and create a dict to pass the edge attributes from
    # the old graph to the nodes of the new graph
    for edge in graph.edges:
        new_nodes.append(new_node)
        edges_to_nodes[edge] = new_node
        edges_to_nodes[(edge[1], edge[0])] = new_node
        new_nodes_to_old_edges[new_node] = edge
        new_node += 1

    new_edges = [] # list with new edges
    
    # We define edges of the new graph as links between immediately neighbouring vessels
    for node in graph.nodes:
        if len(graph.edges(node)) > 1:
            # For each node connected to multiple edges, we create an auxiliary list with connected nodes
            edge_list_aux = [edges_to_nodes[edge] for edge in graph.edges(node)]
            # We iterate over all nodes except for the last one to connect them once
            for idx, src in enumerate(edge_list_aux):
                for dst in edge_list_aux[idx + 1:]:
                    new_edges.append([src, dst])

    # We create a new empty graph
    transformed_graph = nx.Graph()

    # We add nodes, node attributes (former edge attributes) and edges
    for node in new_nodes:
        transformed_graph.add_node(node)
        # Also tranfer the edge attributes to node attributes
        for attribute_key in graph[new_nodes_to_old_edges[node][0]][new_nodes_to_old_edges[node][1]].keys():
            transformed_graph.nodes[node][attribute_key] = graph[new_nodes_to_old_edges[node][0]][new_nodes_to_old_edges[node][1]][attribute_key]
     
    # Finally add edges       
    for edge in new_edges:
        transformed_graph.add_edge(edge[0], edge[1])

    return transformed_graph

def predict_vessel_types(model, tranformed_graph):
    """ 
    Performs inference over the node form graph with the trained graqh U-Net
    model. As output, it creates a dictionary linking all cell_id indices from
    the centerline_segments_array to the predicted vessel types.

    Parameters
    ----------
    model : torch_geometric.nn.models.graph_unet_GraphUNet object
        Trained graph U-Net node classification model.
    tranformed_graph : networkx.Graph
        Graph in node form, where nodes encode vessels.
    
    Returns
    -------
    predicted_vessels : dict
        Dictionary with cell_ids as keys and predicted vessel types as values.
    
    """
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define inference transforms
    inference_transforms = Compose([
            RadiusGraph(10, loop=False),
            CustomNormalizeFeatures(),
            ToDevice(device)
        ])
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    inference_data = ArterialDatasetInference(graph = tranformed_graph, transforms = inference_transforms)
    inference_data_loader = DataLoader(inference_data, batch_size = 1, shuffle = False) # We need this to perform a preprocessing of the data (transforms)
    # Load model to device
    model = model.to(device)
    model.eval()
    # We have to iterate over the DataLoader (even thogh it will just be one graph at the time)
    for graph in inference_data_loader:
        # Inference returns a tensor with the softmax probabilities for the vessel type for each node
        # We perform argmax to obtain the vessel type with the highest probability and pass it to list
        # The result is a 1D list with the predicted vessel types for each node
        predicted_nodes = model(graph.x, graph.edge_index).argmax(dim=1).tolist()

    # We create a dict to link the cell_ids from the centerline_segments_array to the predicted vessel types
    predicted_vessels = {}
    for idx, cell_id in enumerate(inference_data.database_pyg[0].cell_ids):
        predicted_vessels[cell_id] = predicted_nodes[idx]
    
    return predicted_vessels

def save_predicted_graph(case_dir, graph, predicted_vessels):
    """ 
    Integrates the predicted vessel types from the inference of the 
    graph U-Net over the node form graph on the original graph. 

    Saves graph and image with predicted vessel types as:

    >>> case_dir/graph_pred.pickle
    >>> case_dir/graph_pred.png

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    graph : networkx.Graph
        Graph in original form, where edges encode vessels.
    predicted_vessels : dict
        Dictionary with cell_ids as keys and predicted vessel types as values.
    
    Returns
    -------
    
    """
    # Define edge_types dict
    edge_types = dict(zip([idx for idx in range(14)], 
                          ["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"]))
    # Define new graph with same nodes and edges and information from the input graph
    predicted_graph = nx.Graph()
    # For nodes, we only keep the position of the original edge form graph nodes. We use these for visualization in the png file
    node_pos_dict_p = {}
    for node in graph.nodes:
        predicted_graph.add_node(node)
        predicted_graph.nodes(data=True)[node]["pos"] = graph.nodes(data=True)[node]["pos"]
        # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates 
        # (the view will be from the coronal plane, P axis)
        node_pos_dict_p[node] = [graph.nodes(data=True)[node]["pos"][0], graph.nodes(data=True)[node]["pos"][2]]

    # For edges, we keep all information from the original graph, and in addition we set the vessel type from the predicted_vessels dict
    for src, dst in graph.edges:
        predicted_graph.add_edge(src, dst)
        predicted_graph[src][dst]["cell_id"] = graph[src][dst]["cell_id"]
        predicted_graph[src][dst]["vessel type"] = predicted_vessels[graph[src][dst]["cell_id"]]
        predicted_graph[src][dst]["vessel type name"] = edge_types[predicted_vessels[graph[src][dst]["cell_id"]]]
        predicted_graph[src][dst]["features"] = graph[src][dst]["features"]

    # Save the graph and image for quick visualization
    with open(os.path.join(case_dir, "graph_pred.pickle"), "wb") as f:
        pickle.dump(predicted_graph, f, protocol = 4)
    make_graph_plot(case_dir, predicted_graph, "graph_pred.png", label = "vessel type name")

class ArterialDatasetInference(InMemoryDataset):
    """
    This class is used to load the networkx graph to the suitable form
    for PyTorch Geometric models. Each graph will contain the following attributes:
        - pos : torch.tensor, shape 3 x num_nodes
            Contains position of nodes in ijk coordinates.
        - x  : torch.tensor, shape num_nodes x num_attributes
            Contains node attributes for each node.
        - edge_index : torch.tensor, shape 2 x num_edges
            Contains edges in the form of connected node pairs.
        - cell_ids : list
            List with cell_ids from the original graph (from centerline_segments_array).
        - num_nodes : integer
            Number of nodes in the graph.
        - num_edges : integer
            Number of edges in the graph.

    """
    def __init__(self, graph, transforms = None):
        """
        Initalizes an object of the ArterialDatasetInference class.

        Parameters
        ----------
        graph : networkx.Graph
            Graph in original form, where edges encode vessels.
        transforms : torch_geometric.transforms.compose.Compose object
            Composed transforms applied on the graph prior to inference.

        Returns
        -------
        
        """
        # Initialize inherited class (InMemoryDataset)
        super().__init__(graph, transforms)
        # We treat it as if it was a list of graphs (but for inference it will only be one graph always)
        self.database_nx = [graph]
        self.database_pyg = [] # Here we will store all pyg graphs
        # 
        for graph_nx in self.database_nx:
            graph_pyg = Data()
            pos, x, cell_ids, edge_index = [], [], [], []
            for node in graph_nx.nodes:
                pos.append(graph_nx.nodes[node]["pos"])
                x.append([graph_nx.nodes[node]["features"][feature] for feature in graph_nx.nodes[node]["features"].keys()])
                cell_ids.append(graph_nx.nodes[node]["cell_id"])
            for src, dst in graph_nx.edges:
                edge_index.append([src, dst])
            graph_pyg.pos = torch.tensor(np.array(pos), dtype=torch.float32)
            graph_pyg.x = torch.tensor(np.array(x), dtype=torch.float32)
            graph_pyg.edge_index = torch.transpose(torch.tensor(np.array(edge_index), dtype=torch.int64), 1, 0)
            graph_pyg.cell_ids = cell_ids
            graph_pyg.num_nodes = len(graph_nx.nodes)
            graph_pyg.num_edges = len(graph_nx.edges)
            self.database_pyg.append(graph_pyg)
        self.num_node_classes = 14
        self.transforms = transforms
        
    def __len__(self) -> int:
        """
        Returns dataset length.

        Parameters
        ----------

        Returns
        -------
        
        """
        return len(self.database_pyg)

    def __getitem__(self, idx):
        """
        Fetches data object (pyg graph) from self.database_pyg and applies
        normalization upon data object fetching.

        Parameters
        ----------
        idx : integer
            Index to select which object to fetch from the self.database_pyg list.

        Returns
        -------
        
        """
        data = self.database_pyg[idx]
        if self.transforms is not None:
            data = self.transforms(data)
            
        return data


class CustomNormalizeFeatures(BaseTransform):
    """
    Row-normalizes data.x (custom to our case). Normalization constants are
    chosen as the average values for each attribute across the training dataset.
    Some exceptions apply:
        - Relative atributes (e.g., proximal/distal radius ratio, relative length): not normalized.
        - Direction vectors (e.g., direction and departure angle): normalized to norm == 1.

    Returns pyg graph (data) with normalized features (data.x attribute).
        
    """
    def __init__(self):
        """
        Initializes object of the CustomNormalizeFeatures class.
        
        Parameters
        ----------

        Returns
        -------

        """
        # These normalization constants are extracted from the average values of each attribute across
        # the training dataset
        self.normalization_constants = torch.tensor([5.30759506e+00, # mean rad
                                                     6.80836582e+00, # proximal radius
                                                     4.66676946e+00, # distal radius
                                                     1.00000000e+00, # proximal/distal radius
                                                     3.88116811e+00, # minimum radius
                                                     7.70207393e+00, # maximum radius
                                                     5.61961868e+01, # distance
                                                     1.00000000e+00, # relative length
                                                     1.00000000e+00, # direction r                        
                                                     1.00000000e+00, # direction a                                
                                                     1.00000000e+00, # direction s                                 
                                                     1.00000000e+00, # departure angle r
                                                     1.00000000e+00, # departure angle a                             
                                                     1.00000000e+00, # departure angle s                             
                                                     2.89809119e+02, # number of points 
                                                     1.13688445e+02, # proximal bifurcation r
                                                     9.73106914e+01, # proximal bifurcation a
                                                     8.53720525e+01, # proximal bifurcation s
                                                     1.18471183e+02, # distal bifurcation r
                                                     1.09963686e+02, # distal bifurcation a
                                                     1.21200789e+02, # distal bifurcation s
                                                     1.16859019e+02, # pos r
                                                     1.03625028e+02, # pos a
                                                     1.05396114e+02] # pos s
                                                     , dtype=torch.float32)

    def __call__(self, data):
        """
        Applies normalization upon data object (pyg graph) fetching from dataloader.

        Parameters
        ----------
        data : torch_geometric.data.data.Data object
            Pyg graph object.

        Returns
        -------
        normalized_data : torch_geometric.data.data.Data object
            Pyg graph object with normalized features (x).

        """
        normalized_data = data.__copy__()
        normalized_data.x = data.x / self.normalization_constants
        return normalized_data

def make_graph_plot(case_dir, graph, filename, label = None):
    """
    Makes matplotlib.pyplot figure of the coronal plane of a networkx graph.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    graph : networkx.Graph
        Graph that we want to plot.
    filename : string
        Fine name of the final image. Make sure to add a valid extension (e.g. .png, .eps, etc)
    label : string
        Edge attribute to be printed at the center of each graph edge.

    Returns
    -------

    """
    # Generate plot of dense graph for quick visualization
    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()

    # In order to place the nodes in the visualization of the graph in a sagittal view, 
    # we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_p = {}
    for n in graph.nodes():
        node_pos_dict_p[n] = [graph.nodes(data=True)[n]["pos"][0], graph.nodes(data=True)[n]["pos"][2]]

    if label is not None:
        edge_labels = nx.get_edge_attributes(graph, label)
        nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
        nx.draw_networkx_edge_labels(graph, node_pos_dict_p, edge_labels = edge_labels, ax=ax)
    else:
        nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)

    plt.savefig(os.path.join(case_dir, filename))
    plt.close()