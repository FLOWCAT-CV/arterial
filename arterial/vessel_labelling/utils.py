#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

import torch

from torch_geometric.data import Data, InMemoryDataset

from arterial.io.load_and_save_operations import load_json

class EVCDatasetInference(InMemoryDataset):
    """
    Dataset class for the Extarcranial Vascular Centerline (EVC) dataset. The dataset is
    composed of centerline graphs generated from vascular segmentations from Full head-and-neck 
    CTA images, from which the centerlines of the extracranial vessels were extracted. The original 
    dataset contains centerline graphs from 561 cases. Nodes correspond to bifurcations, while edges
    correspond to vascular segments. The graphs contained manually labelled vessel types for each edge.
    However, we perform a node transform to the graphs, converting edges into nodes and maintaining connectivity 
    by adding edges to all segments immediately in contact in all bifurcation points. This way, we convert 
    the problem to a node classification task.

    All vessels are a type of either of these 14 classes: other, AA, BT, RCCA, LCCA, RSA, LSA, RVA, 
        LVA, RICA, LICA, RECA, LECA, BA.

    The dataset description json file is mandatory for normalizaiton purposes. If it is not found, it will be
    created from the raw data. 

    Parameters
    ----------
    root : str
        Path to the root folder of the dataset. We assume that raw_dir (`raw`) and processed_dir (`processed`) 
        are subfolders of root.
    raw_file_names : list of str, optional
        List of raw file names, by default None. If None, all files in raw_dir are used.
    processed_file_names : list of str, optional
        List of processed file names, by default None. If None, all files in processed_dir are used.
    pre_transform : object, optional
        Pre-transform object, by default None. It should input a Data object and output a Data object.

    """
    def __init__(self, raw_graph, pre_transform = None):
        dataset_path = os.path.join(os.environ["arterial_dir"], "vessel_labelling/models/extracranial_vessels", "dataset.json")
        if os.path.exists(dataset_path):
            self.dataset_description = load_json(dataset_path)
        else:
            raise RuntimeError("No dataset description file found. Please retrieve this file for proper inference.")
        self.graphs = [raw_graph]
        self.pre_transform = pre_transform
        self.process()

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return None

    def process(self): 
        # Here you would read your raw files, create Data objects, and apply any pre-transforms
        data_list = []
        for raw_graph in self.graphs:
            graph_nx = node_transform(raw_graph)
            # Pytorch data object
            data = Data()
            x, edge_index, y, pos, cell_ids = [], [], [], [], []
            for node in graph_nx.nodes:
                pos.append(graph_nx.nodes[node]["pos"])
                x.append(graph_nx.nodes[node]["features"])
                cell_ids.append(graph_nx.nodes[node]["cell_id"])
            for n0, n1 in graph_nx.edges:
                edge_index.append(np.array([n0, n1]))
            data.pos = torch.tensor(np.array(pos), dtype=torch.float32)
            data.x = self.normalize_edge_features(torch.tensor(np.array(x), dtype=torch.float32))
            data.edge_index = torch.transpose(torch.tensor(np.array(edge_index), dtype=torch.int64), 1, 0)
            data.num_nodes = len(graph_nx.nodes)
            data.num_edges = len(graph_nx.edges)
            data.cell_ids = torch.tensor(np.array(cell_ids), dtype=torch.int64)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
    
        self.data_list = data_list

    def normalize_edge_features(self, edge_features):
        def z_score_normalization(edge_features, mean, std):
            return (edge_features - mean) / std
        def min_max_normalization(edge_features, min, max):
            return (edge_features - min) / (max - min)
        def mean_centering_normalization(edge_features, mean):
            return edge_features - mean
        def ensure_normalized_vectors(edge_features):
            return edge_features / torch.norm(edge_features, dim=1, keepdim=True)
        
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of edge features.")
            return edge_features
        else:
            # Compute min max values after mean centering for landmark positions normalization
            mean_position = self.dataset_description["mean_edge_features"][21:24]
            min_edge_features_mc = self.dataset_description["min_edge_features"].copy()
            max_edge_features_mc = self.dataset_description["max_edge_features"].copy()
            # Mean centering for r coordinates
            for idx in [15, 18, 21]:
                min_edge_features_mc[idx] -= mean_position[0]
                max_edge_features_mc[idx] -= mean_position[0]
            # Mean centering for a coordinates
            for idx in [16, 19, 22]:
                min_edge_features_mc[idx] -= mean_position[1]
                max_edge_features_mc[idx] -= mean_position[1]
            # Mean centering for s coordinates
            for idx in [17, 20, 23]:
                min_edge_features_mc[idx] -= mean_position[2]
                max_edge_features_mc[idx] -= mean_position[2]
            for idx, feature_name in enumerate(self.dataset_description["edge_feature_names"]):
                # For distance or continuous features, we apply z-score normalization
                if feature_name in ["mean radius", "proximal radius", "distal radius", "proximal/distal radius ratio", "minimum radius", "maximum radius", "distance", "relative length"]:
                    edge_features[:, idx] = z_score_normalization(edge_features[:, idx], self.dataset_description["mean_edge_features"][idx], self.dataset_description["std_edge_features"][idx])
                # For number of points, we apply min-max normalization
                elif feature_name in ["number of points"]:
                    edge_features[:, idx] = min_max_normalization(edge_features[:, idx], self.dataset_description["min_edge_features"][idx], self.dataset_description["max_edge_features"][idx])
                # For positional features, we apply mean centering normalization followed by min-max normalization
                elif feature_name in ["proximal bifurcation position r", "proximal bifurcation position a", "proximal bifurcation position s", 
                                      "distal bifurcation position r", "distal bifurcation position a", "distal bifurcation position s", 
                                      "pos r", "pos a", "pos s"]:
                    edge_features[:, idx] = mean_centering_normalization(edge_features[:, idx], self.dataset_description["mean_edge_features"][idx])
                    edge_features[:, idx] = min_max_normalization(edge_features[:, idx], min_edge_features_mc[idx], max_edge_features_mc[idx])
                # For direction and departure angle, we ensure normalized vectors
                edge_features[:, 8:11] = ensure_normalized_vectors(edge_features[:, 8:11]) # Direction
                edge_features[:, 11:14] = ensure_normalized_vectors(edge_features[:, 11:14]) # Departure angle

            return edge_features
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

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

def make_graph_plot(graph, label=None, subplot=None, show=False, output_path=None):
    """
    Makes matplotlib.pyplot figure of the coronal plane of a networkx graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that we want to plot.
    label : string
        Edge attribute to be printed at the center of each graph edge.
    subplot : matplotlib.axes.Axes
        Subplot on which to draw the graph.

    Returns
    -------

    """
    if subplot is None:
        _, ax = plt.subplots(figsize=[5, 10])
    else:
        ax = subplot

    # In order to place the nodes in the visualization of the graph in a sagittal view, 
    # we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_p = {}
    for n in graph.nodes():
        node_pos_dict_p[n] = [-graph.nodes(data=True)[n]["pos"][0], graph.nodes(data=True)[n]["pos"][2]]

    # Draw nodes
    nx.draw_networkx_nodes(graph, node_pos_dict_p, node_size=20, ax=ax)

    # Create a list of colors
    colors = plt.cm.jet(np.linspace(0, 1, len(graph.edges())))

    # Draw edges
    for (_, _, data), color in zip(graph.edges(data=True), colors):
        if 'coordinate_array' in data:
            ax.plot(-data['coordinate_array'][:, 0], data['coordinate_array'][:, 2], color = color, linewidth = 1)
            
            # Print label at the center of the segments array
            if label is not None and label in data:
                mid_point_idx = len(data['coordinate_array']) // 2
                mid_point = data['coordinate_array'][mid_point_idx]
                # We add a small displacement to the left to the label to better fit the center of the line
                ax.text(-mid_point[0] - 3, mid_point[2], str(data[label]), color = color, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha = 0.8))
        else:
            # If no coordinate_array is present, just draw a line between the two nodes
            if label is not None:
                edge_labels = nx.get_edge_attributes(graph, label)
                nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
                nx.draw_networkx_edge_labels(graph, node_pos_dict_p, edge_labels=edge_labels, ax=ax)
            else:
                nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
            break

    # Add labels to the plot
    plt.text(0.99, 0.5, 'L', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    plt.text(0.01, 0.5, 'R', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.text(0.5, 0.99, 'S', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.text(0.5, 0.01, 'I', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    if subplot is None:
        if output_path is not None:
            plt.savefig(output_path)
        if show:
            plt.show()
        else:
            plt.close()



""" The functions below belonged to an old implementation for intracranial vessel labelling. It is kept here for reference, 
but the final approach will be different. """

# def graph_data_to_df(case_dir, graph):
#     """
   
#     """
#     edges_features_dict = {}  # Initialize the final dictionary
#     intracranial_features = ['proximal bifurcation position i', 
#                              'proximal bifurcation position j', 
#                              'proximal bifurcation position k', 
#                              'distal bifurcation position i' ,
#                              'distal bifurcation position j', 
#                              'distal bifurcation position k', 
#                              'pos i',
#                              'pos j', 
#                              'pos k']

#     # Load the .nii.gz file and get its dimensions
#     i, j, k = nib.load(os.path.join(case_dir, '{}_cta.nii.gz'.format(os.path.basename(case_dir)))).get_fdata().shape

#     for src, dst in graph.edges:  # Iterate over the edges of the graph
#         edge_features_dict = {}  # Initialize the sub-dictionary
        
#         for key in graph[src][dst]['features']:  # Iterate over the features of the current edge
#             # Check if the current feature is in the list of special columns
#             if key in intracranial_features:
#                 # Normalize the feature value based on the dimension it represents
#                 if key.endswith('i'):
#                     edge_features_dict[key] = graph[src][dst]['features'][key] / i
#                 elif key.endswith('j'):
#                     edge_features_dict[key] = graph[src][dst]['features'][key] / j
#                 elif key.endswith('k'):
#                     edge_features_dict[key] = graph[src][dst]['features'][key] / k
#             else:
#                 # If the current feature is not a special column, just add it to the sub-dictionary
#                 edge_features_dict[key] = graph[src][dst]['features'][key]
                
#         edge_features_dict['cell_id'] = graph[src][dst]['cell_id']  # Add the cell ID to the sub-dictionary
        
#         # Add the sub-dictionary to the final dictionary, using the edge's nodes as the key
#         edges_features_dict[str(src)+ '_' + str(dst)] = edge_features_dict

#     return pd.DataFrame(edges_features_dict).T  # Return the final dictionary as a DataFrame

# def save_predicted_graph(case_dir, graph, predicted_vessels, mode = "vessels"):
#     """ 
#     Integrates the predicted vessel types from the inference of the 
#     graph U-Net over the node form graph on the original graph. 

#     Saves graph and image with predicted vessel types as:

#     >>> case_dir/graph_pred.pickle
#     >>> case_dir/graph_pred.png

#     Parameters
#     ----------
#     case_dir : string or path-like object
#         Path to case directory. 
#     graph : networkx.Graph
#         Graph in original form, where edges encode vessels.
#     predicted_vessels : dict
#         Dictionary with cell_ids as keys and predicted vessel types as values.
    
#     Returns
#     -------
    
#     """
#     if mode == "vessels":
#         edge_types = dict(zip([idx for idx in range(14)], 
#                             ["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"]))
#     elif mode == "intracranial_vessels":
#         edge_types = dict(zip([idx for idx in range(8)], 
#                         ["other", "LICA", "RICA", "BA", "LM1", "RM1", "LM2", "RM2", "LA1"]))
#     # Define edge_types dict
#     # Define new graph with same nodes and edges and information from the input graph
#     predicted_graph = nx.Graph()
#     # For nodes, we only keep the position of the original edge form graph nodes. We use these for visualization in the png file
#     node_pos_dict_p = {}
#     for node in graph.nodes:
#         predicted_graph.add_node(node)
#         predicted_graph.nodes(data=True)[node]["pos"] = graph.nodes(data=True)[node]["pos"]
#         # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates 
#         # (the view will be from the coronal plane, P axis)
#         node_pos_dict_p[node] = [graph.nodes(data=True)[node]["pos"][0], graph.nodes(data=True)[node]["pos"][2]]

#     # For edges, we keep all information from the original graph, and in addition we set the vessel type from the predicted_vessels dict
#     for src, dst in graph.edges:
#         predicted_graph.add_edge(src, dst)
#         predicted_graph[src][dst]["cell_id"] = graph[src][dst]["cell_id"]
#         predicted_graph[src][dst]["vessel_type"] = predicted_vessels[graph[src][dst]["cell_id"]]
#         predicted_graph[src][dst]["vessel_type_name"] = edge_types[predicted_vessels[graph[src][dst]["cell_id"]]]
#         predicted_graph[src][dst]["features"] = graph[src][dst]["features"]

#     # Save the graph and image for quick visualization
#     save_pickle(predicted_graph, os.path.join(case_dir, "{}_graph_pred.pickle".format(mode)))
#     make_graph_plot(case_dir, predicted_graph, "{}_graph_pred.png".format(mode), label = "vessel_type_name")