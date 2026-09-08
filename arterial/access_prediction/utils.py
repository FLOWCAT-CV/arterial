#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

import os

import numpy as np
import networkx as nx
import nibabel as nib
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.transforms import RadiusGraph

import scipy.sparse as sp

import vtk
from vtk.util import numpy_support

from arterial.io.load_and_save_operations import load_json

def supersegment_sanity_check(supersegment):
    """
    Performs a sanity check on a supersegment based on minimum length.

    Parameters
    ----------
    supersegment : networkx.Graph
        Supersegment to check.

    Returns
    -------
    bool: True if the supersegment is valid, False otherwise.

    """
    if len(supersegment) < 10:
        return False
    return True


class ArterialGNetDatasetInference(Dataset):
    """
    Dataset class for the Arterial Maps dataset, containing a set of objects encoding
    information for the arterial pathway (supersegment) for a stroke patient who underwent
    endovascular treatment at different scales.

    Information is encoded in the following objects:
    - Global features: features descriptive of the entire supersegment
    - Segment graph: graph of the supersegment, with each node representing a vessel segment
    - Dense graph: graph of the supersegment, with each node representing a local centerline point

    This class allows us to create the final Pytorch Geometric Data objects that will be processed 
    by the models.

    """
    def __init__(self, preprocessed_supersegment_dict, pre_transform = None, transform = None, use_lap_pos_enc = False, pos_enc_dim = 8):
        """
        Initializes the ArterialGNetDatasetInference class.

        Parameters
        ----------
        preprocessed_supersegment_dict : dict
            Dictionary containing the preprocessed supersegment, with the global features,
            segment graph and dense graph.
        pre_transform : torch_geometric.transforms.Compose
            Pre-transform to apply to the data.
        transform : torch_geometric.transforms.Compose
            Transform to apply to the data.
        use_lap_pos_enc : bool
            Whether to use Laplacian positional encoding.
        pos_enc_dim : int
            Dimension of the positional encoding.

        Returns
        -------

        """
        self.raw_file = preprocessed_supersegment_dict
        if not os.path.exists(os.path.join(os.environ["arterial_dir"], "access_prediction/models/dataset.json")):
            raise FileNotFoundError("No dataset description file found. This will be an issue for normalization of features.")
        self.dataset_description = load_json(os.path.join(os.environ["arterial_dir"], "access_prediction/models/dataset.json"))
        self.use_lap_pos_enc = use_lap_pos_enc
        self.pos_enc_dim = pos_enc_dim
        self.pre_transform = pre_transform
        self.transform = transform
        self.process()

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return None
        
    def process(self): 
        """
        Processes the raw file into a Pytorch Geometric Data object. Creates a different
        Data object for the global features (a normal tensor), segment and dense graphs
        (PyG Data objects).

        Then sets the three objects into a normal torch Data object.

        Parameters
        ----------

        Returns
        -------

        """
        raw_global_features = self.raw_file["global_features"]
        raw_segment_graph = self.clean_node_indexing(self.raw_file["segment_graph"])
        raw_dense_graph = self.clean_node_indexing(self.raw_file["dense_graph"])

        # We will wrap the raw data into Pytorch Geometric Data objects
        data = Data()

        # For global features, we just transform it into a tensor
        data.global_data = torch.tensor(self.normalize_global_features(raw_global_features["features_list"]), dtype=torch.float32)

        # For segment features, we create a Pytorch data object
        data.segment_data = Data()
        segment_x, segment_edge_index, segment_egde_attr, segment_pos = [], [], [], []
        for node in raw_segment_graph.nodes:
            segment_pos.append(raw_segment_graph.nodes[node]["pos"])
            segment_x.append(raw_segment_graph.nodes[node]["features_list"])
        for src, dst in raw_segment_graph.edges:
            segment_edge_index.append(np.array([src, dst]))
            segment_egde_attr.append(raw_segment_graph.edges[src, dst]["features_list"])
        data.segment_data.pos = torch.tensor(np.array(segment_pos), dtype=torch.float32)
        data.segment_data.x = self.normalize_segment_node_features(torch.tensor(np.array(segment_x), dtype=torch.float32))
        data.segment_data.edge_attr = self.normalize_segment_edge_features(torch.tensor(np.array(segment_egde_attr), dtype=torch.float32))
        data.segment_data.edge_index = torch.transpose(torch.tensor(np.array(segment_edge_index), dtype=torch.int64), 1, 0)
        data.segment_data.num_nodes = len(raw_segment_graph.nodes)
        data.segment_data.num_edges = len(raw_segment_graph.edges)

        # For dense features, we create a Pytorch data object
        data.dense_data = Data()
        dense_x, dense_edge_index, dense_pos = [], [], []
        for node in raw_dense_graph.nodes:
            dense_pos.append(raw_dense_graph.nodes[node]["pos"])
            dense_x.append(raw_dense_graph.nodes[node]["features_list"])
        for src, dst in raw_dense_graph.edges:
            dense_edge_index.append(np.array([src, dst]))
        data.dense_data.pos = torch.tensor(np.array(dense_pos), dtype=torch.float32)
        data.dense_data.x = self.normalize_dense_node_features(torch.tensor(np.array(dense_x), dtype=torch.float32))
        data.dense_data.edge_index = torch.transpose(torch.tensor(np.array(dense_edge_index), dtype=torch.int64), 1, 0)
        data.dense_data.num_nodes = len(raw_dense_graph.nodes)
        data.dense_data.num_edges = len(raw_dense_graph.edges)
        if self.use_lap_pos_enc:
            data.dense_data.lap_pos_enc = laplacian_positional_encoding(data.dense_data.edge_index, data.dense_data.num_nodes, self.pos_enc_dim)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.data_list = [data]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def clean_node_indexing(self, graph_nx):
        """
        Cleans the node indexing of a networkx graph by reindexing the nodes from 0 to N-1, where N is the number of nodes.

        Parameters
        ----------
        graph_nx : networkx.Graph
            Networkx graph.

        Returns
        -------
        graph_nx : networkx.Graph
            Networkx graph with cleaned node indexing.

        """
        mapping = {node: idx for idx, node in enumerate(graph_nx.nodes)}
        return nx.relabel_nodes(graph_nx, mapping)
    
    def normalize_global_features(self, global_features):      
        """
        Normalizes the global features. Uses the dataset description file to normalize the features.

        Uses min-max normalization for the features that are bounded (i.e., tortuosity index, min polar 
        angle, polar angle, azimuthal angle), no normalization for one-hot encoded features (i.e., side, 
        vessel type), and z-score normalization for the rest of the features.

        These features correspond to extracted features treating the supersegment as a single
        segment. Thus, the featurization process is the same as that used for the segment graph node features
        (which correspond to single vessel segments).

        Parameters
        ----------
        global_features : torch.Tensor
            Global features to normalize.

        Returns
        -------
        global_features : torch.Tensor
            Normalized global features.

        """
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return global_features
        else:
            mean_features = self.dataset_description["mean_global_features"]
            std_features = self.dataset_description["std_global_features"]
            min_features = self.dataset_description["min_global_features"]
            max_features = self.dataset_description["max_global_features"]

            for idx, feature_name in enumerate(self.dataset_description["global_feature_names"]):
                if feature_name in ["side", "vessel_type"]:
                    pass
                elif feature_name in ["tortuosity_index", "min_polar_angle", "polar_angle", "azimuthal_angle"]:
                    global_features[idx] = min_max_normalization(global_features[idx], min_features[idx], max_features[idx])
                else:
                    global_features[idx] = z_score_normalization(global_features[idx], mean_features[idx], std_features[idx])

            return global_features

    def normalize_segment_node_features(self, segment_node_features):
        """
        Normalizes the segment node features. Uses the dataset description file to normalize the features.

        Uses min-max normalization for the features that are bounded (i.e., tortuosity index, min polar 
        angle, polar angle, azimuthal angle), no normalization for one-hot encoded features (i.e., vessel type),
        and z-score normalization for the rest of the features.

        Parameters
        ----------
        segment_node_features : torch.Tensor
            Segment node features to normalize.

        Returns
        -------
        segment_node_features : torch.Tensor
            Normalized segment node features.

        """
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return segment_node_features
        else:
            mean_features = self.dataset_description["mean_segment_node_features"]
            std_features = self.dataset_description["std_segment_node_features"]    
            min_features = self.dataset_description["min_segment_node_features"]
            max_features = self.dataset_description["max_segment_node_features"]

            for idx, feature_name in enumerate(self.dataset_description["segment_node_feature_names"]):
                if feature_name in ["vessel_type"]:
                    pass
                elif feature_name in ["tortuosity_index", "min_polar_angle", "polar_angle", "azimuthal_angle"]:
                    segment_node_features[:, idx] = min_max_normalization(segment_node_features[:, idx], min_features[idx], max_features[idx])
                else:
                    segment_node_features[:, idx] = z_score_normalization(segment_node_features[:, idx], mean_features[idx], std_features[idx])
            
            return segment_node_features
        
    def normalize_segment_edge_features(self, segment_edge_features):
        """
        Normalizes the segment edge features. Uses the dataset description file to normalize the features.

        Uses min-max normalization for the features that are bounded (i.e., max angle difference, max azimuthal difference, max polar difference),
        and z-score normalization for the rest of the features.

        These features correspond to extracted features involving two consecutive vessel segments.

        Parameters
        ----------
        segment_edge_features : torch.Tensor
            Segment edge features to normalize.

        Returns
        -------
        segment_edge_features : torch.Tensor
            Normalized segment edge features.

        """
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return segment_edge_features
        else:
            mean_features = self.dataset_description["mean_segment_edge_features"]
            std_features = self.dataset_description["std_segment_edge_features"]
            min_features = self.dataset_description["min_segment_edge_features"]
            max_features = self.dataset_description["max_segment_edge_features"]

            for idx, feature_name in enumerate(self.dataset_description["segment_edge_feature_names"]):
                if feature_name in ["max angle difference", "max azimuthal difference", "max polar difference"]:
                    segment_edge_features[:, idx] = min_max_normalization(segment_edge_features[:, idx], min_features[idx], max_features[idx])
                else:
                    segment_edge_features[:, idx] = z_score_normalization(segment_edge_features[:, idx], mean_features[idx], std_features[idx])

            return segment_edge_features
    
    def normalize_dense_node_features(self, dense_node_features):
        """
        Normalizes the dense node features. Uses the dataset description file to normalize the features.

        Uses min-max normalization for the features that are bounded (i.e., curvature, torsion, polar angle, 
        azimuthal angle, accumulated length from access), no normalization for one-hot encoded features (i.e., 
        blanking, vessel type), and z-score normalization for the rest of the features.

        Parameters
        ----------
        dense_node_features : torch.Tensor
            Dense node features to normalize.

        Returns
        -------
        dense_node_features : torch.Tensor
            Normalized dense node features.
            
        """
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return dense_node_features
        else:
            mean_features = self.dataset_description["mean_dense_node_features"]
            std_features = self.dataset_description["std_dense_node_features"]
            min_features = self.dataset_description["min_dense_node_features"]
            max_features = self.dataset_description["max_dense_node_features"]

            for idx, feature_name in enumerate(self.dataset_description["dense_node_feature_names"]):
                if feature_name in ["blanking", "vessel_type"]:
                    pass
                elif feature_name in ["curvature", "torsion", "polar_angle", "azimuthal_angle", "accumulated_length_from_access"]:
                    dense_node_features[:, idx] = min_max_normalization(dense_node_features[:, idx], min_features[idx], max_features[idx])
                else:
                    dense_node_features[:, idx] = z_score_normalization(dense_node_features[:, idx], mean_features[idx], std_features[idx])

            return dense_node_features
 
def z_score_normalization(features, mean, std):
    return (features - mean) / std

def min_max_normalization(features, min, max):
    return (features - min) / (max - min)

def mean_centering_normalization(features, mean):
    return features - mean

def normalize_vector(features):
    return features / torch.norm(features, dim=1, keepdim=True)

def laplacian_positional_encoding(edge_index, num_nodes, pos_enc_dim=8):
    """
    Graph positional encoding v/ Laplacian eigenvectors for PyTorch Geometric
    
    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity in COO format with shape [2, num_edges]
    num_nodes : int
        Number of nodes in the graph
    pos_enc_dim : int
        Desired dimension of positional encoding
    
    Returns
    -------
    torch.Tensor
        Laplacian positional encoding matrix with shape [num_nodes, pos_enc_dim].

    """
    # Get normalized Laplacian
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    
    # Convert to scipy sparse matrix
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    
    # Eigenvectors with scipy
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', return_eigenvectors=True)
    EigVec = EigVec.real
    
    # Sort and keep top K eigenvectors
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], EigVec[:, idx]
    
    # Discard the first eigenvector/value
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    
    return lap_pos_enc

def sinusoidal_positional_encoding(num_nodes, pos_enc_dim):  
    """
    Sinusoidal positional encoding.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    pos_enc_dim : int
        Dimension of the positional encoding.

    Returns
    -------
    torch.Tensor
        Positional encoding matrix with shape [num_nodes, dim].

    """
    pos = torch.arange(0, num_nodes).unsqueeze(1)
    i = torch.arange(0, pos_enc_dim // 2).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * i) / pos_enc_dim)
    pos_encoding = torch.zeros(num_nodes, pos_enc_dim)
    pos_encoding[:, 0::2] = torch.sin(pos * angle_rates)
    pos_encoding[:, 1::2] = torch.cos(pos * angle_rates)
    return pos_encoding

class DenseRadiusGraph(object):
    """
    Radius graph transform for the dense graph.
    This is needeed because of the structure of a standard Data object of the
    ArterialGNetDatasetInference class, which contains a dense_data which corresponds
    to a standard PyTorch Geometric Data object, where we want to apply the RadiusGraph
    transform.

    One difference to the normal RadiusGraph transform is that we combine the original
    edge_index with the new edge_index generated by the RadiusGraph transform, and we use
    torch.unique to ensure that we do not have duplicate edges. This is done to preserve
    the original edge_index, while adding the new edges generated by the RadiusGraph
    transform.

    """
    def __init__(self, r, loop=True):
        """
        Initializes the RadiusGraph transform.

        Parameters
        ----------
        r : float
            Radius of the graph.
        loop : bool
            Whether to add self-loops to the graph.

        Returns
        -------

        """
        self.transform = RadiusGraph(r=r, loop=loop)

    def __call__(self, data):
        """
        Applies the RadiusGraph transform to the dense graph. It first
        checks if the dense_data attribute exists in the data object, and if so,
        applies the transform to the dense_data object. Otherwise, it applies the
        transform directly on the data object (same as the normal RadiusGraph transform).

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data object containing the dense graph.

        Returns
        -------
        data : torch_geometric.data.Data
            Data object containing the dense graph with the RadiusGraph transform applied.

        """
        if hasattr(data, 'dense_data'):
            dense_data = data.dense_data
            original_edge_index = dense_data.edge_index
            new_dense_data = self.transform(dense_data)
            new_edge_index = new_dense_data.edge_index

            combined_edge_index = torch.cat([original_edge_index, new_edge_index], dim=1)

            # Transpose to treat each edge as a row
            transposed_edges = combined_edge_index.T
            # Use torch.unique to find unique rows (edges)
            unique_edges = torch.unique(transposed_edges, dim=0)
            # Transpose back to the original edge_index format
            combined_edge_index = unique_edges.T

            dense_data.edge_index = combined_edge_index
            data.dense_data = dense_data
        else:
            original_edge_index = data.edge_index
            new_data = self.transform(data)
            new_edge_index = new_data.edge_index

            combined_edge_index = torch.cat([original_edge_index, new_edge_index], dim=1)

            # Transpose to treat each edge as a row
            transposed_edges = combined_edge_index.T
            # Use torch.unique to find unique rows (edges)
            unique_edges = torch.unique(transposed_edges, dim=0)
            # Transpose back to the original edge_index format
            combined_edge_index = unique_edges.T

            data.edge_index = combined_edge_index

        return data
    
def collate_ArterialGNetInference(data_list):
    """
    Collate function for the ArterialGNetDatasetInference class.
    This is needed because the Data objects are not directly batchable, and we need to
    create a new Data object for the batch.

    Each Data object contains a global_data, segment_data and dense_data attribute. The
    global data object is simply a 1D tensor of features, which can be stacked directly.
    The segment_data and dense_data objects are PyG Data objects, which can be stacked
    using the Batch class.

    Parameters
    ----------
    data_list : list
        List of Data objects.

    Returns
    -------
    batch : torch_geometric.data.Batch
        Batch object.

    """
    device = data_list[0].dense_data.x.device
    # Create a new Data object for the batch
    batch = Data()

    # Standard batching for global data and labels
    batch.global_data = torch.stack([item.global_data for item in data_list], dim=0).to(device)

    # Use PyG's Batch to batch segment_data and dense_data
    # Since these are PyG Data objects, they can be directly batched
    batch.segment_data = Batch.from_data_list([item.segment_data for item in data_list])
    batch.dense_data = Batch.from_data_list([item.dense_data for item in data_list])
    batch.batch = batch.dense_data.batch

    return batch
   
def make_combined_plot(side, segment_graph, dense_graph, output_path=None):
    """
    Makes a combined plot of the segment graph and the dense graph, plotted side by side.

    Parameters
    ----------
    side : str
        Side of the arterial pathway.
    segment_graph : networkx.Graph
        Segment graph representation of the supersegment.
    dense_graph : networkx.Graph
        Dense graph representation of the supersegment.
    output_path : str
        Path to save the plot.

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), dpi = 300)
    # Set side as title of the whole plot
    fig.suptitle(side, fontsize=16)

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in segment_graph.nodes():
        node_pos_dict_P[n] = [-segment_graph.nodes(data=True)[n]["pos"][0], segment_graph.nodes(data=True)[n]["pos"][2]]

    nx.draw(segment_graph, node_pos_dict_P, node_size=10, ax=ax[0])
    ax[0].set_xlim([-200, 10])
    ax[0].set_ylim([-10, 300])

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in dense_graph.nodes():
        node_pos_dict_P[n] = [-dense_graph.nodes(data=True)[n]["pos"][0], dense_graph.nodes(data=True)[n]["pos"][2]]

    nx.draw(dense_graph, node_pos_dict_P, node_size=10, ax=ax[1])
    ax[1].set_xlim([-200, 10])
    ax[1].set_ylim([-10, 300])

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()

def get_lpi_corner_coordinates(cta_nifti):
    """
    Computes LPI corner coordinates from the CTA nifti to adjust positions 
    of the attention maps, for these to be coherent with 3D models directly derived
    from the segmentation nifti.

    Parameters
    ----------
    cta_nifti : nib.Nifti1Image
        CTA nifti image object.

    Returns
    -------
    lpi_corner_coordinates : numpy.ndarray
        LPI corner coordinates of the CTA nifti image.

    """
    # Get lpi corner coordinates
    cta_array = cta_nifti.get_fdata()
    cta_affine = cta_nifti.affine
    # Depending on the orientation of the image, we have to define the corner voxel coordinates and the flipping array
    orientation = nib.aff2axcodes(cta_affine)
    if orientation == ('R', 'A', 'S'):
        lpi_corner_voxel_coordinates = np.array([0, 0, 0])
    elif orientation == ('L', 'A', 'S'):
        lpi_corner_voxel_coordinates = np.array([cta_array.shape[0] - 1, 0, 0])
    elif orientation == ('L', 'P', 'S'):
        lpi_corner_voxel_coordinates = np.array([cta_array.shape[0] - 1, cta_array.shape[1] - 1, 0])

    # Compute lpi corner coordinates in real world coordinates, with the same orientation as the image
    lpi_corner_coordinates = np.dot(cta_affine, np.append(lpi_corner_voxel_coordinates, 1))[:3]

    return lpi_corner_coordinates

def build_final_attention_map(edge_indices, edge_attention_weights, batch, lpi_corner_coordinates):
    """
    Builds final attention map as a 1-D networkx graph, removing self-loops and loops added
    artificially by the RadiusGraph transform.

    Parameters
    ----------
    edge_indices : torch.Tensor
        Edge indices of the attention map.
    edge_attention_weights : torch.Tensor
        Edge attention weights of the attention map.
    batch : torch_geometric.data.Batch
        Batch object of the attention map.
    lpi_corner_coordinates : numpy.ndarray
        LPI corner coordinates of the attention map. Used to adjust the position of the nodes
        for these to be coherent with 3D models directly derived from the segmentation nifti.

    Returns
    -------
    attention_map_graph : networkx.Graph
        Networkx graph of the attention map. Stores attention weights as a edge features, and
        averaged edge attention weights as a node feature.

    """
    # Sort edge_indices by first row
    sorted_indices = np.argsort(edge_indices[0])
    edge_indices = edge_indices[:, sorted_indices]
    edge_attention_weights = edge_attention_weights[sorted_indices]

        
    G = nx.Graph()
    for i, edge in enumerate(edge_indices.T):
        src, dst = edge
        G.add_edge(src, dst, weights=edge_attention_weights[i].mean())
        # Set position of the nodes
        G.nodes[src]["pos"] = batch.dense_data.pos[src].detach().cpu().numpy() + lpi_corner_coordinates
        G.nodes[dst]["pos"] = batch.dense_data.pos[dst].detach().cpu().numpy() + lpi_corner_coordinates

    # Create a node feature that aggregates the attention weights of the edges connected to the node
    for node in G.nodes:
        G.nodes[node]["features femoral"] = {}
        edges = G.edges(node)
        weights = [G[u][v]["weights"] for u, v in edges]
        G.nodes[node]["features femoral"]["attention_weight"] = sum(weights) / len(weights)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Make 1D graph
    attention_map_graph = nx.Graph()
    previous_node = None
    for node in range(G.number_of_nodes()):
        attention_map_graph.add_node(node, pos=G.nodes[node]["pos"])
        attention_map_graph.nodes[node]["features femoral"] = {}
        attention_map_graph.nodes[node]["features femoral"]["attention_weight"] = G.nodes[node]["features femoral"]["attention_weight"]
        if previous_node is not None:
            # Check if connection exists, otherwise create it (rare)
            if G.has_edge(previous_node, node):
                attention_map_graph.add_edge(previous_node, node, weights=G[previous_node][node]["weights"])
            else:
                attention_map_graph.add_edge(previous_node, node)
                attention_map_graph[previous_node][node]["weights"] = (G.nodes[node]["features femoral"]["attention_weight"] + G.nodes[previous_node]["features femoral"]["attention_weight"]) / 2
        previous_node = node
    
    return attention_map_graph

def nx_graph_to_vtk_polydata(G):
    """
    Generates a vtkPolyData object from a networkx graph, with attention weights as 
    point data.

    Parameters
    ----------
    G : networkx.Graph
        Networkx graph of the centerline graph.

    Returns
    -------
    polydata : vtk.vtkPolyData
        vtkPolyData object of the centerline graph with attention weights as point data.

    """
    # Create a vtkPoints object to store node coordinates
    points = vtk.vtkPoints()
    
    # Create a vtkPolyData object to store the entire structure
    polydata = vtk.vtkPolyData()
    
    # Create a vtkCellArray to store the single polyline
    lines = vtk.vtkCellArray()
    
    # Create dictionaries to store scalar arrays
    scalar_arrays = {}
    
    # Create a single vtkPolyLine for the entire centerline
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(G.number_of_nodes())
    
    # Iterate through nodes to add points and extract scalar values
    for i, (node, data) in enumerate(G.nodes(data=True)):
        # Add node coordinates to points
        points.InsertNextPoint(data['pos'])
        
        # Add point to the polyline
        polyline.GetPointIds().SetId(i, i)
        
        # Extract scalar values from 'features femoral'
        for key, value in data['features femoral'].items():
            if key not in scalar_arrays:
                scalar_arrays[key] = []
            scalar_arrays[key].append(value)
    
    # Add the single polyline to the cell array
    lines.InsertNextCell(polyline)
    
    # Set the points and lines in the polydata
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    
    # Add scalar arrays to the polydata
    for key, values in scalar_arrays.items():
        vtk_array = numpy_support.numpy_to_vtk(np.array(values), deep=True)
        vtk_array.SetName(key)
        polydata.GetPointData().AddArray(vtk_array)
    
    return polydata

def nx_graph_to_point_dict(G):
    """
    Generates a new dictionary with the point positions of the graph in a 'points' list, and the
    attention weights of the graph in a 'attention_weight' list.

    Parameters
    ----------
    G : networkx.Graph
        Networkx graph.

    Returns
    -------
    point_dict : dict
        Dictionary with the point positions and attention weights.

    """
    point_dict = {}
    point_dict["points"] = [G.nodes[node]["pos"].tolist() if isinstance(G.nodes[node]["pos"], np.ndarray) else list(G.nodes[node]["pos"])  for node in G.nodes]
    point_dict["attention_weight"] = [float(G.nodes[node]["features femoral"]["attention_weight"]) for node in G.nodes]
    return point_dict