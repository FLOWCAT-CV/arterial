#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToDevice
from torch_geometric.nn.models import GAT

from arterial.vessel_labelling.utils import EVCDatasetInference
from arterial.io.load_and_save_operations import load_json

def perform_inference(graph, mode = "extracranial_vessels", ensemble=True):
    """
    Performs inference of segments graph with a trained graph neural network. 
    
    The implemented model is a node classification GATv2 network. Since in the original 
    form of the graphs, vessels are encoded by edges, we first transform the graphs into "node 
    form", where edges turn into nodes. Edges connect vessels that share bifurcations in the 
    original graph.

    Then inference is performed using one trained model or a 5-fold ensemble.

    Parameters
    ----------
    graph : networkx.Graph
        Graph in edge form, where edges encode vessels.
    mode : string, optional
        Mode of the vessel labeller. The default is "extracranial_vessels", it can also be "intracranial_vessels".
    ensemble : bool, optional
        Whether to use an ensemble of 5 models. The default is True.

    Returns
    -------
    predicted_graph : networkx.Graph
        Graph in edge form, where edges encode vessels and have predicted vessel types.

    """
    if mode == "extracranial_vessels":
        # Perform inference with the trained model
        if ensemble:
            predicted_graph = predict_extracranial_vessel_types_ensemble(graph)
        else:
            predicted_graph = predict_extracranial_vessel_types(graph)
        return predicted_graph
    
    elif mode == "intracranial_vessels":
        raise NotImplementedError("Intracranial vessel labelling is not implemented yet.")

def predict_extracranial_vessel_types(graph):
    """ 
    Performs inference over the node form graph with the trained graqh U-Net
    model for extracranial vessel labelling. Return the same graph with 
    predicted vessel types (vessel_type and vessel_type_name attributes) in the edges.

    Parameters
    ----------
    model : torch_geometric.nn.models.graph_unet_GraphUNet object
        Trained GNN node classification model.
    tranformed_graph : networkx.Graph
        Graph in node form, where nodes encode vessels.
    
    Returns
    -------
    predicted_vessels : dict
        Dictionary with cell_ids as keys and predicted vessel types as values.
    
    """
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define pre-tranforme
    pre_transform = Compose([
        RadiusGraph(r = 0.5, max_num_neighbors = 10),
        ToDevice(device)
    ])
    # Load file in the dataset class
    data = EVCDatasetInference(raw_graph=graph, pre_transform=pre_transform)
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    data_loader = DataLoader(data, batch_size = 1, shuffle = False) 
    # Load the trained model for inference
    state_dict = torch.load(os.path.join(os.environ["arterial_dir"], "vessel_labelling/models/extracranial_vessels/model_weights.pth"), map_location=torch.device(device), weights_only=False)
    model = GAT(
        in_channels = state_dict["init_kwargs"]["in_channels"],
        hidden_channels = state_dict["init_kwargs"]["hidden_channels"],
        out_channels = state_dict["init_kwargs"]["out_channels"],
        num_layers = state_dict["init_kwargs"]["num_layers"],
        v2=True,
        act = state_dict["init_kwargs"]["act"]
    ).to(device)
    model.load_state_dict(state_dict["state_dict"], strict=False)
    # Load model to device
    model.eval()
    # We have to iterate over the DataLoader (even thogh it will just be one graph at the time)
    for graph_preprocessed in data_loader:
        # Inference returns a tensor with the softmax probabilities for the vessel type for each node
        # We perform argmax to obtain the vessel type with the highest probability and pass it to list
        # The result is a 1D list with the predicted vessel types for each node
        predicted_nodes = model(graph_preprocessed.x, graph_preprocessed.edge_index).argmax(dim=1).tolist()

    # We create a dict to link the cell_ids from the centerline_segments_array to the predicted vessel types
    predicted_vessels = {}
    for idx, cell_id in enumerate(data.data_list[0].cell_ids):
        predicted_vessels[int(cell_id)] = predicted_nodes[idx]

    # Save that information in the graph
    predicted_graph = graph.copy()
    # Get edge type dict from the dataset.json
    dataset_description = load_json(os.path.join(os.environ["arterial_dir"], "vessel_labelling/models/extracranial_vessels/dataset.json"))
    edge_labels_dict = dataset_description["edge_labels_dict"]
    # For edges, we keep all information from the original graph, and in addition we set the vessel type from the predicted_vessels dict
    for src, dst in predicted_graph.edges:
        predicted_graph[src][dst]["vessel_type"] = predicted_vessels[predicted_graph[src][dst]["cell_id"]]
        predicted_graph[src][dst]["vessel_type_name"] = edge_labels_dict[str(predicted_graph[src][dst]["vessel_type"])]
    
    return predicted_graph

def predict_extracranial_vessel_types_ensemble(graph):
    """ 
    Performs inference over the node form graph with the trained graqh U-Net
    model for extracranial vessel labelling. Return the same graph with 
    predicted vessel types (vessel_type and vessel_type_name attributes) in the edges.

    Parameters
    ----------
    model : torch_geometric.nn.models.graph_unet_GraphUNet object
        Trained GNN node classification model.
    tranformed_graph : networkx.Graph
        Graph in node form, where nodes encode vessels.
    
    Returns
    -------
    predicted_vessels : dict
        Dictionary with cell_ids as keys and predicted vessel types as values.
    
    """
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define pre-tranforme
    pre_transform = Compose([
        RadiusGraph(r = 0.5, max_num_neighbors = 10),
        ToDevice(device)
    ])
    # Load file in the dataset class
    data = EVCDatasetInference(raw_graph=graph, pre_transform=pre_transform)
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    data_loader = DataLoader(data, batch_size = 1, shuffle = False) 

    # Get edge type dict from the dataset.json
    dataset_description = load_json(os.path.join(os.environ["arterial_dir"], "vessel_labelling/models/extracranial_vessels/dataset.json"))

    # create tensor of size of length(graph_preprocessed.x), with numi_classes number of channels, and depth 5 for folds
    # We will average logits across folds and then argmax to choose final class
    predicted_nodes = torch.zeros(len(next(iter(data_loader)).x), dataset_description["num_edge_classes"], 5).to(device)

    with torch.no_grad():
        for fold in range(5):
            # Load the trained model for inference
            # model = torch.load(os.path.join(os.environ["arterial_dir"], f"vessel_labelling/models/extracranial_vessels/fold_{fold}/model.pth"), map_location=torch.device(device), weights_only=False).to(device)
            state_dict = torch.load(os.path.join(os.environ["arterial_dir"], f"vessel_labelling/models/extracranial_vessels/fold_{fold}/model_weights.pth"), map_location=torch.device(device), weights_only=False)
            model = GAT(
                in_channels = state_dict["init_kwargs"]["in_channels"],
                hidden_channels = state_dict["init_kwargs"]["hidden_channels"],
                out_channels = state_dict["init_kwargs"]["out_channels"],
                num_layers = state_dict["init_kwargs"]["num_layers"],
                v2=True,
                act = state_dict["init_kwargs"]["act"]
            ).to(device)
            model.load_state_dict(state_dict["state_dict"], strict=False)
            # Load model to device
            model.eval()
            # We have to iterate over the DataLoader (even thogh it will just be one graph at the time)
            for graph_preprocessed in data_loader:
                # Inference returns a tensor with the softmax probabilities for the vessel type for each node
                # We perform argmax to obtain the vessel type with the highest probability and pass it to list
                # The result is a 1D list with the predicted vessel types for each node
                predicted_nodes[:, :, fold] = F.softmax(model(graph_preprocessed.x, graph_preprocessed.edge_index), dim=1).to(device)

    # Get mean of the logits across folds. Perform argmax to obtain the vessel type with the highest probability and pass it to list
    predicted_nodes = predicted_nodes.mean(dim=-1).argmax(dim=1).cpu().tolist()
    # 
    # We create a dict to link the cell_ids from the centerline_segments_array to the predicted vessel types
    predicted_vessels = {}
    for idx, cell_id in enumerate(data.data_list[0].cell_ids):
        predicted_vessels[int(cell_id)] = predicted_nodes[idx]

    # Save that information in the graph
    predicted_graph = graph.copy()
    edge_labels_dict = dataset_description["edge_labels_dict"]
    # For edges, we keep all information from the original graph, and in addition we set the vessel type from the predicted_vessels dict
    for src, dst in predicted_graph.edges:
        predicted_graph[src][dst]["vessel_type"] = predicted_vessels[predicted_graph[src][dst]["cell_id"]]
        predicted_graph[src][dst]["vessel_type_name"] = edge_labels_dict[str(predicted_graph[src][dst]["vessel_type"])]
    
    return predicted_graph