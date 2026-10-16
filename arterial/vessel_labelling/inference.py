#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToDevice
from torch_geometric.nn.models import GAT, GraphSAGE

from arterial import model_registry
from arterial.vessel_labelling.utils import EVCDatasetInference, get_edge_transforms
from arterial.vessel_labelling.models import ResGAT
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

def load_vessel_labelling_model(checkpoint_path, device):
    """
    Loads a vessel labelling model and the graph construction it was trained with.

    Checkpoints are dicts with "state_dict" and "init_kwargs". Optional keys describe how the
    model was trained: "model_class" ("GAT", "ResGAT" or "GraphSAGE"), "edge_mode" (see
    utils.get_edge_transforms) and "radius_mm". Checkpoints without them are the original
    weights (GATv2 trained with the "radius_only" edge mode), so these are the defaults.

    Parameters
    ----------
    checkpoint_path : string or path-like object
        Path to the checkpoint (model_weights.pth).
    device : torch.device
        Device to load the model on.

    Returns
    -------
    model : torch.nn.Module
        Model in evaluation mode.
    edge_config : dict
        Dict with the "edge_mode" and "radius_mm" the model was trained with.

    """
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    init_kwargs = state_dict["init_kwargs"]
    model_class = state_dict.get("model_class", "GAT")
    if model_class == "GAT":
        model = GAT(
            in_channels = init_kwargs["in_channels"],
            hidden_channels = init_kwargs["hidden_channels"],
            out_channels = init_kwargs["out_channels"],
            num_layers = init_kwargs["num_layers"],
            v2=True,
            act = init_kwargs["act"]
        ).to(device)
        # Original checkpoints are loaded non-strictly, as before
        model.load_state_dict(state_dict["state_dict"], strict=False)
    elif model_class == "ResGAT":
        model = ResGAT(
            in_channels = init_kwargs["in_channels"],
            hidden_channels = init_kwargs["hidden_channels"],
            out_channels = init_kwargs["out_channels"],
            num_layers = init_kwargs["num_layers"],
            act = init_kwargs["act"]
        ).to(device)
        model.load_state_dict(state_dict["state_dict"], strict=True)
    elif model_class == "GraphSAGE":
        model = GraphSAGE(
            in_channels = init_kwargs["in_channels"],
            hidden_channels = init_kwargs["hidden_channels"],
            out_channels = init_kwargs["out_channels"],
            num_layers = init_kwargs["num_layers"],
            act = init_kwargs["act"]
        ).to(device)
        model.load_state_dict(state_dict["state_dict"], strict=True)
    else:
        raise ValueError(f"Unknown model_class {model_class} in {checkpoint_path}.")
    model.eval()
    edge_config = {"edge_mode": state_dict.get("edge_mode", "radius_only"), "radius_mm": state_dict.get("radius_mm", 25.0)}
    return model, edge_config

def get_pre_transform(edge_config, device):
    """
    Builds the pre-transform for the edge construction a model was trained with.

    Parameters
    ----------
    edge_config : dict
        Dict with "edge_mode" and "radius_mm" (see load_vessel_labelling_model).
    device : torch.device
        Device to move the graph to.

    Returns
    -------
    pre_transform : torch_geometric.transforms.Compose
        Pre-transform.

    """
    return Compose(get_edge_transforms(edge_config["edge_mode"], edge_config["radius_mm"]) + [ToDevice(device)])

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
    # Load the trained model for inference, with the edge construction it was trained with
    model, edge_config = load_vessel_labelling_model(model_registry.model_path("vessel_labelling", "extracranial_vessels", "model_weights.pth"), device)
    # Load file in the dataset class
    data = EVCDatasetInference(raw_graph=graph, pre_transform=get_pre_transform(edge_config, device))
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    data_loader = DataLoader(data, batch_size = 1, shuffle = False) 
    # We have to iterate over the DataLoader (even thogh it will just be one graph at the time)
    with torch.no_grad():
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
    dataset_description = load_json(model_registry.model_path("vessel_labelling", "extracranial_vessels", "dataset.json"))
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
    # Load the 5 fold models. All folds must share the edge construction they were trained with
    models, edge_configs = zip(*[
        load_vessel_labelling_model(model_registry.model_path("vessel_labelling", "extracranial_vessels", f"fold_{fold}", "model_weights.pth"), device)
        for fold in range(5)
    ])
    if any(edge_config != edge_configs[0] for edge_config in edge_configs):
        raise ValueError(f"Fold models were trained with different edge constructions: {edge_configs}.")
    # Load file in the dataset class
    data = EVCDatasetInference(raw_graph=graph, pre_transform=get_pre_transform(edge_configs[0], device))
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    data_loader = DataLoader(data, batch_size = 1, shuffle = False) 

    # Get edge type dict from the dataset.json
    dataset_description = load_json(model_registry.model_path("vessel_labelling", "extracranial_vessels", "dataset.json"))

    # create tensor of size of length(graph_preprocessed.x), with numi_classes number of channels, and depth 5 for folds
    # We will average logits across folds and then argmax to choose final class
    predicted_nodes = torch.zeros(len(next(iter(data_loader)).x), dataset_description["num_edge_classes"], 5).to(device)

    with torch.no_grad():
        for fold, model in enumerate(models):
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