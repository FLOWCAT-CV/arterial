#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose, ToDevice
from torch_geometric.data import Data, Batch

from arterial.access_prediction.utils import ArterialGNetDatasetInference, DenseRadiusGraph, collate_ArterialGNetInference, build_final_attention_map

def perform_inference(preprocessed_supersegment_dict, lpi_corner_coordinates, return_attention_map=True):
    """
    Performs inference of access prediction with a trained ArterialGNet. Assumes that model returns both
    prediction and attention map representable as a dense graph. Predictions are added to the attention map
    graph as global attributes.

    Parameters
    ----------
    preprocessed_supersegment_dict : dict
        Dictionary with preprocessed supersegments (keys: global_features, segment_graph, dense_graph)
    cta_nifti : nibabel.Nifti1Image
        CTA image in Nifti format. Need to get lpi corner coordinates. 
    """
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define pre-tranforme
    pre_transform = Compose([
        DenseRadiusGraph(r=10, loop=True),
        ToDevice(device)
    ])
    # Load file in the dataset class
    data = ArterialGNetDatasetInference(preprocessed_supersegment_dict=preprocessed_supersegment_dict, pre_transform=pre_transform)
    # Load graph to the DataLoader through the ArterialDatasetInference class, with the inference transforms
    data_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_ArterialGNetInference) 
    preprocessed_supersegment = next(iter(data_loader))

    # Initialize output tensor
    pred_list = []
    if not return_attention_map: # Assumes model does not return attention maps (model output is only prediction)
        with torch.no_grad():
            for fold in range(5):
                # Load the trained model for inference
                model = torch.load(os.path.join(os.environ["arterial_dir"], f"access_prediction/models/fold_{fold}/model_latest.pth"), map_location=torch.device(device)).to(device)
                # Load model to device
                model.eval()
                # Perform inference (probabilities for each fold)
                out = model(preprocessed_supersegment)
                pred_list.append(out[0][1].item())

        # Get mean and std of the predictions
        prediction = np.mean(pred_list, axis=0)
        prediction_std = np.std(pred_list, axis=0)

        return prediction, prediction_std, None
    
    else:
        # Initialize attention map tensor. 8 is the number of attention heads of the GAT operator (hardcoded here)
        edge_attention_weights = torch.zeros(preprocessed_supersegment.dense_data.edge_index.shape[1], 8, 5, device="cpu")
        with torch.no_grad():
            for fold in range(5):
                # Load the trained model for inference
                model = torch.load(os.path.join(os.environ["arterial_dir"], f"access_prediction/models/fold_{fold}/model_latest.pth"), map_location=torch.device(device)).to(device)
                # Load model to device
                model.eval()
                # Perform inference (probabilities and attention map for each fold)
                out, attention_weights = model(preprocessed_supersegment)
                pred_list.append(out[0][1].item())
                edge_indices, edge_attention_weights_ = attention_weights
                edge_attention_weights[:, :, fold] = edge_attention_weights_


        # Get mean and std of the predictions
        prediction = np.mean(pred_list, axis=0)
        prediction_std = np.std(pred_list, axis=0)

        # Average attention maps across folds and attention heads
        edge_indices = edge_indices.detach().cpu().numpy()
        edge_attention_weights = edge_attention_weights.mean(axis=2).detach().cpu().numpy()
        attention_map_graph = build_final_attention_map(edge_indices, edge_attention_weights, preprocessed_supersegment, lpi_corner_coordinates)
        
        # Add prediction and prediction_std to the graph
        attention_map_graph.graph["prediction"] = prediction
        attention_map_graph.graph["prediction_std"] = prediction_std

        return prediction, prediction_std, attention_map_graph