#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import networkx as nx

import pickle

import torch

from arterial.vessel_labelling.utils import node_transform, predict_vessel_types, save_predicted_graph, predict_intracranial_vessel_types

def perform_inference(case_dir, mode = "vessels"):
    """
    Performs inference of simple graph with a trained graph U-Net [1] model. 
    
    The implemented graph U-Net model is a node classification algorithm. Since in the original 
    form of the graphs, vessels are encoded by edges, we first transform the graphs into "node 
    form", where edges turn into nodes. Edges connect vessels that share bifurcations in the 
    original graph.

    Then inference is performed using the trained model.

    References:
    [1]     Gao, Hongyang, and Shuiwang Ji. 2019. "Graph U-Nets." 36th International Conference on 
    Machine Learning, ICML 2019 2019-June: 3651-60.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    if mode == "vessels":
        name_centerline_files = "vessel"
        # Load the edge form graph (graph.pickle) created at centerlineGraph.py
        with open(os.path.join(case_dir, "{}_graph_simple.pickle".format(name_centerline_files)), "rb") as f:
            graph = pickle.load(f)

        # Pass the graph to node form
        node_form_graph = node_transform(graph)
        
        # Load the trained graph U-Net model for inference
        print(os.environ["arterial_dir"])
        model_path = os.path.join(os.environ["arterial_dir"], "vessel_labelling/models/vessels/model.pth")
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=torch.device(device))

        # Perform inference with the trained model
        predicted_vessels_types = predict_vessel_types(model, node_form_graph)
    elif mode == "intracranial_vessels":
        name_centerline_files = "intracranial_vessel"
        # Load the edge form graph (graph.pickle) created at centerlineGraph.py
        with open(os.path.join(case_dir, "{}_graph_simple.pickle".format(name_centerline_files)), "rb") as f:
            graph = pickle.load(f)

        predicted_vessels_types = predict_intracranial_vessel_types(graph, case_dir)

    # Save the predicted graph in edge form as graph_pred.pickle
    save_predicted_graph(case_dir, graph, predicted_vessels_types, mode)