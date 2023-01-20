#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import networkx as nx

from arterial.feature_extraction.segment_features.utils import get_single_segments_cell_ids, get_single_segments_vessel_type, plot_single_segments

def perform_segment_feature_extraction(case_dir, centerline_graph):
    """
    Function for segment feature extraction. It performs two kinds of segment 
    feature extraction. On one hand, it extracts features for all segments with 
    a unique cell id, which are linked to a segment in the centerline_segments_array
    and the graph_simple and graph_pred objects. On the other hand, it extracts 
    features for all vessel types present in the centerline_graph, and these are
    stored as global features of the dense centerline graph.

    This function overwrites the existing centerline graphs:

    >>> case_dir/graph.pickle
    >>> case_dir/graph_simple.pickle

    Creates the following image:

    >>> case_dir/single_segments.png

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Dense centerline graph returned by graph builder.

    Returns
    -------
    centerline_graph : networkx.Graph
        Featurized centerline graph with segment features stores in 
        centerline_graph.graph["segment features"].
    
    """

    # Load simple graph
    simple_centerline_graph = nx.read_gpickle(os.path.join(case_dir, "graph_pred.pickle"))
    # Get featurized segment of the graph
    segments_cell_id = get_single_segments_cell_ids(centerline_graph)
    # Save features from segment to simple graph edges with the same cell id
    for cell_id in segments_cell_id.keys():
        for src, dst in simple_centerline_graph.edges:
            if simple_centerline_graph[src][dst]["cell_id"] == cell_id and segments_cell_id[cell_id] is not None:
                if "features" in segments_cell_id[cell_id].graph.keys():
                    simple_centerline_graph[src][dst]["segment features"] = segments_cell_id[cell_id].graph["features"]

    # Overwrite simple graph
    nx.write_gpickle(simple_centerline_graph, os.path.join(case_dir, "graph_pred.pickle"))

    # Initialize segment features dict in centerline_graph.graph
    centerline_graph.graph["segment features"] = {}
    # Extract vessel type segments
    segments_vessel_type = get_single_segments_vessel_type(centerline_graph)
    for vessel_type in segments_vessel_type.keys():
        if segments_cell_id[vessel_type] is not None:
            centerline_graph.graph["segment features"][vessel_type] = segments_cell_id[vessel_type].graph["features"]

    # Overwrite centerline graph
    nx.write_gpickle(centerline_graph, os.path.join(case_dir, "graph.pickle"))

    # Create single segments plot
    plot_single_segments(case_dir, centerline_graph, segments_vessel_type)

    return centerline_graph