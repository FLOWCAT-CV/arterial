#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

import os

from arterial.feature_extraction.segment_features.utils import get_and_featurize_single_segments_cell_ids, get_and_featurize_single_segments_vessel_types, plot_single_segments

def perform_segment_feature_extraction(local_graph, segments_graph):
    """
    Function for segment feature extraction. It performs two kinds of segment 
    feature extraction. On one hand, it extracts features for all segments with 
    a unique cell id, which are linked to a segment in the centerline_segments_array
    and the graph_simple and graph_pred objects. On the other hand, it extracts 
    features for all vessel types present in the local_graph, and these are
    stored as global features of the dense centerline graph.

    Parameters
    ----------
    local_graph : networkx.Graph
        Dense centerline graph returned by graph builder.
    segments_graph : networkx.Graph
        Centerline graph from centerline_segments_array, with vessel types predicted
        by vessel labeller.

    Returns
    -------
    local_graph : networkx.Graph
        Featurized centerline graph with segment features stores in 
        local_graph.graph["segment features"].
    segments_graph : networkx.Graph
        Centerline graph with segment features stored in segments_graph.edges.
    segments_vessel_type_dict : dict
        Dictionary with vessel types as keys and centerline graphs as values.
    
    """
    # Get featurized segment of the graph
    segments_cell_id = get_and_featurize_single_segments_cell_ids(local_graph)
    # Save features from segment to simple graph edges with the same cell id
    for cell_id in segments_cell_id.keys():
        for src, dst in segments_graph.edges:
            if segments_graph[src][dst]["cell_id"] == cell_id and segments_cell_id[cell_id] is not None:
                if "features" in segments_cell_id[cell_id].graph.keys():
                    segments_graph[src][dst]["segment_features"] = segments_cell_id[cell_id].graph["features"]

    # Initialize segment features dict in local_graph.graph
    local_graph.graph["segment_features"] = {}
    # Extract vessel type segments
    segments_vessel_type_dict = get_and_featurize_single_segments_vessel_types(local_graph)
    for vessel_type in segments_vessel_type_dict.keys():
        if segments_vessel_type_dict[vessel_type] is not None:
            local_graph.graph["segment_features"][vessel_type] = segments_vessel_type_dict[vessel_type].graph["features"]

    return local_graph, segments_graph, segments_vessel_type_dict