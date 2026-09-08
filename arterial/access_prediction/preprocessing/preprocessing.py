#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import networkx as nx

from arterial.access_prediction.preprocessing.utils import clean_supersegment, sort_by_hierarchy, filter_vessel_types, \
    compute_segment_graph_node_features, compute_segment_graph_edge_features, get_vessel_type, \
        get_onehot_encoded_vessel_type, clean_node, get_supersegment_vessel_types

def preprocess_supersegment(supersegment, access, side):
    """
    Performs preprocessing on supersegments directoly extracted from Arterial's
    feature extractor. The preprocessing uses the original supersegment, connecting 
    the femoral or radial access site (visible within the CTA) to either side of 
    the anterior circulation (aproximately connects to the ICA terminus).

    Returns a dictionary with the following keys:
    - global_features [dict]: features of the entire supersegment
    - segment_graph [nx.Graph]: graph of the supersegment, with each node representing a vessel segment
    - dense_graph [nx.Graph]: graph of the supersegment, with each node representing a local centerline point

    Parameters
    ----------
    supersegment : nx.Graph
        Supersegment extracted from Arterial
    access : str
        Access site (femoral or radial)
    side : str
        Side of the supersegment (right or left)

    Returns
    -------
    dict
        Dictionary with the global features, segment graph, and dense graph
    
    """
    # Global features
    global_features = {}
    global_features["features"] = {}
    # global_features["features"]["access"] = 0 if access == "femoral" else 1 # Currently unused, to be implemented when radial access is assessed
    global_features["features"]["side"] = 0 if side == "right" else 1
    total_segment = None

    # Segment graph
    segment_graph = None
    vessel_segments = {}

    # Dense graph
    dense_graph = None

    # Get global features
    print("Preprocessing global features...")
    total_segment = clean_supersegment(supersegment)
    vessel_type_names, vessel_types, onehot_vessel_types = get_supersegment_vessel_types(total_segment)
    total_segment.graph["vessel_type_name"] = vessel_type_names
    for node in total_segment:
        total_segment.nodes[node]["features"]["blanking"] = 0 # We remove blanking to compute features from the global segment
    total_segment.graph["features"] = compute_segment_graph_node_features(total_segment)
    total_segment.graph["features_list"] = np.concatenate(([global_features["features"]["side"]], [total_segment.graph["features"][key] for key in ["tortuosity_index", "length", "mean_diameter", "std_diameter", "min_polar_angle", "polar_angle", "azimuthal_angle"]], onehot_vessel_types))
    total_segment.graph["features"]["vessel_type"] = list(np.array(vessel_types).astype(float))
    for feature in ["tortuosity_index", "length", "mean_diameter", "std_diameter", "min_polar_angle", "polar_angle", "azimuthal_angle", "vessel_type"]:
        global_features["features"][feature] = total_segment.graph["features"][feature]
    global_features["features_list"] = list(total_segment.graph["features_list"].astype(float))

    # Get vessel segments
    vessel_type_counts = {}
    vessel_type_hierarchy = {}
    vessel_type_nodes = {}

    for node, data in total_segment.nodes(data=True):
        vessel_type = data["vessel_type_name"]
        hierarchy = data["hierarchy"]

        vessel_type_counts.setdefault(vessel_type, 0)
        vessel_type_counts[vessel_type] += 1

        vessel_type_nodes.setdefault(vessel_type, []).append(node)
        vessel_type_hierarchy.setdefault(vessel_type, []).append(hierarchy)

    for vessel_type, hierarchies in vessel_type_hierarchy.items():
        vessel_type_hierarchy[vessel_type] = np.mean(hierarchies)

    sort_by_hierarchy(vessel_type_counts, vessel_type_hierarchy, vessel_type_nodes)
    filter_vessel_types(vessel_type_counts, vessel_type_nodes, vessel_type_hierarchy)

    for vessel_type in vessel_type_nodes:
        vessel_segments[vessel_type] = total_segment.subgraph(vessel_type_nodes[vessel_type]).copy()
        vessel_segments[vessel_type].graph["vessel_type_name"] = vessel_type
        vessel_segments[vessel_type].graph["pos"] = np.mean([vessel_segments[vessel_type].nodes[node]["pos"] for node in vessel_segments[vessel_type]], axis=0)
        vessel_segments[vessel_type].graph["hierarchy"] = vessel_type_hierarchy[vessel_type]
        vessel_segments[vessel_type].graph["vesssel_type"] = get_vessel_type(vessel_type)
        vessel_segments[vessel_type].graph["onehot_vessel_type"] = get_onehot_encoded_vessel_type(vessel_type)

    # Build segment graph
    print("Building segment graph for supersegment...")
    segment_graph = nx.Graph()

    node = 0
    for vessel_type in vessel_segments:
        segment_graph.add_node(node)
        segment_graph.nodes[node]["vessel_type_name"] = vessel_type
        segment_graph.nodes[node]["pos"] = vessel_segments[vessel_type].graph["pos"]
        segment_graph.nodes[node]["hierarchy"] = vessel_segments[vessel_type].graph["hierarchy"]
        segment_graph.nodes[node]["features"] = compute_segment_graph_node_features(vessel_segments[vessel_type])
        for feature_to_remove in ["min_diameter", "max_diameter", "proximal_diameter", "distal_diameter", "min_max_diameter_ratio", \
                                    "bending_length", "cumulative_curvature", "tortuosity_index_5_cm", "accumulated_polar_angle_differential"]:
            segment_graph.nodes[node]["features"].pop(feature_to_remove)
        segment_graph.nodes[node]["features"]["vessel_type"] = get_vessel_type(vessel_type)
        segment_graph.nodes[node]["vessel_type"] = get_vessel_type(vessel_type)
        segment_graph.nodes[node]["onehot_vessel_type"] = get_onehot_encoded_vessel_type(vessel_type).astype(int)
        segment_graph.nodes[node]["features_list"] = np.concatenate(([value for value in segment_graph.nodes[node]["features"].values()][:-1], segment_graph.nodes[node]["onehot_vessel_type"]))

        segment_graph.nodes[node]["subgraph"] = vessel_segments[vessel_type]

        if node > 0:
            segment_graph.add_edge(node - 1, node)
            segment_graph[node - 1][node]["features"] = compute_segment_graph_edge_features(segment_graph.nodes[node - 1]["subgraph"], segment_graph.nodes[node]["subgraph"])
            segment_graph[node - 1][node]["features_list"] = [value for value in segment_graph[node - 1][node]["features"].values()]

        node += 1

    # Build dense graph
    print("Building dense graph for supersegment...")
    dense_graph = clean_supersegment(supersegment)
    for node in dense_graph:
        clean_node_data = clean_node(dense_graph.nodes[node])
        # Pop everything from the original node
        for key in dense_graph.nodes[node].copy():
            dense_graph.nodes[node].pop(key)
        for key, value in clean_node_data.items():
            nx.set_node_attributes(dense_graph, {node: value}, key)

    return {
        "global_features": global_features,
        "segment_graph": segment_graph,
        "dense_graph": dense_graph
    }

