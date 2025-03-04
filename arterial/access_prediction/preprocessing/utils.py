#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import networkx as nx
import numpy as np

from arterial.feature_extraction.segment_features.utils import extract_segment_features, extract_dual_segment_features

vessel_type_dict = dict(zip(["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"], np.arange(14)))

def clean_supersegment(supersegment):
    segment = nx.Graph()
    for node in supersegment:
        if supersegment.nodes[node]["is_supersegment"] > 0.5:
            segment.add_node(node)
            # Add node attributes
            for key in supersegment.nodes[node]:
                segment.nodes[node][key] = supersegment.nodes[node][key]
            segment.nodes[node]["hierarchy femoral"] = segment.nodes[node]["hierarchy"]
            segment.nodes[node]["features femoral"] = segment.nodes[node]["features"]
            for neighbor in supersegment.neighbors(node):
                if supersegment.nodes[neighbor]["is_supersegment"] > 0.5:
                    segment.add_edge(node, neighbor)
    return segment

def get_supersegment_vessel_types(supersegment):
    vessel_type_names = []
    for node in supersegment:
        if supersegment.nodes[node]["is_supersegment"] > 0.5:
            vessel_type_names.append(supersegment.nodes[node]["vessel_type_name"])
    return list(np.unique(vessel_type_names)), \
        [get_vessel_type(vessel_type_name) for vessel_type_name in np.unique(vessel_type_names)], \
            get_onehot_encoded_vessel_type(list(np.unique(vessel_type_names)))

def sort_by_hierarchy(vessel_type_counts, vessel_type_hierarchy, vessel_type_nodes):
    for vessel_type in vessel_type_hierarchy:
        vessel_type_hierarchy[vessel_type] = np.mean(vessel_type_hierarchy[vessel_type])
    # Sort by hierarchy
    vessel_type_counts = dict(sorted(vessel_type_counts.items(), key=lambda item: vessel_type_hierarchy[item[0]]))
    vessel_type_hierarchy = dict(sorted(vessel_type_hierarchy.items(), key=lambda item: item[1]))
    vessel_type_nodes = dict(sorted(vessel_type_nodes.items(), key=lambda item: item[1]))

def filter_vessel_types(vessel_type_counts, vessel_type_nodes, vessel_type_hierarchy, node_count_threshold=3):
    def get_previous_key(dictionary, current_key):
        keys = list(dictionary.keys())
        current_index = keys.index(current_key)
        return keys[current_index - 1] if current_index > 0 else None
    for vessel_type in list(vessel_type_counts.keys()):
        if vessel_type_counts[vessel_type] < node_count_threshold:
            previous_vessel_type = get_previous_key(vessel_type_counts, vessel_type)
            if previous_vessel_type:
                vessel_type_counts[previous_vessel_type] += vessel_type_counts.pop(vessel_type)
                vessel_type_nodes[previous_vessel_type].extend(vessel_type_nodes.pop(vessel_type))
                vessel_type_hierarchy.pop(vessel_type)

def compute_segment_graph_node_features(segment_graph):
    return extract_segment_features(segment_graph).graph["features"]

def compute_segment_graph_edge_features(segment_1, segment_2):
    # Get positions of the nodes with degree 1 from each segment
    segment_1_positions = np.array([segment_1.nodes[node]["pos"] for node in segment_1.nodes if segment_1.degree(node) == 1])
    segment_2_positions = np.array([segment_2.nodes[node]["pos"] for node in segment_2.nodes if segment_2.degree(node) == 1])

    # Compute the distance between all pairs of nodes
    distances = np.linalg.norm(segment_1_positions[:, None] - segment_2_positions, axis=-1)

    # Get the closest pair of nodes
    closest_pair = np.unravel_index(np.argmin(distances), distances.shape)

    # Get the mean position of the closest pair of nodes
    mean_position = np.mean([segment_1_positions[closest_pair[0]], segment_2_positions[closest_pair[1]]], axis=0)

    # Compute the features
    features = extract_dual_segment_features(segment_1, segment_2)
    features["position_x"] = mean_position[0]
    features["position_y"] = mean_position[1]
    features["position_z"] = mean_position[2]

    return features

def get_onehot_encoded_vessel_type(vessel_type_name):
    onehot_encoded_vessel_type = np.zeros(14)
    if isinstance(vessel_type_name, str):
        onehot_encoded_vessel_type[vessel_type_dict[vessel_type_name]] = 1
    elif isinstance(vessel_type_name, list):
        for vessel_type in vessel_type_name:
            onehot_encoded_vessel_type[vessel_type_dict[vessel_type]] = 1
    
    return onehot_encoded_vessel_type

def get_vessel_type(vessel_type_name):
    return vessel_type_dict[vessel_type_name]

def clean_node(node_data):
    node_data.pop("radius")
    node_data.pop("cell_id")
    node_data.pop("is_supersegment")
    node_data.pop("hierarchy femoral")
    node_data.pop("features femoral")

    for feature_to_remove in ["segment length", "direction module", "is_supersegment"]:
        node_data["features"].pop(feature_to_remove)

    features_to_rename = {
        "pos r": "position_x",
        "pos a": "position_y",
        "pos s": "position_z",
        "radius": "diameter",
        "direction polar": "polar_angle",
        "direction azimuth": "azimuthal_angle",
        "HU intensity": "hu_intensity",
        "accumulated length from access": "accumulated_length_from_access"
    }

    for old_name, new_name in features_to_rename.items():
        node_data["features"][new_name] = node_data["features"][old_name]
        node_data["features"].pop(old_name)

    ordered_features = ["position_x", "position_y", "position_z", "diameter", "polar_angle", "azimuthal_angle", \
                         "curvature", "torsion", "blanking", "hu_intensity", "accumulated_length_from_access", "vessel_type"]
    
    # Reorder features
    node_data["features"] = {key: node_data["features"][key] for key in ordered_features}

    node_data["features"]["diameter"] *= 2
    node_data["onehot_vessel_type"] = get_onehot_encoded_vessel_type(node_data["vessel_type_name"])
    node_data["features_list"] = np.concatenate(([value for value in node_data["features"].values()][:-1], node_data["onehot_vessel_type"]))

    ordered_meta_features = ["vessel_type_name", "pos", "hierarchy", "features", "vessel_type", "onehot_vessel_type", "features_list"]

    # Reorder meta features
    node_data = {key: node_data[key] for key in ordered_meta_features}
    
    return node_data