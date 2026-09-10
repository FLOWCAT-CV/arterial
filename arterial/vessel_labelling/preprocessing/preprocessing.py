#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import numpy as np
import networkx as nx

from arterial.vessel_labelling.preprocessing.utils import extract_features_for_labelling

def build_nx_graph_from_segments_array(centerline_segments_array):
    """
    Creates and featurizes simple centerline graph for vessel labelling.
    
    Saves graph and image as:

    >>> case_dir/simple_graph.pickle
    >>> case_dir/simple_graph.png

    Parmeters
    ---------
    case_dir : string or path-like object
        Path to case directory. 
    mode : string, optional
        Mode of the vessel labeller. The default is "extracranial_vessels", it can also be "intracranial_vessels".

    Returns
    -------

    """
    # Get coordinates array from centerline_segments_array
    coordinate_array = centerline_segments_array[:, 0]
    # Get radius array from centerline_segments_array
    radius_array = centerline_segments_array[:, 1]

    # Initialize simplified graph with networkx
    simple_centerline_graph = nx.Graph()
    # Only taking first and last positions of the curves arrays
    total_nodes = 0 # We only link nodes from the same centerline
    for cell_id in range(len(coordinate_array)):
        if len(coordinate_array[cell_id]) > 1 and len(radius_array[cell_id]) > 1:
            # Add nodes
            # First node of cell (startpoint)
            simple_centerline_graph.add_node(total_nodes + 0, pos=coordinate_array[cell_id][0])
            simple_centerline_graph.nodes[total_nodes + 0]["radius"] = radius_array[cell_id][0]
            # Last node of cell (endpoint)
            simple_centerline_graph.add_node(total_nodes + 1, pos=coordinate_array[cell_id][-1])
            simple_centerline_graph.nodes[total_nodes + 1]["radius"] = radius_array[cell_id][-1]

            # Add edges
            simple_centerline_graph.add_edge(total_nodes, total_nodes + 1, cell_id = cell_id)
            simple_centerline_graph[total_nodes][total_nodes + 1]["coordinate_array"] = coordinate_array[cell_id]
            simple_centerline_graph[total_nodes][total_nodes + 1]["radius_array"] = radius_array[cell_id]

            total_nodes += 2

    # Merge nodes that share the same RAS coordinate (bifurcation spots)
    # First get all nodes that have a degree of 1 (start- and endpoints)
    deg_one_nodes = []
    for node, deg in simple_centerline_graph.degree:
        if deg == 1:
            deg_one_nodes.append(node)
            
    # For all degree 1 nodes, we check position to join corresponding start- and enpoints, as well as bifurcations
    removed_nodes = []
    for _, node in enumerate(deg_one_nodes):
        if node not in removed_nodes:
            aux = deg_one_nodes.copy()
            aux.remove(node)
            for auxNodes in removed_nodes:
                aux.remove(auxNodes)
            for _, node2 in enumerate(aux):
                if np.linalg.norm(simple_centerline_graph.nodes[node]["pos"] - simple_centerline_graph.nodes[node2]["pos"]) < 1e-6:
                    simple_centerline_graph = nx.contracted_nodes(simple_centerline_graph, node, node2)
                    removed_nodes.append(node2)
                    simple_centerline_graph.nodes[node].pop("contraction")

    # Relabel nodes as sequential labels
    mapping = {}
    new_node = 0
    for old_node in simple_centerline_graph.nodes():
        mapping[old_node] = new_node
        new_node += 1
    simple_centerline_graph = nx.relabel.relabel_nodes(simple_centerline_graph, mapping)

    # Featurizes simple_centerline_graph
    simple_centerline_graph = extract_features_for_labelling(simple_centerline_graph)

    return simple_centerline_graph