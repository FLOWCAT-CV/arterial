#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import networkx as nx

from arterial.feature_extraction.utils import resample_centerline_segments_array, get_predicted_vessels_dict, get_hierarchical_order, unify_subgraphs, sanity_check_for_random_islands, build_segments_array_for_individual_centerline_graph

def build_local_graph(centerline_segments_array, segments_graph_pred, sampling_distance_mm=2):
    """
    Builds dense centerline graph from case_dir/centerline_segments_array.npy and 
    case_dir/graph_pred.pickle. Samples node every `sampling_distance_mm` milimiters,
    creates subgraphs for every separate centerine model and unifies them depending on 
    the predicted vessel types.

    Parameters
    ----------
    centerline_segments_array : numpy.ndarray
        Array containing centerline segments with shape (n_cells, 2), where the first
        column contains the coordinates of the centerline points and the second column
        contains the radius of the centerline points.
    segments_graph_pred : networkx.Graph
        Graph containing the predicted vessel types for each centerline segment.
    sampling_distance_mm : int, optional
        Distance in milimiters to sample a node in the centerline graph. The default is 2.

    Returns
    -------
    local_graph : networkx.Graph
        Centerline graph containing with a node sampled at every `sampling_distance_mm` milimiters,
        with ordered indices inidicating catheterization direction from different femoral and radial
        accesses.

    """
    # We initialize two variables for the sanity check for unwanted segmented islands
    sanity_check = False
    skip_cell_ids = []
    while not sanity_check:
        # Get coordinates array from centerline_segments_array
        centerline_coordinate_array = centerline_segments_array[:, 0]
        # Get radius array from centerline_segments_array
        centerline_radius_array = centerline_segments_array[:, 1]
        # Predicted vessel types and vessel type names
        predicted_vessel_types, predicted_vessel_type_names = get_predicted_vessels_dict(segments_graph_pred)
        # If there are no AA vessels in predicted_vessel_types, we raise an error
        if "AA" not in predicted_vessel_type_names.values():
            raise ValueError("No AA vessels were predicted. Feature extraction will not be reliable.")

        # Initialize graph with networkx
        local_graph = nx.Graph()
        # Add predicted vessel type dicts to global attributes
        local_graph.graph["predicted_vessel_types"] = predicted_vessel_types
        local_graph.graph["predicted_vessel_type_names"] = predicted_vessel_type_names
        # Store centerline_segments_array in global features
        local_graph.graph["centerline_segments_array"] = centerline_segments_array
        # Building dense graph
        total_nodes = 0 
        # We register the highest and lowest s coordinates of the nodes (initialization)
        lowest_s_pos = 1e10
        highest_s_pos = -1e10
        # We only link nodes from the same centerline first, and afterwards we contract nodes with the same position
        for cell_id, curve in enumerate(centerline_coordinate_array):
            if cell_id not in skip_cell_ids and cell_id in predicted_vessel_types:
                # Initialize distance for node sampling
                distance = 0
                # We keep track of last node 
                previous_idx = 0
                # Now we loop to every centerline point in each centerline_segments_array cell
                for idx, position in enumerate(curve):
                    # We register the highest and lowest s coordinates of the nodes
                    if position[2] < lowest_s_pos:
                        lowest_s_pos = position[2]
                    if position[2] > highest_s_pos:
                        highest_s_pos = position[2]
                    # For the first node in every cell, we add just a node with its corresponding cell_id
                    if idx == 0:
                        local_graph.add_node(total_nodes, pos = position)
                        local_graph.nodes[total_nodes]["radius"] = centerline_radius_array[cell_id][idx]
                        local_graph.nodes[total_nodes]["cell_id"] = cell_id
                        local_graph.nodes[total_nodes]["vessel_type"] = predicted_vessel_types[cell_id]
                        local_graph.nodes[total_nodes]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                        total_nodes += 1
                        previous_position = position
                    # If distance from last sampled node is larger than selected sampling_distance_mm, add a node with cell_id and an edge to the previous sample 
                    # node, keeping cell_id and the indices of the centerline points in the centerline_segments_array
                    # elif distance >= sampling_distance_mm:
                    elif idx < len(curve) - 1:
                        local_graph.add_node(total_nodes, pos = position)
                        local_graph.nodes[total_nodes]["radius"] = centerline_radius_array[cell_id][idx]
                        local_graph.nodes[total_nodes]["cell_id"] = cell_id
                        local_graph.nodes[total_nodes]["vessel_type"] = predicted_vessel_types[cell_id]
                        local_graph.nodes[total_nodes]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                        local_graph.add_edge(total_nodes - 1, total_nodes)
                        local_graph[total_nodes - 1][total_nodes]["cell_id"] = cell_id
                        local_graph[total_nodes - 1][total_nodes]["vessel_type"] = predicted_vessel_types[cell_id]
                        local_graph[total_nodes - 1][total_nodes]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                        local_graph[total_nodes - 1][total_nodes]["indices"] = np.arange(previous_idx, idx)
                        local_graph[total_nodes - 1][total_nodes]["centerline_coordinate_array"] = centerline_coordinate_array[cell_id][local_graph[total_nodes - 1][total_nodes]["indices"]]
                        local_graph[total_nodes - 1][total_nodes]["centerline_radius_array"] = centerline_radius_array[cell_id][local_graph[total_nodes - 1][total_nodes]["indices"]]
                        total_nodes += 1
                        previous_position = position
                        # We alse reinitialize the accumulated distance and update the prior node previous_idx
                        distance = 0
                        previous_idx = idx
                    # For the last point of every centerline cell (presumably at a distance smaller than sampling_distance_mm) we differentiate between two possible cases
                    elif idx == len(curve) - 1:
                        # If the number of nodes of the previous node of the cell is 0 (which means that the segment's length is smaller than sampling_distance_mm), we add an additional node
                        if len([node for node in local_graph.neighbors(total_nodes - 1)]) == 0:
                            local_graph.add_node(total_nodes, pos = position)
                            local_graph.nodes[total_nodes]["radius"] = centerline_radius_array[cell_id][idx]
                            local_graph.nodes[total_nodes]["cell_id"] = cell_id
                            local_graph.nodes[total_nodes]["vessel_type"] = predicted_vessel_types[cell_id]
                            local_graph.nodes[total_nodes]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                            local_graph.add_edge(total_nodes - 1, total_nodes)
                            local_graph[total_nodes - 1][total_nodes]["cell_id"] = cell_id
                            local_graph[total_nodes - 1][total_nodes]["vessel_type"] = predicted_vessel_types[cell_id]
                            local_graph[total_nodes - 1][total_nodes]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                            local_graph[total_nodes - 1][total_nodes]["indices"] = np.arange(previous_idx, idx)
                            local_graph[total_nodes - 1][total_nodes]["centerline_coordinate_array"] = centerline_coordinate_array[cell_id][local_graph[total_nodes - 1][total_nodes]["indices"]]
                            local_graph[total_nodes - 1][total_nodes]["centerline_radius_array"] = centerline_radius_array[cell_id][local_graph[total_nodes - 1][total_nodes]["indices"]]
                            total_nodes += 1
                            previous_position = position
                        # If it is not, which will be the general case, we do not add a new node, but instead we transform the last added node and change its position to be placed at the bifurcation/endpoint
                        else:
                            local_graph.nodes[total_nodes - 1]["pos"] = position
                            local_graph.nodes[total_nodes - 1]["radius"] = centerline_radius_array[cell_id][idx]
                            local_graph.nodes[total_nodes - 1]["cell_id"] = cell_id
                            local_graph.nodes[total_nodes - 1]["vessel_type"] = predicted_vessel_types[cell_id]
                            local_graph.nodes[total_nodes - 1]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                            local_graph[total_nodes - 2][total_nodes - 1]["cell_id"] = cell_id
                            local_graph[total_nodes - 2][total_nodes - 1]["vessel_type"] = predicted_vessel_types[cell_id]
                            local_graph[total_nodes - 2][total_nodes - 1]["vessel_type_name"] = predicted_vessel_type_names[cell_id]
                            local_graph[total_nodes - 2][total_nodes - 1]["indices"] = np.append(local_graph[total_nodes - 2][total_nodes - 1]["indices"], np.arange(previous_idx, idx))
                            local_graph[total_nodes - 2][total_nodes - 1]["centerline_coordinate_array"] = centerline_coordinate_array[cell_id][local_graph[total_nodes - 2][total_nodes - 1]["indices"]]
                            local_graph[total_nodes - 2][total_nodes - 1]["centerline_radius_array"] = centerline_radius_array[cell_id][local_graph[total_nodes - 2][total_nodes - 1]["indices"]]
                            previous_position = position
                    # If the accumulated distance from the last sample node is smaller than sampling_distance_mm, and we are not in either the first or last nodes of the centerline_segments_array cell, just update the distance
                    else:
                        distance += np.linalg.norm(previous_position - position)
                        previous_position = position
                    
        # Merge nodes that share the same coordinate (bifurcation spots)
        # First get all nodes that have a degree of 1 (start- and endpoints)
        deg_one_nodes = []
        for node, deg in local_graph.degree:
            if deg == 1:
                deg_one_nodes.append(node)

        # For all degree 1 nodes, we check position to join corresponding startpoints and endpoints
        contracted_nodes = []
        for _, node in enumerate(deg_one_nodes):
            if node not in contracted_nodes:
                aux = deg_one_nodes.copy()
                aux.remove(node)
                for aux_nodes in contracted_nodes:
                    aux.remove(aux_nodes)
                for _, node2 in enumerate(aux):
                    coordinate_1 = local_graph.nodes[node]["pos"]
                    coordinate_2 = local_graph.nodes[node2]["pos"]
                    if np.linalg.norm(coordinate_1 - coordinate_2) < 1e-2:
                        local_graph = nx.contracted_nodes(local_graph, node, node2)
                        contracted_nodes.append(node2)
                        local_graph.nodes[node].pop("contraction")
                        
        # If any nodes with degree == 0 are present, remove them
        remove_nodes = []
        for node in local_graph:
            if local_graph.degree(node) == 0:
                remove_nodes.append(node)
        for node in remove_nodes:
            local_graph.remove_node(node)

        # Relabel nodes as sequential labels
        mapping = {}
        new_node = 0
        for old_node in local_graph.nodes():
            mapping[old_node] = new_node
            new_node += 1
        local_graph = nx.relabel.relabel_nodes(local_graph, mapping)

        # Divide the dense graph into disconnected subgraphs
        subgraphs = [local_graph.subgraph(components) for components in nx.connected_components(local_graph)]
        # Remove short (less than 5 nodes) subgraphs (do not introduce much information and can easily corrupt feature extraction)
        if len(subgraphs) > 1:
            delete_subgraphs = []
            for idx, subgraph in enumerate(subgraphs):
                if len(subgraph) < 5:
                    print("Removing subgraph {} with length {}".format(idx, len(subgraphs)))
                    delete_subgraphs.append(idx)
                    subgraphs.remove(subgraph)
            # Performs sanity check in case there are multiple significant subgraphs
            sanity_check, skip_cell_ids = sanity_check_for_random_islands(subgraphs, skip_cell_ids)
        else:
            sanity_check = True
    else:
        # Initialize rightmost node
        rightmost_node = 0
        # Once we go out of the while loop, we compute the range of the s coordinate
        # We limit the rightmost node search to the lowest 60% of the image
        range_pos = highest_s_pos - lowest_s_pos
        # The rightmost_node node can be used for hierarchical indexing from radial access                          
        rightmost_node_position = centerline_coordinate_array[0][0]
        for node in local_graph:
            if local_graph.nodes[node]["pos"][0] > rightmost_node_position[0] and local_graph.degree(node) == 1 and local_graph.nodes[node]["pos"][2] < range_pos * 0.6 + lowest_s_pos:
                rightmost_node = node
                rightmost_node_position = local_graph.nodes[node]["pos"]

        # Store rightmost in global attributes
        local_graph.graph["rightmost"] = rightmost_node
        
        # Loop over the subgraphs and get hierarchical indexing for each (separately). We have to divide the graphs into subgraphs for the hierarchical indexing to be 
        # applied properly, so that graph unification can be performed
        for idx, subgraph in enumerate(subgraphs):
            start_node = [node for node in subgraph][0]
            # # For the start_node of disconnected subgraphs, get the node with the lowest S coordinate for hierarchical ordering (it is only an approximation for vessel labelling, not very relevant)
            if idx > 0:
                min_s = subgraph.nodes[start_node]["pos"][2]
                for node in subgraphs[idx]:
                    if subgraph.nodes[node]["pos"][2] < min_s and subgraph.degree(node) == 1:
                        min_s = subgraph.nodes[node]["pos"][2]
                        start_node = node
            subgraph = get_hierarchical_order(subgraph, "femoral", start_node)
            subgraphs[idx] = subgraph

        # Perform graph unification
        local_graph = unify_subgraphs(centerline_segments_array, local_graph, subgraphs)
        # Remove nodes with degree 0
        remove_nodes = []
        for node in local_graph:
            if local_graph.degree(node) == 0:
                remove_nodes.append(node)
        for node in remove_nodes:
            local_graph.remove_node(node)

        # Get hierarchical order again after graph unification from both accessess
        local_graph = get_hierarchical_order(local_graph, "femoral", 0)
        local_graph = get_hierarchical_order(local_graph, "radial", local_graph.graph["rightmost"])    

        return local_graph
    
def build_individual_centerline_graph_from_vtkpolydata(centerline_model, affine, image_shape, sampling_distance_mm=0.5, radius_array_name="MaximumInscribedSphereRadius", centerline_id=None):
    """
    Builds networkx graph from a vtkPolyData object containing a centerline model corresponding 
    to a single vascular segment. This is build in coherence with the 
    CenterlineExtractor.extract_centerline_between_endpoints function.

    Parameters
    ----------
    centerline_model : vtkPolyData
        vtkPolyData object containing a centerline model corresponding 
    sampling_distance_mm : int, optional
        Distance in milimiters to sample a node in the centerline graph. The default is 2.

    Returns
    -------
    individual_centerline_graph : networkx.Graph
        Networkx graph containing the centerline model with a node sampled at every `sampling_distance_mm` milimiters.
        No ordering is performed, assumes the centerline model is already ordered.

    """
    centerline_segments_array = build_segments_array_for_individual_centerline_graph(centerline_model, affine, image_shape, radius_array_name)
    centerline_segments_array = resample_centerline_segments_array(centerline_segments_array, sampling_distance_mm)

    # Extract coordinate and radius arrays from the segments array
    # centerline_segments_array has shape (n_segments, 2) with dtype=object
    # [:, 0] contains coordinate arrays of shape (n_points, 3)
    # [:, 1] contains radius arrays of shape (n_points,)
    centerline_coordinate_array = centerline_segments_array[:, 0]
    centerline_radius_array = centerline_segments_array[:, 1]

    # Initialize graph with networkx
    individual_centerline_graph = nx.Graph()
    # Add predicted vessel type dicts to global attributes
    individual_centerline_graph.graph["predicted_vessel_types"] = None
    individual_centerline_graph.graph["predicted_vessel_type_names"] = None
    # Store centerline_segments_array in global features
    individual_centerline_graph.graph["centerline_segments_array"] = centerline_segments_array
    # Building dense graph
    total_nodes = 0 
    # We only link nodes from the same centerline first, and afterwards we contract nodes with the same position
    for cell_id, curve in enumerate(centerline_coordinate_array):
        # Initialize distance for node sampling
        distance = 0
        # We keep track of last node 
        previous_idx = 0
        # Now we loop to every centerline point in each centerline_segments_array cell
        for idx, position in enumerate(curve):
            # For the first node in every cell, we add just a node with its corresponding cell_id
            if idx == 0:
                individual_centerline_graph.add_node(total_nodes, pos = position)
                individual_centerline_graph.nodes[total_nodes]["radius"] = centerline_radius_array[cell_id][idx]
                individual_centerline_graph.nodes[total_nodes]["cell_id"] = cell_id
                individual_centerline_graph.nodes[total_nodes]["vessel_type"] = None
                individual_centerline_graph.nodes[total_nodes]["vessel_type_name"] = centerline_id
                total_nodes += 1
                previous_position = position
            # If distance from last sampled node is larger than selected sampling_distance_mm, add a node with cell_id and an edge to the previous sample 
            # node, keeping cell_id and the indices of the centerline points in the centerline_segments_array
            elif idx < len(curve) - 1:
            # elif distance > sampling_distance_mm:
                individual_centerline_graph.add_node(total_nodes, pos = position)
                individual_centerline_graph.nodes[total_nodes]["radius"] = centerline_radius_array[cell_id][idx]
                individual_centerline_graph.nodes[total_nodes]["cell_id"] = cell_id
                individual_centerline_graph.nodes[total_nodes]["vessel_type"] = None
                individual_centerline_graph.nodes[total_nodes]["vessel_type_name"] = centerline_id
                individual_centerline_graph.add_edge(total_nodes - 1, total_nodes)
                individual_centerline_graph[total_nodes - 1][total_nodes]["cell_id"] = cell_id
                individual_centerline_graph[total_nodes - 1][total_nodes]["vessel_type"] = None
                individual_centerline_graph[total_nodes - 1][total_nodes]["vessel_type_name"] = centerline_id
                individual_centerline_graph[total_nodes - 1][total_nodes]["indices"] = np.arange(previous_idx, idx)
                individual_centerline_graph[total_nodes - 1][total_nodes]["centerline_coordinate_array"] = centerline_coordinate_array[cell_id][individual_centerline_graph[total_nodes - 1][total_nodes]["indices"]]
                individual_centerline_graph[total_nodes - 1][total_nodes]["centerline_radius_array"] = centerline_radius_array[cell_id][individual_centerline_graph[total_nodes - 1][total_nodes]["indices"]]
                total_nodes += 1
                previous_position = position
                # We alse reinitialize the accumulated distance and update the prior node previous_idx
                distance = 0
                previous_idx = idx
            # For the last point of every centerline cell (presumably at a distance smaller than sampling_distance_mm) we differentiate between two possible cases
            elif idx == len(curve) - 1:
                # If the number of nodes of the previous node of the cell is 0 (which means that the segment's length is smaller than sampling_distance_mm), we add an additional node
                if len([node for node in individual_centerline_graph.neighbors(total_nodes - 1)]) == 0:
                    individual_centerline_graph.add_node(total_nodes, pos = position)
                    individual_centerline_graph.nodes[total_nodes]["radius"] = centerline_radius_array[cell_id][idx]
                    individual_centerline_graph.nodes[total_nodes]["cell_id"] = cell_id
                    individual_centerline_graph.nodes[total_nodes]["vessel_type"] = None
                    individual_centerline_graph.nodes[total_nodes]["vessel_type_name"] = centerline_id
                    individual_centerline_graph.add_edge(total_nodes - 1, total_nodes)
                    individual_centerline_graph[total_nodes - 1][total_nodes]["cell_id"] = cell_id
                    individual_centerline_graph[total_nodes - 1][total_nodes]["vessel_type"] = None
                    individual_centerline_graph[total_nodes - 1][total_nodes]["vessel_type_name"] = centerline_id
                    individual_centerline_graph[total_nodes - 1][total_nodes]["indices"] = np.arange(previous_idx, idx)
                    individual_centerline_graph[total_nodes - 1][total_nodes]["centerline_coordinate_array"] = centerline_coordinate_array[cell_id][individual_centerline_graph[total_nodes - 1][total_nodes]["indices"]]
                    individual_centerline_graph[total_nodes - 1][total_nodes]["centerline_radius_array"] = centerline_radius_array[cell_id][individual_centerline_graph[total_nodes - 1][total_nodes]["indices"]]
                    total_nodes += 1
                    previous_position = position
                # If it is not, which will be the general case, we do not add a new node, but instead we transform the last added node and change its position to be placed at the bifurcation/endpoint
                else:
                    individual_centerline_graph.nodes[total_nodes - 1]["pos"] = position
                    individual_centerline_graph.nodes[total_nodes - 1]["radius"] = centerline_radius_array[cell_id][idx]
                    individual_centerline_graph.nodes[total_nodes - 1]["cell_id"] = cell_id
                    individual_centerline_graph.nodes[total_nodes - 1]["vessel_type"] = None
                    individual_centerline_graph.nodes[total_nodes - 1]["vessel_type_name"] = centerline_id
                    individual_centerline_graph[total_nodes - 2][total_nodes - 1]["cell_id"] = cell_id
                    individual_centerline_graph[total_nodes - 2][total_nodes - 1]["vessel_type"] = None
                    individual_centerline_graph[total_nodes - 2][total_nodes - 1]["vessel_type_name"] = centerline_id
                    individual_centerline_graph[total_nodes - 2][total_nodes - 1]["indices"] = np.append(individual_centerline_graph[total_nodes - 2][total_nodes - 1]["indices"], np.arange(previous_idx, idx))
                    individual_centerline_graph[total_nodes - 2][total_nodes - 1]["centerline_coordinate_array"] = centerline_coordinate_array[cell_id][individual_centerline_graph[total_nodes - 2][total_nodes - 1]["indices"]]
                    individual_centerline_graph[total_nodes - 2][total_nodes - 1]["centerline_radius_array"] = centerline_radius_array[cell_id][individual_centerline_graph[total_nodes - 2][total_nodes - 1]["indices"]]
                    previous_position = position
            # If the accumulated distance from the last sample node is smaller than sampling_distance_mm, and we are not in either the first or last nodes of the centerline_segments_array cell, just update the distance
            else:
                distance += np.linalg.norm(previous_position - position)
                previous_position = position
                
    # Merge nodes that share the same coordinate (bifurcation spots)
    # First get all nodes that have a degree of 1 (start- and endpoints)
    deg_one_nodes = []
    for node, deg in individual_centerline_graph.degree:
        if deg == 1:
            deg_one_nodes.append(node)

    # For all degree 1 nodes, we check position to join corresponding startpoints and endpoints
    contracted_nodes = []
    for _, node in enumerate(deg_one_nodes):
        if node not in contracted_nodes:
            aux = deg_one_nodes.copy()
            aux.remove(node)
            for aux_nodes in contracted_nodes:
                aux.remove(aux_nodes)
            for _, node2 in enumerate(aux):
                coordinate_1 = individual_centerline_graph.nodes[node]["pos"]
                coordinate_2 = individual_centerline_graph.nodes[node2]["pos"]
                if np.linalg.norm(coordinate_1 - coordinate_2) < 1e-2:
                    individual_centerline_graph = nx.contracted_nodes(individual_centerline_graph, node, node2)
                    contracted_nodes.append(node2)
                    individual_centerline_graph.nodes[node].pop("contraction")
                    
    # If any nodes with degree == 0 are present, remove them
    remove_nodes = []
    for node in individual_centerline_graph:
        if individual_centerline_graph.degree(node) == 0:
            remove_nodes.append(node)
    for node in remove_nodes:
        individual_centerline_graph.remove_node(node)

    # Relabel nodes as sequential labels
    mapping = {}
    new_node = 0
    for old_node in individual_centerline_graph.nodes():
        mapping[old_node] = new_node
        new_node += 1
    individual_centerline_graph = nx.relabel.relabel_nodes(individual_centerline_graph, mapping)
    
    # Get hierarchical order (use femoral for simplicity when adapting to the previously existing FE module)
    individual_centerline_graph = get_hierarchical_order(individual_centerline_graph, "femoral", 0)

    return individual_centerline_graph