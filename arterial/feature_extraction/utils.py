#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np
import networkx as nx

import pickle

import matplotlib.pyplot as plt

def predicted_vessels_dict(case_dir):
    """
    Builds cell_id to vessel type and vessel type name dictionaries.
    
    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory.

    Returns
    -------
    predicted_vessel_types : dictionary
        Dictionary with cell_ids from centerline_segments_array as keys and vessel types
        as values.
    predicted_vessel_type_names : dictionary
        Dictionary with cell_ids from centerline_segments_array as keys and vessel type names
        as values.

    """
    # Load predicted graph
    with open(os.path.join(case_dir, "graph_pred.pickle"), "rb") as f:
        graph_pred = pickle.load(f)
    # Declare empty dicts
    predicted_vessel_types, predicted_vessel_type_names = {}, {}
    # Build dicts from graph edges and their cell_ids, vessel types and vessel type names
    for src, dst in graph_pred.edges:
        predicted_vessel_types[graph_pred[src][dst]["cell_id"]] = graph_pred[src][dst]["vessel type"]
        predicted_vessel_type_names[graph_pred[src][dst]["cell_id"]] = graph_pred[src][dst]["vessel type name"]

    return predicted_vessel_types, predicted_vessel_type_names

def get_hierarchical_order(graph, access = "femoral", start_node = 0):
    """
    Computes hierarchization of graph. Associates each node to an index that
    indicates the number of nodes to the closest startpoint.

    Parameters
    ----------
    graph : networkx.Graph
        Centerline graph the nodes of which we want to order.
    access : string
        Access point for catheterization. Can either be "femoral" or "radial". 
        "femoral" by default.
    start_node : integer
        Node from `graph` that we want to start from (hierarhcy = 0).

    Returns
    -------
    graph : networkx.Graph
        Centerline graph with ordered nodes. Adds `hierarchy {access}` to node 
        attributes.
    
    """
    # We need a boolean variable to see if the analysis is finished
    hierarchy_done = False
    # We use this list to get the source nodes with the same hierarchy value at each iteration
    source_nodes = [start_node]
    # We use this list to avoid repetition of any already analyzed nodes
    used_nodes = [start_node]
    # Hierarchy of start_node is 0
    graph.nodes[start_node]["hierarchy {}".format(access)] = 0
    # Initialize hierarchy value
    hierarchy = 1 

    # Start hierarchization
    while not hierarchy_done:
        # Target nodes will be neighbors of source_nodes that have not been yet analyzed (not in used_nodes)
        target_nodes = []
        # Loop through source_nodes (nodes that share hierarchy value)
        for src in source_nodes:
            # Append them to used_nodes
            used_nodes.append(src)
            # Check neighbors that have not been used yet
            for dst in graph.neighbors(src):
                if dst not in used_nodes:
                    # Attribute them the corresponding hierarchy index
                    graph.nodes[dst]["hierarchy {}".format(access)] = hierarchy
                    # Store them for next iteration
                    target_nodes.append(dst)
        # Update hierarchy index
        hierarchy += 1
        # Pass previous target_nodes to source_nodes of next iteration
        source_nodes = target_nodes.copy()

        # Check if analysis is finished. If no more target nodes are present and the number of used nodes is 
        # equal to the number of nodes in the graph, the analysis is done
        if len(target_nodes) == 0 and len(used_nodes) >= len(graph.nodes()):
            hierarchy_done = True
        # If no more target nodes are present but there are still unused nodes, get the first unused node and 
        # attribute it with the following hierarchy value
        elif len(target_nodes) == 0 and len(used_nodes) < len(graph.nodes()):
            for node in graph.nodes():
                if node not in used_nodes:
                    # Add node to source nodes
                    source_nodes = [node]
                    # Add hierarchy value to graph node
                    graph.nodes[start_node]["hierarchy {}".format(access)] = hierarchy
                    # Update hierarchy index
                    hierarchy += 1
                    break

    for src, dst in graph.edges:
        if graph.nodes[src]["hierarchy {}".format(access)] < graph.nodes[dst]["hierarchy {}".format(access)]:
            graph[src][dst]["hierarchy {}".format(access)] = graph.nodes[src]["hierarchy {}".format(access)]
        else:
            graph[src][dst]["hierarchy {}".format(access)] = graph.nodes[dst]["hierarchy {}".format(access)]

    return graph

def unify_subgraphs(case_dir, centerline_graph, subgraphs):
    """
    This funciton makes one large connected graph resulting from the union between all 
    subgraphs. To do that, the candidate points for graph union are identified and then a union node 
    is searched in the main graph (largest subgraph).
    
    Candidate points are taken as nodes with degree 1 from secondary subgraphs, that are more proximal
    than the alternative extremal point of the segment. VA extremal points from the main subgraph 
    are also considered. 

    Then, once union candidate points from secondary subgraphs are gathered, candidate points for 
    artificial edge connections to the main subgraph (or larger subgraphs) are searched. A 
    preferential sequence is then followed. First, a candidate from a segment with the same vessel 
    type from all larger subgraphs (excluding the same one, except for the VA candidate search of the 
    main subgraph) is searched. If this is not available, then a preferred vessel type for each vessel 
    type is defined. This preferred vessel type is then searched in all smaller subgraphs (same 
    criteria than before). If this is not found either, then the candidate nodes are joint to the 
    closest node from the main subgraph. 

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Centerline graph built in build_centerline_graph.
    subgraphs : list of networkx.Graph objects
        Separate subgraphs from the centerline_graph build in build_centerline_graph.

    Returns
    -------
    centerline_graph : networkx.Graph
        Unified centerline graph.

    """
    # Get centerline_segments_array
    centerline_segments_array = np.load(os.path.join(case_dir, "centerline_segments_array.npy"), allow_pickle = True)
    # Get coordinates array from centerline_segments_array
    coordinate_array = centerline_segments_array[:, 0]
    # Get radius array from centerline_segments_array
    radius_array = centerline_segments_array[:, 1]
    # Preferred vessel types for subgraph union
    preferred_vessel_types = {"AA": None,
                              "BT": "AA",
                              "RCCA": "BT",
                              "RSA": "BT",
                              "RVA": "RSA",
                              "RICA": "RCCA",
                              "RECA": "RCCA",
                              "LCCA": "AA",
                              "LSA": "AA",
                              "LVA": "LSA", 
                              "LICA": "LCCA",
                              "LECA": "LCCA", 
                              "BA": "RVA",
                              "other": None}
    # Declare empty list for subgraphs union edges
    subgraphs_union_edges = []
    # Predicted vessel types and vessel type names
    predicted_vessel_types = centerline_graph.graph["predicted_vessel_types"]
    predicted_vessel_type_names = centerline_graph.graph["predicted_vessel_type_names"]

    # Compute the center of mass of each of the subgraphs
    if len(subgraphs) > 1:
        centers_of_mass_s = []
        for subgraph in subgraphs[1:]:
            center_of_mass = np.ndarray([0, 3])
            for node in subgraph:
                center_of_mass = np.append(center_of_mass, [subgraph.nodes[node]["pos"]], axis = 0)
            center_of_mass = np.mean(center_of_mass, axis = 0)
            # Store the S coordinates of each subgraph's center of mass
            centers_of_mass_s.append(center_of_mass[2])
        # Reorder subgraphs according to the S coordinate of their center of mass in ascending order. Maintain subgraph at 0 position
        subgraphs = list(np.array(subgraphs, dtype = object)[np.insert(np.argsort(centers_of_mass_s) + 1, 0, 0)])

    # We can get the cell_id for all segments that form the subgraph 
    subgraphs_cell_ids = []
    for subgraph in subgraphs:
        subgraph_cell_ids = []
        for node in subgraph:
            if subgraph.nodes[node]["cell_id"] not in subgraph_cell_ids:
                subgraph_cell_ids.append(subgraph.nodes[node]["cell_id"])      
        subgraphs_cell_ids.append(subgraph_cell_ids)
    
    # Pool all AA points for proximal/distal orientation
    aortic_arch_coordinates_array = np.ndarray([0, 3])
    for cell_id in predicted_vessel_type_names.keys():
        if cell_id in subgraphs_cell_ids[0] and predicted_vessel_type_names[cell_id] == "AA":
            for point in coordinate_array[cell_id]:
                aortic_arch_coordinates_array = np.append(aortic_arch_coordinates_array, [point], axis = 0)

    # Now, we can use the segmentsArray to get the extremal points for all cells
    candidates_for_subgraphs_union = []
    opposite_nodes_for_subgraphs_union = []
    for idx, subgraph_cell_ids in enumerate(subgraphs_cell_ids):
        candidates_for_subgraph_union = []
        opposite_nodes_for_subgraph_union = []
        for cell_id in subgraph_cell_ids:
            # For all other subgraphs, we search for all vesselTypes
            if predicted_vessel_type_names[cell_id] not in ["AA", "other"]:
                # Gather both extremal nodes from each segment
                extremal_node_0 = None
                extremal_node_1 = None
                for node in subgraphs[idx]:
                    if (subgraphs[idx].nodes[node]["pos"] == coordinate_array[cell_id][0]).all() and extremal_node_0 not in candidates_for_subgraph_union: 
                        extremal_node_0 = node
                    if (subgraphs[idx].nodes[node]["pos"] == coordinate_array[cell_id][-1]).all() and extremal_node_1 not in candidates_for_subgraph_union: 
                        extremal_node_1 = node
                # 0 is more proximal than 1 and 0 has degree 1, store node
                if np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[extremal_node_0]["pos"], axis = 1)) < np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[extremal_node_1]["pos"], axis = 1)):
                    if subgraphs[idx].degree(extremal_node_0) == 1:
                        candidates_for_subgraph_union.append(extremal_node_0)
                        opposite_nodes_for_subgraph_union.append(extremal_node_1)
                # 1 is more proximal than 0 and 1 has degree 1, store node
                elif np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[extremal_node_0]["pos"], axis = 1)) >= np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[extremal_node_1]["pos"], axis = 1)):
                    if subgraphs[idx].degree(extremal_node_1) == 1:
                        candidates_for_subgraph_union.append(extremal_node_1)
                        opposite_nodes_for_subgraph_union.append(extremal_node_0)
            # If two nodes are found close by (within 20 mm) and they share the same vesselType, only proximalest (to AA) will be chosen (rarely happens)
            delete_close_nodes = []
            for idx_node_a, node_a in enumerate(candidates_for_subgraph_union):
                for idx_node_b, node_b in enumerate(candidates_for_subgraph_union[:idx_node_a]):
                    if node_a != node_b:
                        if np.linalg.norm(subgraphs[idx].nodes[node_a]["pos"] - subgraphs[idx].nodes[node_b]["pos"]) < 20 and predicted_vessel_type_names[subgraphs[idx].nodes[node_a]["cell_id"]] == predicted_vessel_type_names[subgraphs[idx].nodes[node_b]["cell_id"]]:
                            # We delete the more distal of the two
                            if np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[node_a]["pos"], axis = 1)) < np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[node_b]["pos"], axis = 1)):
                                delete_close_nodes.append(idx_node_b)
                            elif np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[node_a]["pos"], axis = 1)) >= np.amin(np.linalg.norm(aortic_arch_coordinates_array - subgraphs[idx].nodes[node_b]["pos"], axis = 1)):
                                delete_close_nodes.append(idx_node_a)
            candidates_for_subgraph_union = list(np.delete(candidates_for_subgraph_union, delete_close_nodes))
            opposite_nodes_for_subgraph_union = list(np.delete(opposite_nodes_for_subgraph_union, delete_close_nodes))
        # Append each candidate union node separately depending on the subgraph
        candidates_for_subgraphs_union.append(candidates_for_subgraph_union)
        opposite_nodes_for_subgraphs_union.append(opposite_nodes_for_subgraph_union)

    # Auxiliar list to store already joint subgraphs
    composed_subgraphs = []

    # Search for alternate union points in main subgraph or larger subgraphs following a preference system
    for idx, candidates_for_subgraph_union in enumerate(candidates_for_subgraphs_union):
        for candidate_idx, candidate_node in enumerate(candidates_for_subgraph_union):
            # Pool all node and positions from the main graph with the same vesselType as the candidate node
            candidate_main_graph_coordinates = np.ndarray([0, 3])
            candidate_main_graph_nodes = []
            # Boolean variable to end search for union node
            found_union = False
            # Auxiliar variable to store subgraph index to perform composition
            candidate_subgraph_idx = None
            if predicted_vessel_type_names[subgraphs[idx].nodes[candidate_node]["cell_id"]] != "other":
                for subgraph_idx in range(max([idx, 1])):
                    if predicted_vessel_type_names[subgraphs[idx].nodes[candidate_node]["cell_id"]] in [predicted_vessel_type_names[cell_id] for cell_id in subgraphs_cell_ids[subgraph_idx]] and not found_union:
                    # Check if the same vessel type exists in the larger subgraphs
                        # If it exists and has a different cell_id, store all node coordinates
                        for node_nubgraph_idx in subgraphs[subgraph_idx]:
                            if predicted_vessel_type_names[subgraphs[idx].nodes[candidate_node]["cell_id"]] == predicted_vessel_type_names[subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["cell_id"]] and subgraphs[idx].nodes[candidate_node]["cell_id"] != subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["cell_id"]:
                                # In the rare event that the connection node is found to be the centerline_graph.graph["rightmost"], then choose its neighbor
                                if node_nubgraph_idx == centerline_graph.graph["rightmost"]:
                                    node_nubgraph_idx = subgraphs[subgraph_idx].neighbors(centerline_graph.graph["rightmost"]).__next__()
                                # In the case that we are looking at the same subgraph as the candidate node, forbid union with segments in contact
                                if subgraph_idx == idx:
                                    cell_ids_in_contact = []
                                    for neighbor in subgraphs[subgraph_idx].neighbors(opposite_nodes_for_subgraphs_union[idx][candidate_idx]):
                                        cell_ids_in_contact.append(subgraphs[subgraph_idx].nodes[neighbor]["cell_id"])
                                    if subgraphs[idx].nodes[node_nubgraph_idx]["cell_id"] in cell_ids_in_contact:
                                        pass
                                    else:
                                        candidate_main_graph_coordinates = np.append(candidate_main_graph_coordinates, [subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["pos"]], axis = 0)
                                        candidate_main_graph_nodes.append(node_nubgraph_idx)
                                        candidate_subgraph_idx = subgraph_idx
                                        found_union = True
                                # This tries to forbid union in the scenario where the same vessel type is found in the main subgraph and a secondary subgraph 
                                # and the one in the main subgraph is above the secondarty subgraph by a large margin (30 mm). This cound create weird union, 
                                # so it is better not to have this union at all and search for an alternative union
                                elif subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["pos"][2] - subgraphs[idx].nodes[candidate_node]["pos"][2] > 30:
                                    pass
                                # Otherwise, go on with analysis
                                else:
                                    candidate_main_graph_coordinates = np.append(candidate_main_graph_coordinates, [subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["pos"]], axis = 0)
                                    candidate_main_graph_nodes.append(node_nubgraph_idx)
                                    candidate_subgraph_idx = subgraph_idx
                                    found_union = True
                # Check if the preferred vessel type exists in the larger subgraphs
                # Look at all graphs larger than the one we are analyzing (except for main subgraph, where we only look at itself)
                for subgraph_idx in range(max([idx, 1])):
                    if preferred_vessel_types[predicted_vessel_type_names[subgraphs[idx].nodes[candidate_node]["cell_id"]]] in [predicted_vessel_type_names[cell_id] for cell_id in subgraphs_cell_ids[subgraph_idx]] and not found_union:
                        # If it exists, store all node coordinates
                        for node_nubgraph_idx in subgraphs[subgraph_idx]:
                            if preferred_vessel_types[predicted_vessel_type_names[subgraphs[idx].nodes[candidate_node]["cell_id"]]] == predicted_vessel_type_names[subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["cell_id"]]:
                                # In the rare event that the connection node is found to be the centerline_graph.graph["rightmost"], then choose its neighbor
                                if node_nubgraph_idx == centerline_graph.graph["rightmost"]:
                                    node_nubgraph_idx = subgraphs[subgraph_idx].neighbors(centerline_graph.graph["rightmost"]).__next__()
                                # In the case that we are looking at the same subgraph as the candidate node, group cell_ids in contact with candidate node. Forbid union with segments in contact
                                if subgraph_idx == idx:
                                    cell_ids_in_contact = []
                                    for neighbor in subgraphs[subgraph_idx].neighbors(opposite_nodes_for_subgraphs_union[idx][candidate_idx]):
                                        cell_ids_in_contact.append(subgraphs[subgraph_idx].nodes[neighbor]["cell_id"])
                                    if subgraphs[idx].nodes[node_nubgraph_idx]["cell_id"] in cell_ids_in_contact:
                                        pass
                                    else:
                                        candidate_main_graph_coordinates = np.append(candidate_main_graph_coordinates, [subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["pos"]], axis = 0)
                                        candidate_main_graph_nodes.append(node_nubgraph_idx)
                                        candidate_subgraph_idx = subgraph_idx
                                        found_union = True
                                # Otherwise, go on with analysis
                                else:
                                    candidate_main_graph_coordinates = np.append(candidate_main_graph_coordinates, [subgraphs[subgraph_idx].nodes[node_nubgraph_idx]["pos"]], axis = 0)
                                    candidate_main_graph_nodes.append(node_nubgraph_idx)
                                    candidate_subgraph_idx = subgraph_idx
                                    found_union = True
            # If none of the above have worked, pool all node coordinates from main subgraph
            if not found_union:
                for main_graph_node in subgraphs[0]:
                    if predicted_vessel_type_names[subgraphs[0].nodes[main_graph_node]["cell_id"]]:
                        candidate_main_graph_coordinates = np.append(candidate_main_graph_coordinates, [subgraphs[0].nodes[main_graph_node]["pos"]], axis = 0)
                        candidate_main_graph_nodes.append(main_graph_node)
                        candidate_subgraph_idx = 0
                        found_union = True

            # Compose graphs into largest one. Update subgraphs_cell_ids for next candidate node union search
            if found_union:
                # Choose closest node from the candidate pool
                main_graph_node = candidate_main_graph_nodes[np.argmin(np.linalg.norm(candidate_main_graph_coordinates - subgraphs[idx].nodes[candidate_node]["pos"], axis = 1))]
                # Append node pairs altogether
                subgraphs_union_edges.append([main_graph_node, candidate_node])
                # If the the union is found within the same subgraph, or both subgraphs have already been composed, just add the edge (or don't, this is just an intermediate result which is left unused)
                if candidate_subgraph_idx == idx or [candidate_subgraph_idx, idx] in composed_subgraphs:
                    pass
                # Otherwise, compose both graphs. This will be handy for union search of future candidate nodes
                else:
                    # Compose subgraphs onto larger subgraph (candidate_subgraph_idx will be smaller than idx, by design)
                    subgraphs[candidate_subgraph_idx] = nx.compose(subgraphs[idx], subgraphs[candidate_subgraph_idx])
                    subgraphs[candidate_subgraph_idx].add_edge(main_graph_node, candidate_node)
                    for cell_id in subgraphs_cell_ids[idx]:
                        subgraphs_cell_ids[candidate_subgraph_idx].append(cell_id)
                    # Add to already composed list
                    composed_subgraphs.append([candidate_subgraph_idx, idx])
            else:
                # If no union is found, remove candidates and opposite nodes from lists
                candidates_for_subgraphs_union[idx].remove(candidate_node)
                opposite_nodes_for_subgraphs_union[idx].remove(opposite_nodes_for_subgraphs_union[idx][candidate_idx])

    # Prepare next_cell_id for segment splitting
    next_cell_id = len(coordinate_array)
    # In order to deliver a more complete supersegment visualization, and accurately deliver supersegments with their bifurcating segments, 
    # we will include artificial edges to the graphs, and we will divide segments (segmentsCoordinatesArray and radius_array) into new cell_ids
    # We analyze each artificial union
    for main_graph_node, candidate_node in subgraphs_union_edges:
        candidate_cell_id = centerline_graph.nodes[candidate_node]["cell_id"]
        # We add the position and radius of the main_graph_node to the candidate_cell_id segment to the first (or last) position of the segments arrays
        if np.linalg.norm(centerline_graph.nodes[main_graph_node]["pos"] - coordinate_array[candidate_cell_id][0]) < np.linalg.norm(centerline_graph.nodes[main_graph_node]["pos"] - coordinate_array[candidate_cell_id][-1]):
            coordinate_array[candidate_cell_id] = np.insert(coordinate_array[candidate_cell_id], 0, [centerline_graph.nodes[main_graph_node]["pos"]], axis = 0)
            radius_array[candidate_cell_id] = np.insert(radius_array[candidate_cell_id], 0, centerline_graph.nodes[main_graph_node]["radius"]) 
        else:
            coordinate_array[candidate_cell_id] = np.append(coordinate_array[candidate_cell_id], [centerline_graph.nodes[main_graph_node]["pos"]], axis = 0)
            radius_array[candidate_cell_id] = np.append(radius_array[candidate_cell_id], centerline_graph.nodes[main_graph_node]["radius"]) 
        
        # We add the edge to the graph
        centerline_graph.add_edge(main_graph_node, candidate_node, cell_id = centerline_graph.nodes[candidate_node]["cell_id"])
        centerline_graph[main_graph_node][candidate_node]["vessel type"] = predicted_vessel_types[centerline_graph[main_graph_node][candidate_node]["cell_id"]]
        centerline_graph[main_graph_node][candidate_node]["vessel type name"] = predicted_vessel_type_names[centerline_graph[main_graph_node][candidate_node]["cell_id"]]
        centerline_graph[main_graph_node][candidate_node]["indices"] = np.array([])
        centerline_graph[main_graph_node][candidate_node]["coordinate_array"] = np.ndarray([0, 3])
        centerline_graph[main_graph_node][candidate_node]["radius_array"] = np.array([])
        
        # Now, for the modification of the segments arrays and the cell_ids and indices of nodes and edges, we perform an indepth analysis.
        # First of all, it only makes sense to split the segment if the node has degree 2 (otherwise it will already be a border between different segments)
        if centerline_graph.degree(main_graph_node) == 3:
            # The position of the main_graph_node will be the division point between segments
            cutOffIdx = np.argmin(np.linalg.norm(coordinate_array[centerline_graph.nodes[main_graph_node]["cell_id"]] - centerline_graph.nodes[main_graph_node]["pos"], axis = 1))
            # Check if segment goes downstream with respect to hierarchy. If it is, go against hierarchy. If it is not (normal case), go with hierarchy
            downstream = False
            for neighbor in centerline_graph.neighbors(main_graph_node):
                # If neighbor with higher hierarchy has smaller indices indices than cutOffIdx, the segment is downstream. Otherwise it is not
                if centerline_graph.nodes[main_graph_node]["cell_id"] == centerline_graph.nodes[neighbor]["cell_id"] and centerline_graph.nodes[neighbor]["hierarchy femoral"] > centerline_graph.nodes[main_graph_node]["hierarchy femoral"] and np.mean(centerline_graph[neighbor][main_graph_node]["indices"]) < cutOffIdx:
                    downstream = True
            # We keep main_graph_node as initial previous_node for recursive node analysis
            previous_node = main_graph_node
            # Boolean variable to stop the neighbor sweeping
            cell_id_change_completed = False
            while not cell_id_change_completed:
                # Auxiliar boolean variable to check if a neighbor fulfilling the conditions has been found
                neighbor_found = False
                # Sweep through neighbors of previous_node
                for neighbor in centerline_graph.neighbors(previous_node):
                    # If not downstream and there is a node with the same cell_id as the main_graph_node and a higher hierarchy
                    if not downstream and centerline_graph.nodes[neighbor]["cell_id"] == centerline_graph.nodes[main_graph_node]["cell_id"] and centerline_graph.nodes[neighbor]["hierarchy femoral"] > centerline_graph.nodes[previous_node]["hierarchy femoral"]:
                        # Update node cell_id
                        centerline_graph.nodes[neighbor]["cell_id"] = next_cell_id
                        # Update edge cell_id
                        centerline_graph[previous_node][neighbor]["cell_id"] = next_cell_id
                        # Update edge indices with cutOffIdx
                        centerline_graph[previous_node][neighbor]["indices"] = centerline_graph[previous_node][neighbor]["indices"] - cutOffIdx
                        # Eliminate negative edges if found (this)
                        while centerline_graph[previous_node][neighbor]["indices"][0] < 0 and len(centerline_graph[previous_node][neighbor]["indices"]) > 1:
                            centerline_graph[previous_node][neighbor]["indices"] = np.delete(centerline_graph[previous_node][neighbor]["indices"], 0)
                            centerline_graph[previous_node][neighbor]["coordinate_array"] = np.delete(centerline_graph[previous_node][neighbor]["coordinate_array"], 0, axis = 0)
                            centerline_graph[previous_node][neighbor]["radius_array"] = np.delete(centerline_graph[previous_node][neighbor]["radius_array"], 0)
                        # Update previous_node
                        previous_node = neighbor
                        # Check found neighbor
                        neighbor_found = True
                    # If downstream and there is a node with the same cell_id as the main_graph_node and a lower hierarchy
                    elif downstream and centerline_graph.nodes[neighbor]["cell_id"] == centerline_graph.nodes[main_graph_node]["cell_id"] and centerline_graph.nodes[neighbor]["hierarchy femoral"] < centerline_graph.nodes[previous_node]["hierarchy femoral"]:
                        # Update node cell_id
                        centerline_graph.nodes[neighbor]["cell_id"] = next_cell_id
                        # Update edge cell_id
                        centerline_graph[previous_node][neighbor]["cell_id"] = next_cell_id
                        # Update edge indices with cutOffIdx
                        centerline_graph[previous_node][neighbor]["indices"] = centerline_graph[previous_node][neighbor]["indices"] - cutOffIdx
                        while centerline_graph[previous_node][neighbor]["indices"][0] < 0 and len(centerline_graph[previous_node][neighbor]["indices"]) > 1:
                            centerline_graph[previous_node][neighbor]["indices"] = np.delete(centerline_graph[previous_node][neighbor]["indices"], 0)
                            centerline_graph[previous_node][neighbor]["coordinate_array"] = np.delete(centerline_graph[previous_node][neighbor]["coordinate_array"], 0, axis = 0)
                            centerline_graph[previous_node][neighbor]["radius_array"] = np.delete(centerline_graph[previous_node][neighbor]["radius_array"], 0)
                        # Update previous_node
                        previous_node = neighbor
                        # Check found neighbor
                        neighbor_found = True
                # When a neighbor fulfilling the conditions is not found, the graph update is complete
                if not neighbor_found:
                    cell_id_change_completed = True
            # Now update the coordinate_array and the radius_array
            # Create a new object at the end of the array
            coordinate_array = np.hstack((coordinate_array, np.empty(1)))
            # The new cell will contain all points from the original cell_id from cutOffIdx onwards
            coordinate_array[next_cell_id] = coordinate_array[centerline_graph.nodes[main_graph_node]["cell_id"]][cutOffIdx:]
            # The original cell will only keep points up until cutOffIdx (included)
            coordinate_array[centerline_graph.nodes[main_graph_node]["cell_id"]] = coordinate_array[centerline_graph.nodes[main_graph_node]["cell_id"]][:cutOffIdx + 1]
            # Create a new object at the end of the array
            radius_array = np.hstack((radius_array, np.empty(1)))
            # The new cell will contain all points from the original cell_id from cutOffIdx onwards
            radius_array[next_cell_id] = radius_array[centerline_graph.nodes[main_graph_node]["cell_id"]][cutOffIdx:]
            # The original cell will only keep points up until cutOffIdx (included)
            radius_array[centerline_graph.nodes[main_graph_node]["cell_id"]] = radius_array[centerline_graph.nodes[main_graph_node]["cell_id"]][:cutOffIdx + 1]
            # Also, add new cell_id to label dicts
            predicted_vessel_types[next_cell_id] = predicted_vessel_types[centerline_graph.nodes[main_graph_node]["cell_id"]]
            predicted_vessel_type_names[next_cell_id] = predicted_vessel_type_names[centerline_graph.nodes[main_graph_node]["cell_id"]]
            # Update next_cell_id
            next_cell_id += 1

    # Check for separate subgraphs after graph unification
    subgraphs_aux = [centerline_graph.subgraph(components) for components in nx.connected_components(centerline_graph)]

    # If more than one subgraph is still found, it is probably a problematic one. We remove all nodes from all remaining secondary subgraphs
    if len(subgraphs_aux) > 1:
        positions_subgraphs = []
        for subgraph in subgraphs_aux[1:]:
            for subgraphNode in subgraph:
                positions_subgraphs.append(subgraph.nodes[subgraphNode]["pos"])

        remove_nodes = []
        for node in centerline_graph:
            if np.amin(np.linalg.norm(centerline_graph.nodes[node]["pos"] - positions_subgraphs, axis = 1)) < 1e-5:
                remove_nodes.append(node)

        for node in remove_nodes:
            centerline_graph.remove_node(node)

    # Add edges to remove to global attributes
    centerline_graph.graph["subgraphs_union_edges"] = subgraphs_union_edges

    # Initiaize new centerline segments array
    new_centerline_segments_array = np.ndarray([len(coordinate_array), 2], dtype = object)
    # Overwrite coordinates array from centerline_segments_array
    new_centerline_segments_array[:, 0] = coordinate_array
    # Overwrite radius array from centerline_segments_array
    new_centerline_segments_array[:, 1] = radius_array
    # Update global features
    centerline_graph.graph["predicted_vessel_types"] = predicted_vessel_types
    centerline_graph.graph["predicted_vessel_type_names"] = predicted_vessel_type_names
    centerline_graph.graph["centerline_segments_array"] = new_centerline_segments_array

    return centerline_graph

def sanity_check_for_random_islands(subgraphs, skip_cell_ids):
    """
    Search for potentially randomly segmented islands depending on the distance between 
    centers of mass of the different subgraphs. If an anomaly is found (i.e., a segment
    is found far from the overall center of mass of the complete graph), then segments in
    the subgraph are eliminated fromt he centerline_segments_array.npy and the analysis is repeated

    Parameters
    ----------
    subgraphs : list of networkx.Graph objects
        Separate subgraphs from the centerline_graph build in build_centerline_graph.

    Returns
    -------
    sanity_check : bool
        Determines if sanity check is passed or not. Graph will be built repeatedly until sanity 
        check is passed - becomes True.
    skip_cell_ids : list
        Contains which cell_ids will be skipped for the next iteration of centerline graph built.
    
    """
    # Compute the center of mass of each of the subgraphs, as well as the overall center of mass
    global_center_of_mass = np.ndarray([0, 3])
    centers_of_mass = []
    for subgraph in subgraphs:
        center_of_mass = np.ndarray([0, 3])
        for node in subgraph:
            center_of_mass = np.append(center_of_mass, [subgraph.nodes[node]["pos"]], axis = 0)
            global_center_of_mass = np.append(global_center_of_mass, [subgraph.nodes[node]["pos"]], axis = 0)
        center_of_mass = np.mean(center_of_mass, axis = 0)
        # Store the center of mass of each subgraph
        centers_of_mass.append(center_of_mass)
    # Compute global center of mass
    global_center_of_mass = np.mean(global_center_of_mass, axis = 0)
    # Compute distance from all subgraph's center of mass to the global center of mass
    distances = np.linalg.norm(np.array(centers_of_mass) - np.array([global_center_of_mass]), axis = 1)
    
    delete_subgraphs = []
    new_skip_cell_ids = []
    for idx, distance in enumerate(distances):
        if distance > 2.5 * np.mean(distances):
            delete_subgraphs.append(idx)
            for node in subgraphs[idx]:
                if subgraphs[idx].nodes[node]["cell_id"] not in new_skip_cell_ids:
                    new_skip_cell_ids.append(subgraphs[idx].nodes[node]["cell_id"])

    if len(new_skip_cell_ids) > 0:
        sanity_check = False
        print("Found random islands containing the following cell_ids: {}".format(new_skip_cell_ids))
        # Reinitializing graph built skipping the corresponding cell_ids
    else:
        sanity_check = True
    # We concatenate the old cell_ids with the new ones
    skip_cell_ids += new_skip_cell_ids

    return sanity_check, skip_cell_ids

def make_graph_plot(case_dir, graph, filename, label = None):
    """
    Makes matplotlib.pyplot figure of the coronal plane of a networkx graph.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    graph : networkx.Graph
        Graph that we want to plot.
    filename : string
        Fine name of the final image. Make sure to add a valid extension (e.g. .png, .eps, etc)
    label : string
        Edge attribute to be printed at the center of each graph edge.

    Returns
    -------

    """
    # Generate plot of dense graph for quick visualization
    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()

    # In order to place the nodes in the visualization of the graph in a sagittal view, 
    # we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_p = {}
    for n in graph.nodes():
        node_pos_dict_p[n] = [graph.nodes(data=True)[n]["pos"][0], graph.nodes(data=True)[n]["pos"][2]]

    if label is not None:
        edge_labels = nx.get_edge_attributes(graph, label)
        nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
        nx.draw_networkx_edge_labels(graph, node_pos_dict_p, edge_labels = edge_labels, ax=ax)
    else:
        nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)

    plt.savefig(os.path.join(case_dir, filename))
    plt.close()