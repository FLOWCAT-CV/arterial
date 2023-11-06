#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os, shutil, json

import math
import numpy as np
import networkx as nx

import pickle

import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

def supersegment_prediction(centerline_graph):
    """ 
    Performs supersegment search and associates all possible supersegment configurations to all
    reference configurations.

    For each access, starts iterative search at the startNode (hierarchy == 0) and stores all 
    cell_ids sequences for the supersegment candidate segments as well as the bifurcating segments.

    Then, a one-hot encoded version of the supersegments is generated, encoded for the present 
    vessel_types in the cell_id sequence. This is compared to the reference thrombectomy configurations
    (also one-hot encoded, with enhanced weights for all unique vessel_types of each configuration) 
    and the cosine similarity between encoded sequences is used as a similarity measure. The 
    supersegment candidate sequence for each configuration is selected and stored in 
    predicted_configurations, keeping the cell_id sequences of the supersegment and the bifurcating 
    segments.

    Parameters
    ----------
    centerline_graph : networkx.Graph
        Featurized dense centerline graph.

    Returns
    -------
    predicted_configurations : dict
        Stores all possible catheter pathways (in the form of networkx.Graph objects) from the 
        startpoint for each access (femoral or radial) to all other endpoints.
    
    """
    def rescale_hierarchy(graph, access):
        """
        Rescales hierarchy for startpoint to start from hierarchy = 0. In this script, we use it 
        just to retrieve the maximum hierarchy.

        Parameters 
        ----------
        graph : networkx.Graph
            Segment graph.

        Returns
        -------
        graph : networkx.Graph
            Segment graph with updated hierarchy.
        new_max_hierarchy : 
            Maximum hierarchy of the new rescaled graph.
    
        """
        # Initialize minimum and maximum hierarchy
        min_hierarchy = 1000
        max_hierarchy = 0
        # Iterate over all nodes to findmin and max hierarchies
        for node in graph:
            if graph.nodes[node]["hierarchy {}".format(access)] < min_hierarchy:
                min_hierarchy = graph.nodes[node]["hierarchy {}".format(access)]
            if graph.nodes[node]["hierarchy {}".format(access)] > max_hierarchy:
                max_hierarchy = graph.nodes[node]["hierarchy {}".format(access)]
        # Rescale all hierarchy indices to start from 0
        for node in graph:
            graph.nodes[node]["hierarchy {}".format(access)] = graph.nodes[node]["hierarchy {}".format(access)] - min_hierarchy

        return graph, max_hierarchy - min_hierarchy

    def vessel_type_sequence_to_one_hot(sequence, highlights = None, counter_highlights = None):
        """
        Passes any vessel type sequence to one-hot encoding. Highlights and counter_highlights
        (optional) can be used to favor/disfavor the presence of some vessel types for some 
        configurations. 

        Parameters
        ----------
        sequence : list
            Sequence of integers corresponding to the presence of vessel types in a given
            catheter pathway.
        highlights : list
            List of vessel types to be favored for a given configuration.
        counter_highlights : list
            List of vessel types to be disfavored for a given configuration.

        Returns
        -------
        supersegment_candidate_one_hot : list
            List with activations (values different from 0) in the positions of the
            present vessel types in a sequence, with modified values according to highlighted
            vessel types.

        """
        vessel_type_one_hot_code = {
            "other": 0,
            "AA": 1,
            "BT": 2,
            "RCCA": 3,
            "LCCA": 4,
            "RSA": 5,
            "LSA": 6,
            "RVA": 7,
            "LVA": 8,
            "RICA": 9,
            "LICA": 10,
            "RECA": 11,
            "LECA": 12,
            "BA": 13
        }
        
        supersegment_candidate_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for vessel_type in sequence:
            supersegment_candidate_one_hot[vessel_type_one_hot_code[vessel_type]] = 1

        if highlights is not None:
            for highlight in highlights:
                supersegment_candidate_one_hot[vessel_type_one_hot_code[highlight]] += 1

        if counter_highlights is not None:
            for counterHighlight in counter_highlights:
                supersegment_candidate_one_hot[vessel_type_one_hot_code[counterHighlight]] -= 1
            
        return supersegment_candidate_one_hot

    def cosine_similarity(u, v):
        """
        Computes cosine of the angle drawn between two vectors. 
        We use it as a similarity measure between supersegment candidates
        and reference thrombectomy configurations. We use the labelled vessel sequence 
        and pass it to one-hot encoding to perform this measurement.

        Parameters
        ----------
        u : numpy.array
            First vector.
        v : numpy.array
            Second vector.
        
        Returns
        -------
        cosine_similarity : numpy.array
            Array with cosine similarity values along the vector positions.
        
        """
        # Vectors have to be the same length (one-hot encoding ensures this)
        assert len(u) == len(v)
        # The cosine distance or similarity between two vectors is simply the dot product of both vectors
        # divided by the product of their modules
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    # Build standard reference configurations
    configurations = {}
    # Configurations from femoral access
    configurations["femoral"] = []
    # Configuration 0: femoral + right + anterior
    priority_order = ["AA", "BT", "RCCA", "RICA"]
    highlights = ["RCCA", "RICA"]
    counter_highlights = ["RSA"]
    configurations["femoral"].append([priority_order, highlights, counter_highlights, "0_femoral_right_anterior"])
    # Configuration 1: femoral + right + posterior
    priority_order = ["AA", "BT", "RSA", "RVA", "BA"]
    highlights = ["RVA"]
    counter_highlights = ["LVA"]
    configurations["femoral"].append([priority_order, highlights, counter_highlights, "1_femoral_right_posterior"])
    # Configuration 2: femoral + left + anterior
    priority_order = ["AA", "LCCA", "LICA"]
    highlights = ["LCCA", "LICA"]
    counter_highlights = []
    configurations["femoral"].append([priority_order, highlights, counter_highlights, "2_femoral_left_anterior"])
    # Configuration 3: femoral + left + posterior
    priority_order = ["AA", "LSA", "LVA", "BA"]
    highlights = ["LVA"]
    counter_highlights = ["RVA"]
    configurations["femoral"].append([priority_order, highlights, counter_highlights, "3_femoral_left_posterior"])
    # Configurations from radial access
    configurations["radial"] = []
    # Configuration 4: radial + right + anterior
    priority_order = ["RSA", "RCCA", "RICA"]
    highlights = ["RCCA", "RICA"]
    counter_highlights = ["BT"]
    configurations["radial"].append([priority_order, highlights, counter_highlights, "4_radial_right_anterior"])
    # Configuration 5: radial + right + posterior
    priority_order = ["RSA", "RVA", "BA"]
    highlights = ["RVA"]
    counter_highlights = ["LVA"]
    configurations["radial"].append([priority_order, highlights, counter_highlights, "5_radial_right_posterior"])
    # Configuration 6: radial + left + anterior
    priority_order = ["RSA", "BT", "LCCA", "LICA"]
    highlights = ["LCCA", "LICA"]
    counter_highlights = []
    configurations["radial"].append([priority_order, highlights, counter_highlights, "6_radial_left_anterior"])
    # Configuration 7: radial + left + posterior
    priority_order = ["RSA", "BT", "LSA", "LVA", "BA"]
    highlights = ["LVA"]
    counter_highlights = ["RVA"]
    configurations["radial"].append([priority_order, highlights, counter_highlights, "7_radial_left_posterior"])

    # We also have to pass the reference configurations to one-hot encoding
    configurations_one_hot = {}
    for access in ["femoral", "radial"]:
        configurations_one_hot[access] = []
        for configuration in configurations[access]:
            configSequence, highlights, counter_highlights, _ = configuration
            configurations_one_hot[access].append(vessel_type_sequence_to_one_hot(configSequence, highlights, counter_highlights))

    # Define predicted configurations (cell_ids sequences)
    predicted_configurations = {}
    predicted_vessel_type_names = centerline_graph.graph["predicted_vessel_type_names"]

    # Get coordinate array
    segments_coordinate_array = centerline_graph.graph["centerline_segments_array"][0]

    # Create supersegment_candidates, storing all nodes for all possible paths for each access startNode (hierarchy == 0)
    supersegment_candidates = {}
    # Initialize dicts for supersegment and bifurcating segments cell_ids and vessel_types
    supersegment_candidates_cell_ids = {}
    supersegment_candidates_vessel_types = {}
    bifurcating_segments_candidates_cell_ids = {}
    bifurcating_segments_candidates_vessel_types = {}
    # For both access, perform supersegment search
    for idx, access in enumerate(["femoral", "radial"]):
        _, max_hierarchy = rescale_hierarchy(centerline_graph, access)
        # Initialize list for supersegment depending on access
        supersegment_candidates[access] = []
        # This is used to avoid advancing over finished supersegments segments (supersegment candidates are added when an endnode is reached)
        finished_paths = []
        # This is necessary to include segments with only one node in the path computations (sometimes happens when two bifurcations are very close)
        number_of_nodes_for_cell_id = {}
        for cell_id in range(len(segments_coordinate_array)):
            number_of_nodes = 0
            for node in centerline_graph:
                if centerline_graph.nodes[node]["cell_id"] == cell_id:
                    number_of_nodes += 1
            number_of_nodes_for_cell_id[cell_id] = number_of_nodes

        # Now, we loop over all nodes and stack them depending on the hierarchy indices for each access
        # Every time we find a bifurcation, we create a new path. From here, we can extract a cell_id sequence for each of the paths    
        # To detect bifurcations, we check the node degree        
        for current_hierarchy in range(max_hierarchy + 1):
            for node in centerline_graph:
                if centerline_graph.nodes[node][f"hierarchy {access}"] == current_hierarchy:
                    # Store neighbor nodes
                    neighbor_nodes = []
                    for neighbor in centerline_graph.neighbors(node):
                        neighbor_nodes.append(neighbor)
                    # For startNode, just start a new supersegment candidate
                    if len(supersegment_candidates[access]) == 0:
                        supersegment_candidates[access].append([node])
                    # All other nodes
                    else:
                        # In every iteration, search for new segments
                        new_segments = []
                        for idx, path_aux in enumerate(supersegment_candidates[access]):
                            path = path_aux.copy()
                            if idx not in finished_paths:
                                # Endpoints
                                # Since the startNode is treated differently, all nodes with degree == 0 are endpoints
                                if centerline_graph.degree(node) == 1 and path[-1] in neighbor_nodes:
                                    supersegment_candidates[access][idx].append(node)
                                    # When an endpoint is reached, add the sequence to finished_paths to discontinue attention over it
                                    finished_paths.append(idx)
                                # Normal node
                                # For nodes with degree == 2, just add to every active sequence
                                elif centerline_graph.degree(node) == 2 and path[-1] in neighbor_nodes:
                                    supersegment_candidates[access][idx].append(node)
                                # Multifurcations
                                # For multifurcations, create new segments for all bifurcations except for one (which can continue previously existing segment)
                                elif centerline_graph.degree(node) > 2 and path[-1] in neighbor_nodes:
                                    # Auxiliar boolean variable
                                    first_bifurcation = True
                                    supersegment_candidates[access][idx].append(node)
                                    for _, neighbor in enumerate(neighbor_nodes):
                                        if neighbor != path[-1] and neighbor not in supersegment_candidates[access][idx]:
                                            # For the first neighbor, we add it to the the supersegment candidate
                                            if first_bifurcation:
                                                supersegment_candidates[access][idx].append(neighbor)
                                                first_bifurcation = False
                                            # For the first neighbor, we add it to the the supersegment candidate
                                            # For all other neighbors, we create new segments
                                            else:
                                                new_segment = supersegment_candidates[access][idx][:-1].copy()
                                                new_segment.append(neighbor)
                                                new_segments.append(new_segment)
                                # Special case: when two bifurcations come in consecutive nodes
                                # The second bifurcation will share hierarchy with the neighbors from the first bifurcation,
                                # and it will have already been added to one of the paths
                                elif centerline_graph.degree(node) > 2 and node == path[-1]:
                                    first_bifurcation = True
                                    for _, neighbor in enumerate(neighbor_nodes):
                                        if neighbor != path[-1] and neighbor not in supersegment_candidates[access][idx]:
                                            # For the first neighbor, we add it to the the supersegment candidate
                                            if first_bifurcation:
                                                supersegment_candidates[access][idx].append(neighbor)
                                                first_bifurcation = False
                                            # For all other neighbors, we create new segments
                                            else:
                                                new_segment = supersegment_candidates[access][idx][:-1].copy()
                                                new_segment.append(neighbor)
                                                new_segments.append(new_segment)
                        # Once a hierarchy index is fully covered, add new segments to the supersegment_candidates list
                        if len(new_segments) > 0:
                            for segment in new_segments:
                                supersegment_candidates[access].append(segment)

        # Define an empty list for each access for cell_ids and vessel types for supersegment and bifurcating segments
        supersegment_candidates_cell_ids[access] = []   
        bifurcating_segments_candidates_cell_ids[access] = []
        supersegment_candidates_vessel_types[access] = []
        bifurcating_segments_candidates_vessel_types[access] = []
        # Add all cell_ids for the whole node sequence for every possible path from each access startNode
        for idx, supersegment_candidate in enumerate(supersegment_candidates[access]):
            supersegment_candidate_cell_ids = []
            bifurcating_segments_candidate_cell_ids = []
            # Add cell_id from first node
            supersegment_candidate_cell_ids.append(centerline_graph.nodes[supersegment_candidate[0]]["cell_id"])
            # Keep track of previous node
            previous_node = None
            # Iterate over all nodes searching for the cell_id sequence
            for node in supersegment_candidate:
                # First node
                if previous_node is None:
                    previous_node = node
                    # Every other node
                else:
                    # If cell_id from current node corresponds to the last added cell_id from the supersegment canidate cell_id list, skip node
                    if centerline_graph[previous_node][node]["cell_id"] != supersegment_candidate_cell_ids[-1]:
                        # Else, add cell_id from edge, not from node. This helps avoid problems when covering segments that advance downstream
                        supersegment_candidate_cell_ids.append(centerline_graph[previous_node][node]["cell_id"])
                        # If previous node is a bifurcation, add all other cell_ids to bifurcating segments list
                        if centerline_graph.degree(previous_node) > 2:
                            for neighbor in centerline_graph.neighbors(previous_node):
                                if centerline_graph[previous_node][neighbor]["cell_id"] not in supersegment_candidate_cell_ids:
                                    bifurcating_segments_candidate_cell_ids.append(centerline_graph[previous_node][neighbor]["cell_id"])
                    # Update previous node
                    previous_node = node
            # Add supersegment and bifuracating segment cell_id sequences to candidate list
            supersegment_candidates_cell_ids[access].append(supersegment_candidate_cell_ids)
            bifurcating_segments_candidates_cell_ids[access].append(bifurcating_segments_candidate_cell_ids)
        # Pass cell_id sequences to vessel type sequences
        for supersegment_candidate_cell_ids in supersegment_candidates_cell_ids[access]:
            supersegment_candidate_vessel_types = []
            for cell_id in supersegment_candidate_cell_ids:
                if predicted_vessel_type_names[cell_id] not in supersegment_candidate_vessel_types:
                    supersegment_candidate_vessel_types.append(predicted_vessel_type_names[cell_id])
            supersegment_candidates_vessel_types[access].append(supersegment_candidate_vessel_types)
        for bifurcating_segments_candidate_cell_ids in bifurcating_segments_candidates_cell_ids[access]:
            bifurcating_segments_candidate_vessel_types = []
            for cell_id in bifurcating_segments_candidate_cell_ids:
                if predicted_vessel_type_names[cell_id] not in bifurcating_segments_candidate_vessel_types:
                    bifurcating_segments_candidate_vessel_types.append(predicted_vessel_type_names[cell_id])
            bifurcating_segments_candidates_vessel_types[access].append(bifurcating_segments_candidate_vessel_types)
        
        # If two segments are exactly the same in terms of vessel_types but end in two different cell_ids, we choose the longer one
        delete_idx = []
        for idx_a, supersegment_candidate_cell_ids_a in enumerate(supersegment_candidates_cell_ids[access]):
            for idx_b, supersegment_candidate_cell_ids_b in enumerate(supersegment_candidates_cell_ids[access][:idx_a]):
                if supersegment_candidate_cell_ids_a[:-1] == supersegment_candidate_cell_ids_b[:-1] and predicted_vessel_type_names[supersegment_candidate_cell_ids_a[-1]] == predicted_vessel_type_names[supersegment_candidate_cell_ids_b[-1]]:
                    distance_a = 0
                    distance_b = 0
                    for src, dst in centerline_graph.edges:
                        if centerline_graph[src][dst]["cell_id"] == supersegment_candidate_cell_ids_a[-1] and len(centerline_graph[src][dst]["indices"]) != 0:
                            distance_a += np.linalg.norm(centerline_graph.nodes[src]["pos"] - centerline_graph.nodes[dst]["pos"])
                        elif centerline_graph[src][dst]["cell_id"] == supersegment_candidate_cell_ids_b[-1] and len(centerline_graph[src][dst]["indices"]) != 0:
                            distance_b += np.linalg.norm(centerline_graph.nodes[src]["pos"] - centerline_graph.nodes[dst]["pos"])
                    if distance_a > distance_b:
                        delete_idx.append(idx_b)
                    else:
                        delete_idx.append(idx_a)

        # Delete discarded sequences
        supersegment_candidates_cell_ids[access] = list(np.delete(np.array(supersegment_candidates_cell_ids[access], dtype = object), delete_idx))
        supersegment_candidates_vessel_types[access] = list(np.delete(np.array(supersegment_candidates_vessel_types[access], dtype = object), delete_idx))
        bifurcating_segments_candidates_cell_ids[access] = list(np.delete(np.array(bifurcating_segments_candidates_cell_ids[access], dtype = object), delete_idx))
        bifurcating_segments_candidates_vessel_types[access] = list(np.delete(np.array(bifurcating_segments_candidates_vessel_types[access], dtype = object), delete_idx))

    # Build a one-hot encoded version of the paths (encoding vessel_type). We use the same name convention as in the vessel labelling problem
    supersegment_candidates_one_hot = {}
    for access in ["femoral", "radial"]:
        supersegment_candidates_one_hot[access] = []
        for sequence in supersegment_candidates_vessel_types[access]:
            if type(sequence) is not list:
                sequence = [sequence]
            supersegment_candidates_one_hot[access].append(vessel_type_sequence_to_one_hot(sequence))
        
    # Now we select the closest candidate to each of the reference configurations, using the cosine similarity between the one-hot
    # encoded version of the vessel_type sequences and the enhanced one-hot encoded version of the reference configurations
    for access in supersegment_candidates_one_hot.keys():
        predicted_configurations[access] = []
        for idx, configuration_one_hot in enumerate(configurations_one_hot[access]):
            cosine_similarities = []
            for supersegment_candidate_one_hot in supersegment_candidates_one_hot[access]:
                cosine_similarities.append(cosine_similarity(configuration_one_hot, supersegment_candidate_one_hot))
            # For the most similar configuration, we store the cell_id sequences for the supersegment and the bifurcating segments
            predicted_configurations[access].append([supersegment_candidates_cell_ids[access][np.argmax(cosine_similarities)], bifurcating_segments_candidates_cell_ids[access][np.argmax(cosine_similarities)]])

    return predicted_configurations

def supersegment_built(case_dir, centerline_graph, predicted_configurations):
    """ 
    Using the predicted_configurations dictionary, creates featurized supersegments 
    for each of the possible thrombectomy configurations. Uses the same process as 
    for the creation of the centerline_graph, but including nodes from the cell_id 
    sequences stored in predicted_configurations. Nodes from the bifurcating segment 
    cell_id list are only included up to a distance equal to a chosen threshold.

    Returns predicted_configurations and saves supersegments and an image as:

    >>> case_dir/supersegments/{configuration_name}.pickle
    >>> case_dir/supersegments.png

    There are 8 possible configurations, resulting from two binary features that
    describe the access of a past operation and the occlusion localization:
        Access: femoral or radial.
        Laterality: right or left.
        Antero-posterior: anterior or posterior.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Featurized dense centerline graph.
    predicted_configurations : dict
        Dictionary with two lists, one for each access, with all possible configurations
        for the catheter pathways.

    Returns
    -------
    
    """
    def make_supersegment_plots(case_dir, supersegments):
        """
        Makes plots for all possible supersegment configurations and saves an
        image with all 8 configurations:

        >>> case_dir/supersegments.png

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 
        supersegments : dict
            Dictionary with two lists (one for each access, femoral and radial)
            with all supersegment graphs.

        Returns
        -------
        
        """
        # We can visualize first the original hierarchic dense graph
        configuration_titles = ["Femoral + right + anterior",
                                "Femoral + right + posterior",
                                "Femoral + left + anterior",
                                "Femoral + left + posterior",
                                "Radial + right + anterior",
                                "Radial + right + posterior",
                                "Radial + left + anterior",
                                "Radial + left + posterior"]

        rows = 2
        columns = len(configuration_titles) // rows

        _, ax = plt.subplots(rows, columns, figsize = [16, 18])

        for idx_access, access in enumerate(supersegments.keys()):
            for idx, supersegment in enumerate(supersegments[access]):
                highlight_node = None
                for node in supersegment:
                    if supersegment.nodes[node]["hierarchy"] == 0:
                        highlight_node = node
                    
                colorPalette = mcp.gen_color(cmap = "bwr", n = 2)
                color_map = [colorPalette[not supersegment.nodes[node]["is_supersegment"]] for node in supersegment] 
                
                if highlight_node is not None:
                    color_map[highlight_node] = "chartreuse"

                # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
                node_pos_dict_p = {}
                for n in supersegment.nodes():
                    node_pos_dict_p[n] = [-supersegment.nodes(data=True)[n]["pos"][0], supersegment.nodes(data=True)[n]["pos"][2]]

                nx.draw(supersegment, node_pos_dict_p, node_size=10, node_color=color_map, ax=ax[(4 * idx_access + idx) // columns, (4 * idx_access + idx) % columns])
                ax[(4 * idx_access + idx) // columns, (4 * idx_access + idx) % columns].set_title(configuration_titles[(4 * idx_access + idx)], fontsize=12)
                ax[(4 * idx_access + idx) // columns, (4 * idx_access + idx) % columns].set_xlim([-200, 10])
                ax[(4 * idx_access + idx) // columns, (4 * idx_access + idx) % columns].set_ylim([-10, 350])
            
        plt.savefig(os.path.join(case_dir, "supersegments.png"))
        plt.close()

    supersegments = {}

    # Make supersegment directory in case it is missing
    if not os.path.isdir(os.path.join(case_dir, "supersegments")): os.mkdir(os.path.join(case_dir, "supersegments"))

    # Specify the maximum length for a bifurcating segment
    limit_bifurcation_length = 15

    configuration_names = [
        "right + anterior",
        "right + posterior",
        "left + anterior",
        "left + posterior"]

    # Extract sequences for both accesses
    for access in ["femoral", "radial"]:
        # Initiaize supersegment list for both accesses
        supersegments[access] = []
        # Build supersegment for each of the existing reference configurations
        for config_idx, predicted_configuration in enumerate(predicted_configurations[access]):
            configuration_name = "{} + {}".format(access, configuration_names[config_idx])
            # Get predicted confirguration
            supersegment_segments, bifurcating_segments = predicted_configuration
            # Initialize a graph with networkx for each supersegment 
            supersegment = centerline_graph.copy()
            # Define empty list to store centerline_graph nodes connected by edges with cell_id in supersegment_segments
            supersegment_nodes = []
            # Define empty list to store centerline_graph edges with cell_id in supersegment_segments
            supersegment_edges = []
            # Define empty list to store centerline_graph nodes connected by edges with cell_id in bifurcating_segments up to the limit_bifurcation_length
            bifurcating_nodes = []
            # Define empty list to store first centerline_graph edges with cell_id in bifurcating_segments
            bifurcating_edges = []
            
            # Store all corresponding nodes and edges in supersegment_nodes and supersegment_edges
            for src, dst in supersegment.edges:
                if supersegment[src][dst]["cell_id"] in supersegment_segments:
                    if src not in supersegment_nodes:
                        supersegment_nodes.append(src)
                    if dst not in supersegment_nodes:
                        supersegment_nodes.append(dst)
                    supersegment_edges.append((src, dst))
                    
            # Store all corresponding nodes and edges from immediate bfiurcating edges in bifurcating_nodes and bifurcating_edges
            for src, dst in supersegment.edges:
                if (src, dst) not in supersegment_edges and supersegment[src][dst]["cell_id"] in bifurcating_segments:
                    if src in supersegment_nodes and dst not in supersegment_nodes:
                        bifurcating_nodes.append(dst)
                        bifurcating_edges.append((src, dst))
                    elif dst in supersegment_nodes and src not in supersegment_nodes:
                        bifurcating_nodes.append(src)
                        bifurcating_edges.append((src, dst))
                        
            # Store all corresponding nodes left in bifurcating_nodes and bifurcating_edges
            for src, dst in bifurcating_edges:
                if src in bifurcating_nodes:
                    current_dst = src
                else:
                    current_dst = dst
                distance = 0
                continue_search = True
                check = True
                # We only include cases where first bifurcating edge is connected to a deg = 2 node. Else, we only include the first bifurcating node
                if supersegment.degree(current_dst) == 2:
                    while distance < limit_bifurcation_length and continue_search and check:
                        check = False
                        for neighbor in supersegment.neighbors(current_dst):
                            if neighbor not in supersegment_nodes and neighbor not in bifurcating_nodes:
                                check = True
                                if supersegment.degree(neighbor) == 2:
                                    bifurcating_nodes.append(neighbor)
                                    distance += np.linalg.norm(supersegment.nodes[current_dst]["pos"] - supersegment.nodes[neighbor]["pos"])
                                    current_dst = neighbor
                                else:
                                    bifurcating_nodes.append(neighbor)
                                    continue_search = False

            # We now search for all non-included nodes from hierarchicDenseG, which will be masked out
            remove_nodes = []
            for node in supersegment:
                if node not in supersegment_nodes and node not in bifurcating_nodes:
                    remove_nodes.append(node)
                else:
                    if node in supersegment_nodes:
                        supersegment.nodes[node]["is_supersegment"] = 1
                    else:
                        supersegment.nodes[node]["is_supersegment"] = 0
                    supersegment.nodes[node]["hierarchy"] = supersegment.nodes[node][f"hierarchy {access}"]
                    supersegment.nodes[node]["features"] = supersegment.nodes[node][f"features {access}"]
                    supersegment.nodes[node]["features"]["is_supersegment"] = supersegment.nodes[node]["is_supersegment"]
                    supersegment.nodes[node].pop("features femoral")
                    supersegment.nodes[node].pop("features radial")
                    supersegment.nodes[node].pop("hierarchy femoral")
                    supersegment.nodes[node].pop("hierarchy radial")

            for src, dst in supersegment.edges:
                if (src, dst) in supersegment_edges:
                    supersegment[src][dst]["is_supersegment"] = 1
                else:
                    supersegment[src][dst]["is_supersegment"] = 0

            # Perform masking (remove non-included nodes)
            for node in remove_nodes:
                supersegment.remove_node(node)
                    
            # Relabel nodes as sequential labels
            mapping = {}
            new_node = 0
            for old_node in supersegment.nodes():
                mapping[old_node] = new_node
                new_node += 1
            supersegment = nx.relabel.relabel_nodes(supersegment, mapping)
            
            #### Only thing left would be to remove artificial edges (they do not have edge features)

            # Add to the supersegments dict
            supersegments[access].append(supersegment)
            # Save supersegment as pickle
            with open(os.path.join(case_dir, "supersegments", f"{configuration_name}.pickle"), "wb") as f:
                pickle.dump(supersegment, f, protocol = 4)

    # Make plot with all supersegments
    make_supersegment_plots(case_dir, supersegments)

def select_configuration(case_dir, centerline_graph):
    """
    Selects supersegment configuration if patient_configuration.json is present in case_dir.
    Also adds global features derived from a past intervention to the global attributes of the
    graph.

    Creats new dir case_dir/thrombectomy_configuration, and stores selected supersegment and image:

    >>> case/dir/thrombectomy_configuration/supersegment.pickle
    >>> case/dir/thrombectomy_configuration/supersegment.png

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Dense centerline graph returned by graph builder.

    Returns
    -------
    
    """

    def select_vertebrobasilar_laterality(graph, laterality):
        """
        If, in the patient_configuration.json file, laterality is registered as
        `Vertebrobasilar`, both vertebral arteries are compared and if one is
        longer than 3 times the other, that one is chosen. Otherwise, the one 
        with a larger mean diameter is chosen.

        Parameters
        ----------
        graph : networkx.Graph
            Centerline graph.
        laterality : string
            Should be `Vertebrobasilar`. If no decision can be made between both 
            vertebral arteries, it will remain unchanged.

        Returns
        -------
        laterality : string
            Laterality of the chosen side.
        
        """
        radius_r, radius_l = [], []

        for node in graph:
            if graph.nodes[node]["vessel_type_name"] == "RVA":
                try:
                    radius_r.append(graph.nodes[node]["features femoral"]["radius"])
                except:
                    pass
            elif graph.nodes[node]["vessel_type_name"] == "LVA":
                try:
                    radius_l.append(graph.nodes[node]["features femoral"]["radius"])
                except:
                    pass

        if len(radius_r) == 0 and len(radius_l) != 0:
            return "Left"
        elif len(radius_r) != 0 and len(radius_l) == 0:
            return "Right"
        elif len(radius_r) == 0 and len(radius_l) == 0:
            return laterality
        else:
            if len(radius_r) > 3 * len(radius_l):
                return "Right"
            elif len(radius_l) > 3 * len(radius_r):
                return "Left"
            else:
                if np.mean(radius_r) >= np.mean(radius_l):
                    return "Right"
                else:
                    return "Left"


    def add_configuration_features(graph, patient_configuration):
        """
        Adds global features from patient configuration to graph.

        Paremeters
        ----------
        graph : networkx.Graph
            Graph object where information from the patient configuration will
            be stored as global attributes.
        patient_configuration : dict
            Dictionary with information derived from a past thrombectomy operation.

        Returns
        -------
        graph : networkx.Graph
            Graph with updated global attributes, including features derived on
            the thrombectomy configuration.

        """
        # Interventionalist
        # Define dict
        interventionalists_dict = {}
        interventionalists_dict["David"] = 0
        interventionalists_dict["Tomasello"] = 1
        interventionalists_dict["Ribo"] = 2
        interventionalists_dict["Piñana"] = 3
        interventionalists_dict["Coscojuela"] = 4
        interventionalists_dict["Remullo"] = 5
        interventionalists_dict["Bellvitge"] = 6
        interventionalists_dict["Requena"] = 7
        interventionalists_dict["Marta"] = 8
        # Pass to one hot
        internventionalists_one_hot = np.zeros(9)
        if patient_configuration["Interventionalist"] in interventionalists_dict.keys():
            internventionalists_one_hot[interventionalists_dict[patient_configuration["Interventionalist"]]] = 1.
        graph.graph["Interventionalist"] = internventionalists_one_hot

        # Date
        year, month, _ = patient_configuration["Date"].split("-")
        months_from_jan_2018 = 12 * (int(year) - 2018) + int(month) - 1
        graph.graph["Months from Jan 2018"] = months_from_jan_2018

        # Access
        if patient_configuration["Access"] == "Femoral":
            graph.graph["Access"] = 0
        elif patient_configuration["Access"] == "Radial":
            graph.graph["Access"] = 1
        else:
            graph.graph["Access"] = 2

        # DCP
        if not math.isnan(patient_configuration["DCP"]):
            graph.graph["DCP"] = 1
        else:
            graph.graph["DCP"] = 0

        # Antero-posterior
        if patient_configuration["Antero-posterior"] == "Anterior":
            graph.graph["Antero-posterior"] = 0
        elif patient_configuration["Antero-posterior"] == "Posterior":
            graph.graph["Antero-posterior"] = 1
        else:
            graph.graph["Antero-posterior"] = 2
            
        # Laterality
        if patient_configuration["Laterality"] == "Right":
            graph.graph["Laterality"] = 0
        elif patient_configuration["Laterality"] == "Left":
            graph.graph["Laterality"] = 1
        else:
            graph.graph["Laterality"] = 2

        # Impossible accesses
        if not math.isnan(patient_configuration["TFA impossible"]):
            graph.graph["TFA impossible"] = 1
        else:
            graph.graph["TFA impossible"] = 0
            
        if not math.isnan(patient_configuration["TRA impossible"]):
            graph.graph["TRA impossible"] = 1
        else:
            graph.graph["TRA impossible"] = 0

        # Total supersegment length
        total_length = 0
        for node in graph:
            if graph.nodes[node]["is_supersegment"] > 0.5:
                total_length += graph.nodes[node]["features"]["segment length"]    
        graph.graph["Total length"] = total_length

        return graph

    def make_supersegment_plot(case_dir, supersegment, patient_configuration):
        """
        Makes plot of supersegment with chosen thrombectomy configuration. Prints 
        configuration and time to first angiography series acquisition.

        Saves image of supersegment of the patient configuration as:

        >>> case_dir/thrombectomy_configuration/supersegment.png

        Paremeters
        ----------
        case_dir : string or path-like object
            Path to case directory. 
        supersegment : networkx.Graph
            Graph containing only the nodes from a supersegment.
        patient_configuration : dict
            Dictionary with the necessary information to characterize a thrombectomy.

        Returns
        -------

        """
        _ = plt.figure(figsize = [5, 10])
        ax = plt.gca()

        highlightNode = None
        for node in supersegment:
            if supersegment.nodes[node]["hierarchy"] == 0:
                highlightNode = node
            
        colorPalette = mcp.gen_color(cmap = "bwr", n = 2)
        colorMap = [colorPalette[not supersegment.nodes[node]["is_supersegment"]] for node in supersegment] 
        
        if highlightNode is not None:
            colorMap[highlightNode] = "chartreuse"

        # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
        node_pos_dict_P = {}
        for n in supersegment.nodes():
            node_pos_dict_P[n] = [-supersegment.nodes(data=True)[n]["pos"][0], supersegment.nodes(data=True)[n]["pos"][2]]

        nx.draw(supersegment, node_pos_dict_P, node_size=10, node_color=colorMap)
        ax.set_title(patient_configuration["Access"] + " + " + patient_configuration["Laterality"] + " + " + patient_configuration["Antero-posterior"] + ". time: " + str(patient_configuration["Time first angiography"]), fontsize=12)
        ax.set_xlim([-200, 10])
        ax.set_ylim([-10, 350])
            
        plt.savefig(os.path.join(case_dir, "thrombectomy_configuration", "supersegment.png"))

    with open(os.path.join(case_dir, "patient_configuration.json")) as jsonFile:
        patient_configuration = json.load(jsonFile)[os.path.basename(case_dir)]

    # If laterality for thrombectomy is undetermined but it is known that occlusion was vertebrobasilar, choose side with larger VA (mean radius)
    if "Vertebrobasilar" in patient_configuration["Laterality"]:
        patient_configuration["Laterality"] = select_vertebrobasilar_laterality(centerline_graph, patient_configuration["Laterality"])

    if patient_configuration["Laterality"] in ["Right", "Left"]:
        configuration_id = 0
        if patient_configuration["Access"] == "Femoral":
            configuration_id += 0
        elif patient_configuration["Access"] == "Radial": 
            configuration_id += 4
            
        if patient_configuration["Laterality"] == "Right":
            configuration_id += 0
        elif patient_configuration["Laterality"] == "Left": 
            configuration_id += 2
            
        if patient_configuration["Antero-posterior"] == "Anterior":
            configuration_id += 0
        elif patient_configuration["Antero-posterior"] == "Posterior": 
            configuration_id += 1
            
        print("Access:                  ", patient_configuration["Access"])
        print("Laterality:              ", patient_configuration["Laterality"])
        print("Antero-posterior:        ", patient_configuration["Antero-posterior"])
        print("Configuration selected:  ", configuration_id)

        if not os.path.isdir(os.path.join(case_dir, "thrombectomy_configuration")): os.mkdir(os.path.join(case_dir, "thrombectomy_configuration"))

        supersegment_path = "{} + {} + {}.pickle".format(patient_configuration["Access"].lower(), patient_configuration["Laterality"].lower(), patient_configuration["Antero-posterior"].lower())
        print("Selecting supersegment:", supersegment_path)
        shutil.copyfile(os.path.join(case_dir, "supersegments", supersegment_path), os.path.join(case_dir, "thrombectomy_configuration", "supersegment.pickle"))
        with open(os.path.join(case_dir, "thrombectomy_configuration", "supersegment.pickle"), "rb") as f:
            supersegment = pickle.load(f)
        make_supersegment_plot(case_dir, supersegment, patient_configuration)

        supersegment = add_configuration_features(supersegment, patient_configuration)

        supersegment.graph["features"] = {}
        for feature in supersegment.graph.keys():
            if feature not in ["time to first series", "features", "DCP"]:                                                      # Should add more!
                supersegment.graph["features"][feature] = supersegment.graph[feature]

        # Specially added for database preparation
        supersegment.graph["time to first series"] = patient_configuration["Time first angiography"]
        if patient_configuration["Time first angiography"] <= 15:
            supersegment.graph["time to first series over 15 min"] = 0
        else:
            supersegment.graph["time to first series over 15 min"] = 1

        with open(os.path.join(case_dir, "thrombectomy_configuration", "supersegment.pickle"), "wb") as f:
            pickle.dump(supersegment, f, protocol = 4)

    else:
        print("Laterality is ambiguous:", patient_configuration["Laterality"])