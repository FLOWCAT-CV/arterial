#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os, math

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

def get_single_segments_vessel_type(centerline_graph):
    """"
    Gets an ordered, oriented single segment graph for each of the present vessel
    types of a centerline graph.

    Parameters
    ----------
    centerline_graph : netowrkx.Graph

    Returns
    -------
    segments_vessel_type : dict
        Dictionary with all present vessel types as keys and featurized graphs 
        (networkx.Graph objects) as values.

    """
    def find_vessel_types(centerline_graph):
        """
        Finds all unique vessel types from a centerline graph.

        Parameters
        ----------
        centerline_graph : networkx.Graph
            Centerline predicted graph.

        Returns
        -------
        vessel_type_list : list
            List with all present vessel types within the predicted graph.
        
        """
        # Initializes vessel type list
        vessel_type_list = []
        # Iterates over all edges of the simple graph to find unique vessel types
        for node in centerline_graph:
            vessel_type = centerline_graph.nodes[node]["vessel_type_name"]
            if vessel_type != "other" and vessel_type not in vessel_type_list:
                vessel_type_list.append(vessel_type)

        return vessel_type_list

    # Initialize empty segment dict
    segments_vessel_type = {}
    # Get vessel type list
    vessel_type_list = find_vessel_types(centerline_graph)
    # Iterate over all vessel types in list to find segments for all of them
    for vessel_type in vessel_type_list:
        # Extract individual segments according to vessel type
        segments_vessel_type[vessel_type] = get_single_segment(centerline_graph, vessel_type, identifier_type = "vessel_type")        
        if segments_vessel_type[vessel_type] is not None:
            # Perform feature extraction
            segments_vessel_type[vessel_type] = extract_segment_features(segments_vessel_type[vessel_type])
        else:
            # If segment is none, pop from dict to eliminate future errors
            segments_vessel_type.pop(vessel_type)

    if "RCCA" in segments_vessel_type.keys() and "RICA" in segments_vessel_type.keys():
        # Get both segments
        # Add them together (Add nodes from RICA to RCCA)
        # For all nodes in RICA, add max hierarchy from RCCA
        rcca_segment = segments_vessel_type["RCCA"].copy()
        rica_segment = segments_vessel_type["RICA"].copy()
        
        max_hierarchy_rcca = 0
        previous_node = None
        for node in rcca_segment:
            if rcca_segment.nodes[node]["hierarchy femoral"] > max_hierarchy_rcca:
                max_hierarchy_rcca = rcca_segment.nodes[node]["hierarchy femoral"]
                previous_node = node
        
        max_hierarchy_rica = 0
        for node in rica_segment:
            if rica_segment.nodes[node]["hierarchy femoral"] > max_hierarchy_rica:
                max_hierarchy_rica = rica_segment.nodes[node]["hierarchy femoral"]
                
        for hierarchy in range(max_hierarchy_rica):
            for node in rica_segment:
                if rica_segment.nodes[node]["hierarchy femoral"] == hierarchy and node not in rcca_segment:
                    rcca_segment.add_node(node)
                    rcca_segment.add_edge(previous_node, node)
                    for key in rica_segment.nodes[node].keys():
                        rcca_segment.nodes[node][key] = rica_segment.nodes[node][key]
                    rcca_segment.nodes[node]["hierarchy femoral"] += max_hierarchy_rcca + 1
                    previous_node = node
                    
        # Add to dict
        segments_vessel_type["RCA"] = rcca_segment
        # Perform feature extraction
        segments_vessel_type["RCA"] = extract_segment_features(segments_vessel_type["RCA"])
        
    if "LCCA" in segments_vessel_type.keys() and "LICA" in segments_vessel_type.keys():
        # Get both segments
        # Add them together (Add nodes from LICA to LCCA)
        # For all nodes in LICA, add max hierarchy from LCCA
        lcca_segment = segments_vessel_type["LCCA"].copy()
        lica_segment = segments_vessel_type["LICA"].copy()
        
        max_hierarchy_lcca = 0
        previous_node = None
        for node in lcca_segment:
            if lcca_segment.nodes[node]["hierarchy femoral"] > max_hierarchy_lcca:
                max_hierarchy_lcca = lcca_segment.nodes[node]["hierarchy femoral"]
                previous_node = node
        
        max_hierarchy_lica = 0
        for node in lica_segment:
            if lica_segment.nodes[node]["hierarchy femoral"] > max_hierarchy_lica:
                max_hierarchy_lica = lica_segment.nodes[node]["hierarchy femoral"]
                
        for hierarchy in range(max_hierarchy_lica):
            for node in lica_segment:
                if lica_segment.nodes[node]["hierarchy femoral"] == hierarchy and node not in lcca_segment:
                    lcca_segment.add_node(node)
                    lcca_segment.add_edge(previous_node, node)
                    for key in lica_segment.nodes[node].keys():
                        lcca_segment.nodes[node][key] = lica_segment.nodes[node][key]
                    lcca_segment.nodes[node]["hierarchy femoral"] += max_hierarchy_lcca + 1
                    previous_node = node
                    
        # Add to dict
        segments_vessel_type["LCA"] = lcca_segment
        # Perform feature extraction
        segments_vessel_type["LCA"] = extract_segment_features(segments_vessel_type["LCA"])

    return segments_vessel_type

def get_single_segments_cell_ids(centerline_graph):
    """"
    Gets an ordered, oriented single segment graph for each of the present cell ids of 
    a centerline graph.

    Parameters
    ----------
    centerline_graph : netowrkx.Graph

    Returns
    -------
    segments_cell_id : dict
        Dictionary with all present cell ids as keys and featurized graphs 
        (networkx.Graph objects) as values.

    """
    def find_cell_ids(centerline_graph):
        """
        Finds all unique cell ids from a centerline graph.

        Parameters
        ----------
        centerline_graph : networkx.Graph
            Centerline predicted graph.

        Returns
        -------
        cell_id_list : list
            List with all present cell ids within the predicted graph.
        
        """
        # Initializes vessel type list
        cell_id_list = []
        # Iterates over all edges of the simple graph to find unique vessel types
        for node in centerline_graph:
            cell_id = centerline_graph.nodes[node]["cell_id"]
            if cell_id not in cell_id_list:
                cell_id_list.append(cell_id)

        return cell_id_list
        
    # Initialize empty segment dict
    segments_cell_id = {}
    # Get vessel type list
    cell_id_list = find_cell_ids(centerline_graph)
    # Iterate over all vessel types in list to find segments for all of them
    for cell_id in cell_id_list:
        # Extract individual segments according to cell id
        segments_cell_id[cell_id] = get_single_segment(centerline_graph, cell_id, identifier_type = "cell_id")
        if segments_cell_id[cell_id] is not None:
            # Perform feature extraction
            segments_cell_id[cell_id] = extract_segment_features(segments_cell_id[cell_id])
        
    return segments_cell_id

def get_single_segment(centerline_graph, identifier, identifier_type = "vessel_type"):
    """
    Extracts single centerline segment linked to an identifier (vessel type or cell id).
    Extracted segment will be 2D (no bifurcations), oriented with respect to the aortic 
    arch's center of mass and hierarchy will start at 0.

    Parameters
    ----------
    centerline_graph : networkx.Graph
        Dense centerline graph.
    identifier : string or integer
        Can be either vessel type (str) or cell id (int).
    identifier_type : str
        Selects extracted segments will be linked to a cell id or vessel type. Can either
        be `vessel_type` or `cell_id` and will raise a KeyError otherwise. Default: `vessel_type`. 

    Returns
    -------
    segment : networkx.Graph
        Graph of the individual segment.
    
    """
    def get_orientation_reference(centerline_graph):
        """
        Get center of mass of aortic arch as orientation reference for segments.

        Parameters
        ----------
        centerline_graph : networkx.Graph
            Dense centerline graph.

        Returns
        -------
        orientation_reference : numpy.array
            Center of mass of aortic arch coordinates.

        """
        # Initialize array
        aortic_arch_coordinates = np.ndarray([0, 3])
        # Store all aortic arch coordinates
        for node in centerline_graph:
            if centerline_graph.nodes[node]["vessel_type_name"] == "AA":
                aortic_arch_coordinates = np.append(aortic_arch_coordinates, [centerline_graph.nodes[node]["pos"]], axis = 0)
        # Compute center of mass as mean position of all aortic arch nodes
        return np.mean(aortic_arch_coordinates, axis = 0)

    def rescale_hierarchy(graph):
        """
        Rescales hierarchy for startpoint to start from hierarchy = 0.

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
            if graph.nodes[node]["hierarchy femoral"] < min_hierarchy:
                min_hierarchy = graph.nodes[node]["hierarchy femoral"]
            if graph.nodes[node]["hierarchy femoral"] > max_hierarchy:
                max_hierarchy = graph.nodes[node]["hierarchy femoral"]
        # Rescale all hierarchy indices to start from 0
        for node in graph:
            graph.nodes[node]["hierarchy femoral"] = graph.nodes[node]["hierarchy femoral"] - min_hierarchy

        return graph, max_hierarchy - min_hierarchy

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

    # Select the key for node identifier acoording to the selector
    if identifier_type == "vessel_type":
        key_name = "vessel_type_name"
    elif identifier_type == "cell_id":
        key_name = "cell_id"
    else:
        raise KeyError("Identifier type should be `vessel_type` or `cell_id`")
    
    # Initialize subsegment graph (it will be a masked centerline_graph)
    masked_centerline_graph = centerline_graph.copy()
    # We want to remove all nodes not connected to at least one edge of the explored vessel type
    remove_nodes = []
    for node in masked_centerline_graph:
        is_vessel_type = False
        for neighbor in masked_centerline_graph.neighbors(node):
            if masked_centerline_graph[node][neighbor][key_name] == identifier:
                is_vessel_type = True
        if not is_vessel_type:
            remove_nodes.append(node)

    for node in remove_nodes:
        masked_centerline_graph.remove_node(node)

    subgraphs = [masked_centerline_graph.subgraph(components) for components in nx.connected_components(masked_centerline_graph)]

    # Compute all 1D paths present in all subgraphs
    segments_1d = []
    for subgraph in subgraphs:
        subgraph, max_hierarchy = rescale_hierarchy(subgraph)
        # Initialize list for supersegment depending on access
        subgraph_segments_1d = []
        # This is used to avoid advancing over finished supersegments segments (supersegment candidates are added when an endnode is reached)
        finished_paths = []
        for hierarchy in range(max_hierarchy + 1):
            for node in subgraph:
                if subgraph.nodes[node]["hierarchy femoral"] == hierarchy:
                    # Store neighbor nodes
                    neighbor_nodes = []
                    for neighbor in subgraph.neighbors(node):
                        neighbor_nodes.append(neighbor)
                    # For start_node, just start a new supersegment candidate
                    if len(subgraph_segments_1d) == 0:
                        subgraph_segments_1d.append([node])
                    # All other nodes
                    else:
                        # In every iteration, search for new segments
                        new_segments = []
                        for idx, path_aux in enumerate(subgraph_segments_1d):
                            path = path_aux.copy()
                            if idx not in finished_paths:
                                # Endpoints
                                # Since the start_node is treated differently, all nodes with degree == 0 are endpoints
                                if subgraph.degree(node) == 1 and path[-1] in neighbor_nodes:
                                    subgraph_segments_1d[idx].append(node)
                                    # When an endpoint is reached, add the sequence to finished_paths to discontinue attention over it
                                    finished_paths.append(idx)
                                # Normal node
                                # For nodes with degree == 2, just add to every active sequence
                                elif subgraph.degree(node) == 2 and path[-1] in neighbor_nodes:
                                    subgraph_segments_1d[idx].append(node)
                                # Multifurcations
                                # For multifurcations, create new segments for all bifurcations except for one (which can continue previously existing segment)
                                elif subgraph.degree(node) > 2 and path[-1] in neighbor_nodes:
                                    # Auxiliar boolean variable
                                    first_bifurcation = True
                                    subgraph_segments_1d[idx].append(node)
                                    for _, neighbor in enumerate(neighbor_nodes):
                                        if neighbor != path[-1] and neighbor not in subgraph_segments_1d[idx]:
                                            # For the first neighbor, we add it to the the supersegment candidate
                                            if first_bifurcation:
                                                subgraph_segments_1d[idx].append(neighbor)
                                                first_bifurcation = False
                                            # For the first neighbor, we add it to the the supersegment candidate
                                            # For all other neighbors, we create new segments
                                            else:
                                                new_segment = subgraph_segments_1d[idx][:-1].copy()
                                                new_segment.append(neighbor)
                                                new_segments.append(new_segment)
                                # Special case: when two bifurcations come in consecutive nodes
                                # The second bifurcation will share hierarchy with the neighbors from the first bifurcation,
                                # and it will have already been added to one of the paths
                                elif subgraph.degree(node) > 2 and node == path[-1]:
                                    first_bifurcation = True
                                    for _, neighbor in enumerate(neighbor_nodes):
                                        if neighbor != path[-1] and neighbor not in subgraph_segments_1d[idx]:
                                            # For the first neighbor, we add it to the the supersegment candidate
                                            if first_bifurcation:
                                                subgraph_segments_1d[idx].append(neighbor)
                                                first_bifurcation = False
                                            # For all other neighbors, we create new segments
                                            else:
                                                new_segment = subgraph_segments_1d[idx][:-1].copy()
                                                new_segment.append(neighbor)
                                                new_segments.append(new_segment)
                        # Once a hierarchy index is fully covered, add new segments to the supersegmentCandidates list
                        if len(new_segments) > 0:
                            for segment in new_segments:
                                subgraph_segments_1d.append(segment)
        for subgraph_segment_1d in subgraph_segments_1d: 
            segments_1d.append(subgraph_segment_1d)

    # Select longest segment (now it is computed from the number of nodes, but could really be computed by the overall actual distance of each segment)
    length_segments_1d = [len(segment_1d) for segment_1d in segments_1d]
    # Store all nodes not in longest segment
    remove_nodes = []
    for node in masked_centerline_graph:
        if node not in segments_1d[np.argmax(length_segments_1d)]:
            remove_nodes.append(node)
    # Remove all nodes not in longest segment
    for node in remove_nodes:
        masked_centerline_graph.remove_node(node)

    # Finally, orient segment. If it is AA, compare position of both endpoints and start hierarchy = 0 in the one closest to the RAS origin 
    # (careful consideration of the LPS/RAS/ijk coordinates!)
    # It it is not AA, compare distance between both endpoints
    if identifier == "AA":
        end_nodes = []
        distance_end_nodes = []
        for node in masked_centerline_graph:
            if masked_centerline_graph.degree(node) == 1:
                end_nodes.append(node)
                # Compute distance to RAS origin
                distance_end_nodes.append(np.linalg.norm(masked_centerline_graph.nodes[node]["pos"] - [0., 0., 0.]))
    else:
        end_nodes = []
        distance_end_nodes = []
        for node in masked_centerline_graph:
            if masked_centerline_graph.degree(node) == 1:
                end_nodes.append(node)
                # Compute distance to AA center of mass
                distance_end_nodes.append(np.linalg.norm(masked_centerline_graph.nodes[node]["pos"] - get_orientation_reference(centerline_graph)))

    # Select closest node to reference
    if len(distance_end_nodes) > 0:
        start_node = end_nodes[np.argmin(distance_end_nodes)]

        masked_centerline_graph = get_hierarchical_order(masked_centerline_graph, start_node = start_node)

        # Ensure right ordering of the segmentsArray values in the edges that might be used for feature computation
        # To do that, compare the direction of the hierarchy values between nodes and the direction of the indices values
        # If they are coherent (same direction), do nothing. If they are incoherent, flip the segmentsArray arrays
        for src, dst in masked_centerline_graph.edges:
            if len(masked_centerline_graph[src][dst]["indices"]) > 0:
                if np.sign(masked_centerline_graph.nodes[src]["hierarchy femoral"] - masked_centerline_graph.nodes[dst]["hierarchy femoral"]) == np.sign(masked_centerline_graph[src][dst]["indices"][-1] - masked_centerline_graph[src][dst]["indices"][0]):
                    pass
                else:
                    masked_centerline_graph[src][dst]["indices"] = np.flip(masked_centerline_graph[src][dst]["indices"])
                    masked_centerline_graph[src][dst]["coordinate_array"] = np.flip(masked_centerline_graph[src][dst]["coordinate_array"], axis = 0)
                    masked_centerline_graph[src][dst]["radius_array"] = np.flip(masked_centerline_graph[src][dst]["radius_array"])

    if len(masked_centerline_graph) > 2:
        return masked_centerline_graph
    else:
        return None

def extract_segment_features(segment):
    """
    Performs segment-level feature extraction over a vascular centerline segment.
    Calls all available feature extrtaction functions and keeps all data in a  

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    
    Returns
    -------
    segment : networkx.Graph
        Featurized graph of the individual segment.
    
    """
    # Find endnodes nodes
    proximal_node = find_proximal_node(segment)
    distal_node = find_distal_node(segment)
    proximal_node_no_blanking = find_proximal_node(segment, use_blanking=False)
    distal_node_no_blanking = find_distal_node(segment, use_blanking=False)

    segment.graph["features"] = {}
    segment.graph["features"]["length"] = length(segment, proximal_node_no_blanking)
    segment.graph["features"]["mean diameter"] = mean_diameter(segment)
    segment.graph["features"]["std diameter"] = std_diameter(segment)
    segment.graph["features"]["min diameter"] = min_diameter(segment)
    segment.graph["features"]["max diameter"] = max_diameter(segment)
    segment.graph["features"]["proximal diameter"] = proximal_diameter(segment, proximal_node)
    segment.graph["features"]["distal diameter"] = distal_diameter(segment, distal_node)
    segment.graph["features"]["min max diameter ratio"] = min_max_diameter_ratio(segment)
    segment.graph["features"]["tortuosity_index"] = tortuosity_index(segment, proximal_node, distal_node)
    segment.graph["features"]["bending length"] = bending_length(segment, proximal_node, distal_node)
    segment.graph["features"]["cumulative curvature"] = cumulative_curvature(segment, proximal_node_no_blanking)
    segment.graph["features"]["tortuosity index 5 cm"] = tortuosity_index_first_5_cm(segment, proximal_node, distal_node)
    segment.graph["features"]["min polar angle"] = min_polar_angle(segment)
    segment.graph["features"]["accumulated_polar_angle_differential"] = accumulated_polar_angle_differential(segment, proximal_node)

    return segment

def find_proximal_node(segment, use_blanking = True):
    """
    Finds proximal node of the segment. Detects node with smallest hierarhcy
    and blanking = 0. If no nodes have blanking = 0, then it just find node
    with smallest hierarchy.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    proximal_node : integer
        Proximal node of the segment.

    """
    # Initialize hierarhy and proximal node
    hierarchy = 10000
    proximal_node = None
    # Iterate over all nodes to find node with smallest hierarchy and blanking = 0
    if use_blanking:
        for node in segment:
            if segment.nodes[node]["hierarchy femoral"] < hierarchy and segment.nodes[node]["features femoral"]["blanking"] < 0.5:
                proximal_node = node
                hierarchy = segment.nodes[node]["hierarchy femoral"]
    # If no nodes are found (no nodes with blanking = 0), select node with smallest hierarchy
    if proximal_node is None:
        for node in segment:
            if segment.nodes[node]["hierarchy femoral"] < hierarchy:
                proximal_node = node
                hierarchy = segment.nodes[node]["hierarchy femoral"]

    return proximal_node

def find_distal_node(segment, use_blanking = True):
    """
    Finds distal node of the segment. Detects node with largest hierarhcy
    and blanking = 0. If no nodes have blanking = 0, then it just find node
    with largest hierarchy.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    distal_node : integer
        Distal node of the segment.

    """
    # Initialize hierarhy and distal node
    hierarchy = 0
    distal_node = None
    if use_blanking:
        # Iterate over all nodes to find node with largest hierarchy and blanking = 0
        for node in segment:
            if segment.nodes[node]["hierarchy femoral"] >= hierarchy and segment.nodes[node]["features femoral"]["blanking"] < 0.5:
                distal_node = node
                hierarchy = segment.nodes[node]["hierarchy femoral"]
    # If no nodes are found (no nodes with blanking = 0), select node with largest hierarchy
    if distal_node is None:
        for node in segment:
            if segment.nodes[node]["hierarchy femoral"] >= hierarchy:
                distal_node = node
                hierarchy = segment.nodes[node]["hierarchy femoral"]

    return distal_node

def length(segment, proximal_node):
    """
    Finds length of segment. Collects lengths from all edges of the
    segment and computes sum. 

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    proximal_node : integer
        Proximal node of the segment.
    distal_node : integer
        Distal node of the segment.

    Returns
    -------
    length : float
        Length of segment.

    """
    # Initializes line integral to 0
    actual_length = 0
    # Iterate over all nodes between proximal and distal and adds distance between nodes as differentials of the
    # line integral. Go from neighbor to neighbor until we reach the end and all neigbors have been visited
    nodes_visited = [proximal_node]
    node = proximal_node
    done = False
    while not done:
        done = True
        for neighbor in segment.neighbors(node):
            if neighbor not in nodes_visited:
                actual_length += np.linalg.norm(segment.nodes[node]["pos"] - segment.nodes[neighbor]["pos"])
                nodes_visited.append(node)
                node = neighbor
                done = False
                break

    return actual_length

def mean_diameter(segment):  
    """
    Finds mean diameter of segment. Collects diameters from all nodes of the
    segment and computes mean. 

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    mean_diameter : float
        Mean diameter of segment.

    """
    # Initialize list
    diameters = []
    # Collect diameters from all nodes
    for node in segment:
        if segment.nodes[node]["features femoral"]["blanking"] < 0.5:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
    # Compute mean
    if len(diameters) > 0:
        return np.mean(diameters)
    # If no nodes with blanking 0, choose between all nodes
    else:
        for node in segment:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
        return np.mean(diameters)

def std_diameter(segment):  
    """
    Finds mean diameter of segment. Collects diameters from all nodes of the
    segment and computes mean. 

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    mean_diameter : float
        Mean diameter of segment.

    """
    # Initialize list
    diameters = []
    # Collect diameters from all nodes
    for node in segment:
        if segment.nodes[node]["features femoral"]["blanking"] < 0.5:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
    # Compute mean
    if len(diameters) > 0:
        return np.std(diameters)
    # If no nodes with blanking 0, choose between all nodes
    else:
        for node in segment:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
        return np.std(diameters)

def max_diameter(segment):
    """
    Finds maximum diameter along segment. Only looks at nodes with blanking 0.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    max_diameter : float
        Maximum diameter of segment.

    """
    # Initialize list
    diameters = []
    # Collect diameters from all nodes
    for node in segment:
        if segment.nodes[node]["features femoral"]["blanking"] < 0.5:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
    # Select maximum
    if len(diameters) > 0:
        return np.amax(diameters)
    # If no nodes with blanking 0, choose between all nodes
    else:
        for node in segment:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
        return np.amax(diameters)

def min_diameter(segment):
    """
    Finds minimum diameter along segment. Only looks at nodes with blanking 0.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    min_diameter : float
        Minimum diameter of segment.

    """
    # Initialize list
    diameters = []
    # Collect diameters from all nodes
    for node in segment:
        if segment.nodes[node]["features femoral"]["blanking"] < 0.5:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
    # Select minimum
    if len(diameters) > 0:
        return np.amin(diameters)
    # If no nodes with blanking 0, choose between all nodes
    else:
        for node in segment:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
        return np.amin(diameters)

def proximal_diameter(segment, proximal_node):
    """
    Finds proximal diameter of segment.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    proximal_node : integer
        Proximal node of the segment.

    Returns
    -------
    proximal_diameter : float
        Proximal diameter of segment.

    """
    # Reurns diameter
    return segment.nodes[proximal_node]["features femoral"]["radius"] * 2

def distal_diameter(segment, distal_node):
    """
    Finds distal diameter of segment.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    distal_node : integer
        Distal node of the segment.

    Returns
    -------
    distal_diameter : float
        Distal diameter of segment.

    """
    # Reurns diameter
    return segment.nodes[distal_node]["features femoral"]["radius"] * 2
    
def min_max_diameter_ratio(segment):
    """
    Computes ratio between minimum and maximum ratio. A ratio close to 1 indicates 
    that there is small variability of the radius along the segment. A ratio closer to 
    0 may indicate the presence of one or more stenosis.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.

    Returns
    -------
    min_max_diameter_ratio : float
        Ratio between minimum and maximum diameter.

    """
    # Initialize list
    diameters = []
    # Collect diameters from all nodes
    for node in segment:
        if segment.nodes[node]["features femoral"]["blanking"] < 0.5:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
    # Computes ratio between min and max diameters
    if len(diameters) > 0:
        return np.amin(diameters) / np.amax(diameters)
    # If no nodes with blanking 0, choose between all nodes
    else:
        for node in segment:
            diameters.append(2 * segment.nodes[node]["features femoral"]["radius"])
        return np.amin(diameters) / np.amax(diameters)

def tortuosity_index(segment, proximal_node, distal_node):
    """
    Computes tortuosity index of a segment.

    First it finds the proximal and distal nodes of the segment and then it computes 
    the euclidean distance between both and the distance along the centerline, adding
    the distance between consecutive nodes until teh distal node is reached. Then, it 
    computes the ratio between both and the resulting value is subtracted from 1.

    Segments with low tortuosity have a small tortuosity index (approaching 0), while 
    cases with high tortuosity present a larger tortuosity index (approaching 1).

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    proximal_node : integer
        Proximal node of the segment.
    distal_node : integer
        Distal node of the segment.

    Returns
    -------
    tortuosity_index : float
        Tortuosity index of a segment.

    """
    # Computes euclidean distance as norm between both nodes
    euclidean_distance = np.linalg.norm(segment.nodes[proximal_node]["pos"] - segment.nodes[distal_node]["pos"])
    actual_length = length(segment, proximal_node)
    # Check that we are not gonna divide by nan or 0 (if we do, return nan)
    if not math.isnan(actual_length) and actual_length > 0:
        return 1 - euclidean_distance / actual_length
    else:
        return math.nan

def bending_length(segment, proximal_node, distal_node):
    """
    Computes maximum bending length along a segment. It measures the normal
    distances between each node position of the segment to the axis connecting 
    the proximal and distal ends of the segment. Then, it returns the maximum of
    these distances.
    
    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    proximal_node : integer
        Proximal node of the segment.
    distal_node : integer
        Distal node of the segment.

    Returns
    -------
    bending_length : float
        Bending length of a segment.

    """
    # Initialize list and endnodes
    bending_lengths = []
    # Assign node positions
    proximal_node_pos = segment.nodes[proximal_node]["pos"]
    distal_node_pos = segment.nodes[distal_node]["pos"]

    nodes_visited = [proximal_node]
    node = proximal_node
    done = False
    while not done:
        done = True
        for neighbor in segment.neighbors(node):
            if neighbor not in nodes_visited:
                node_pos = segment.nodes[neighbor]["pos"]
                # Compute distance from reference axis
                if (np.linalg.norm(distal_node_pos - proximal_node_pos) * np.linalg.norm(node_pos - proximal_node_pos)) > 1e-4:
                    bending_lengths.append(np.linalg.norm(node_pos - proximal_node_pos) * np.sqrt(max(0, 1 - (np.dot(distal_node_pos - proximal_node_pos, node_pos - proximal_node_pos) / (np.linalg.norm(distal_node_pos - proximal_node_pos) * np.linalg.norm(node_pos - proximal_node_pos))) ** 2)))
                # If a warning is raised, print the values that are causing it
                nodes_visited.append(node)
                node = neighbor
                done = False
                break

    # Return maxium bending length
    if len(bending_lengths) > 0:
        return np.amax(bending_lengths)
    else:
        return 0

def cumulative_curvature(segment, proximal_node):
    """
    Computes cumulative curvature of a segment by adding up the curvatures
    of each node of the segment.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    proximal_node : integer
        Proximal node of the segment.
    distal_node : integer
        Distal node of the segment.

    Returns
    -------
    cumulative_curvature : float
        Cumulative curvature along a segment

    """
    # Initialize cumulative curvature 
    cumulative_curvature = 0
    nodes_visited = [proximal_node]
    node = proximal_node
    done = False
    while not done:
        done = True
        for neighbor in segment.neighbors(node):
            if neighbor not in nodes_visited:
                # Compute distance from reference axis
                cumulative_curvature += segment.nodes[neighbor]["features femoral"]["curvature"]
                nodes_visited.append(node)
                node = neighbor
                done = False
                break

    return cumulative_curvature

def tortuosity_index_first_5_cm(segment, proximal_node, distal_node):
    """
    Computes tortuosity index of first 5 centimeters of the segment.
    First it detects which nodes are within a 5 centimeter distance 
    along the centerline, then it removes all other nodes and finally
    it computes the tortuosity index from the remaining subsegment.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    proximal_node : integer
        Proximal node of the segment.
    distal_node : integer
        Distal node of the segment.

    Returns
    -------
    tortuosty_index_first_5_cm : float
        Tortuosity index of the first 5 cm of the segment.
    
    """
    # Create a copy of the segment
    segment_copy = segment.copy()
    # Initialize list for nodes within 5 cm of the proximal node and cumulative distance
    cumulative_distance = 0
    node = proximal_node
    nodes_visited = [proximal_node]
    done = False
    while not done and cumulative_distance < 50:
        done = True
        for neighbor in segment.neighbors(node):
            if neighbor not in nodes_visited:
                # Compute distance from reference axis
                cumulative_distance += np.linalg.norm(segment_copy.nodes[neighbor]["pos"] - segment_copy.nodes[node]["pos"])
                nodes_visited.append(node)
                node = neighbor
                done = False
                break

    # Eliminate all other nodes from the segment
    for node in segment_copy.copy():
        if node not in nodes_visited:
            segment_copy.remove_node(node)

    proximal_node_new_segment = find_proximal_node(segment_copy, use_blanking=False)
    distal_node_new_segment = find_distal_node(segment_copy, use_blanking=False)
    # Compute tortuosity index from the remaining segment
    return tortuosity_index(segment_copy, proximal_node_new_segment, distal_node_new_segment)

def min_polar_angle(segment):
    """
    Computes the smallest angle along a segment.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    
    Returns
    -------
    theta : float
        Smallest angle along a segment.
    """
    theta = np.pi / 2
    for node in segment:
        if segment.nodes[node]["features femoral"]["direction polar"] < theta:
            theta = segment.nodes[node]["features femoral"]["direction polar"]

    return theta

def accumulated_polar_angle_differential(segment, proximal_node):
    """
    Computes the accumulated polar angle differential along a segment.

    Parameters
    ----------
    segment : networkx.Graph
        Graph of the individual segment.
    
    Returns
    -------
    theta : float
        Accumulated polar angle along a segment.
    """
    nodes_visited = [proximal_node]
    node = proximal_node
    previous_polar_angle = segment.nodes[node]["features femoral"]["direction polar"]
    accumulate_polar_angle_differential = 0
    done = False
    while not done:
        done = True
        for neighbor in segment.neighbors(node):
            if neighbor not in nodes_visited:
                # Compute distance from reference axis
                accumulate_polar_angle_differential += np.abs(previous_polar_angle - segment.nodes[neighbor]["features femoral"]["direction polar"])
                nodes_visited.append(node)
                node = neighbor
                previous_polar_angle = segment.nodes[node]["features femoral"]["direction polar"]
                done = False
                break

    return accumulate_polar_angle_differential

## Measurements between two segments
    
def largest_azimuthal_difference(segment_1, segment_2, abs_polar_angle_threshold = 60 * np.pi / 180):
    """
    Computes the largest azimuthal difference between two segments, considering
    only nodes with and absolute polar angle below a certain threshold.

    Parameters
    ----------
    segment_1 : networkx.Graph
        Graph of the individual segment.
    segment_2 : networkx.Graph
        Graph of the individual segment.
    abs_polar_angle_threshold : float
        Threshold for the absolute polar angle.

    Returns
    -------
    delta_phi : float
        Largest azimuthal difference between two segments.
    """
    delta_phi = 0

    for node_1 in segment_1:
        if np.abs(segment_1.nodes[node_1]["features femoral"]["direction polar"]) < abs_polar_angle_threshold:
            for node_2 in segment_2:
                if np.abs(segment_2.nodes[node_2]["features femoral"]["direction polar"]) < abs_polar_angle_threshold:
                    delta_phi = np.maximum(delta_phi, np.abs(segment_1.nodes[node_1]["features femoral"]["direction azimuth"] - segment_2.nodes[node_2]["features femoral"]["direction azimuth"]))

    return delta_phi

def largest_polar_difference(segment_1, segment_2):
    """
    Computes the largest polar difference between two segments.

    Parameters
    ----------
    segment_1 : networkx.Graph
        Graph of the individual segment.
    segment_2 : networkx.Graph
        Graph of the individual segment.
    abs_polar_angle_threshold : float
        Threshold for the absolute polar angle.

    Returns
    -------
    delta_phi : float
        Largest polar difference between two segments.
    """
    delta_theta = 0

    for node_1 in segment_1:
        for node_2 in segment_2:
            delta_theta = np.maximum(delta_theta, np.abs(segment_1.nodes[node_1]["features femoral"]["direction polar"] - segment_2.nodes[node_2]["features femoral"]["direction polar"]))

    return delta_theta

## Plot functions

def plot_single_segments(case_dir, centerline_graph, segments_vessel_type):
    """
    Makes graph plot of all individual centerline segments for a case.

    Creates new image file as:

    >>> case_dir/single_segments.png

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Dense centerline graph returned by graph builder.
    segments_vessel_type : dict
        Dictionary with all detected segments as keys and 
        networkx.Graph objects for each segment as values.

    Returns
    -------
    
    """
    # Make dummy plot of centerline_graph to get xlim and ylim
    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()
    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in centerline_graph.nodes():
        node_pos_dict_P[n] = [-centerline_graph.nodes(data=True)[n]["pos"][0], centerline_graph.nodes(data=True)[n]["pos"][2]]
    # Draw centerline_graph
    for _ in range(1):
        nx.draw(centerline_graph, node_pos_dict_P, node_size=10, ax=ax)
    # get xlim and ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Determine number of columns according to the number of entries in segments_vessel_type
    # We will limit the number of rows to 2, and adjust the number of columns 
    if len(segments_vessel_type) % 2 == 0:
        columns = len(segments_vessel_type) // 2
    else:
        columns = len(segments_vessel_type) // 2 + 1
    rows = len(segments_vessel_type) // columns
    if len(segments_vessel_type) % columns > 0:
        rows += 1
    # Create template for subplots
    _, ax = plt.subplots(rows, columns, figsize = [5 * columns, 10 * rows])
    # Draw each segment in a subplot space
    for idx, vessel_type in enumerate(segments_vessel_type.keys()):
        # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
        node_pos_dict_P = {}
        for n in segments_vessel_type[vessel_type].nodes():
            node_pos_dict_P[n] = [-segments_vessel_type[vessel_type].nodes(data=True)[n]["pos"][0], segments_vessel_type[vessel_type].nodes(data=True)[n]["pos"][2]]
        
        # If there are multiple axes, use them
        if isinstance(ax, np.ndarray):
            for _ in range(1):
                nx.draw(segments_vessel_type[vessel_type], node_pos_dict_P, node_size = 10, ax = ax[idx // columns, idx % columns])
                ax[idx // columns, idx % columns].set_title(vessel_type, fontsize=12)
                ax[idx // columns, idx % columns].set_xlim(xlim)
                ax[idx // columns, idx % columns].set_ylim(ylim)
        else:
            for _ in range(1):
                nx.draw(segments_vessel_type[vessel_type], node_pos_dict_P, node_size = 10, ax = ax)
                ax.set_title(vessel_type, fontsize=12)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

    # Save plot
    plt.savefig(os.path.join(case_dir, "single_segments.png"))