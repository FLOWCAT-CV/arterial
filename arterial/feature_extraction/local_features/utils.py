#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

import math

import numpy as np

def featurize_node(local_graph, node, access, cta_array, cta_affine, lpi_corner_coordinates, branch_model_coordinates=None, blanking=None):
    """
    Computes local features for a node in the centerline graph, updating the graph.

    Parameters
    ----------
    local_graph : network.Graph
        Dense centerline graph.
    node : integer
        Node of the centerline graph.
    access : string
        Access site for thrombectomy configuration. Can be either "femoral" or "radial".
    cta_array : numpy.array or array-like object.
        Array containing the CTA data.
    cta_affine : numpy.array or array-like object.
        Affine matrix of the CTA data.
    lpi_corner_coordinates : numpy.array or array-like object.
        Coordinates of the corner voxel of the CTA data.
    branch_model_coordinates : numpy.array or array-like object.    
        Array containing the coordinates of the branch model.
    blanking : numpy.array or array-like object.
        Array containing the blanking of the branch model.

    Returns
    -------

    """
    local_points = find_neighbour_points(local_graph, node, access)
    # Get the index of the node among the points
    node_pos = local_graph.nodes[node]["pos"]
    node_idx = np.argmin(np.linalg.norm(local_points - node_pos, axis = 1))

    # Feature extraction           
    local_graph.nodes[node][f"features {access}"] = {}
    
    # Node position
    # We have to compute the ijk positions for normalization purposes (either this or subtract the translation from the affine matrix)
    local_graph.nodes[node][f"features {access}"]["pos r"] = node_pos[0]
    local_graph.nodes[node][f"features {access}"]["pos a"] = node_pos[1]
    local_graph.nodes[node][f"features {access}"]["pos s"] = node_pos[2]
    
    # Radius features
    local_graph.nodes[node][f"features {access}"]["radius"] = local_graph.nodes[node]["radius"]
    # Segment length
    local_graph.nodes[node][f"features {access}"]["segment length"] = sum([np.linalg.norm(local_points[idx - 1] - local_points[idx]) for idx in range(1, len(local_points))]) / len(local_points)
    # Curvature and torsion (perhaps we could use the data from )
    if len(local_points) == 3:
        curvature, torsion = compute_curvature_and_torsion(local_points, 1)
    else:
        curvature, torsion = compute_curvature_and_torsion(local_points, node_idx)
    local_graph.nodes[node][f"features {access}"]["curvature"] = curvature
    local_graph.nodes[node][f"features {access}"]["torsion"] = torsion

    # Directional features
    module, polar, azimuth = compute_spherical_angles(local_points[-1] - local_points[0])
    local_graph.nodes[node][f"features {access}"]["direction module"] = module
    local_graph.nodes[node][f"features {access}"]["direction polar"] = polar
    local_graph.nodes[node][f"features {access}"]["direction azimuth"] = azimuth
    # Other features
    if branch_model_coordinates is not None and blanking is not None:
        if len(branch_model_coordinates) > 0:
            local_graph.nodes[node][f"features {access}"]["blanking"] = blanking[find_point_id(node_pos, branch_model_coordinates)]
        else:
            local_graph.nodes[node][f"features {access}"]["blanking"] = 0
    else:
        local_graph.nodes[node][f"features {access}"]["blanking"] = 0

    # Tranform to ijk
    i, j, k = np.matmul(np.linalg.inv(cta_affine), np.append(node_pos + lpi_corner_coordinates, 1.0))[:3].astype(int)
    local_graph.nodes[node][f"features {access}"]["HU intensity"] = cta_array[i, j, k]

def find_neighbour_points(local_graph, node, access="femoral"):
    """
    Fins the points of the centerline that are used for feature extraction. Each node type, 
    depending on its degree, has a different strategy to find the points, according to hierarchy and 
    access.

    Parameters
    ----------
    local_graph : network.Graph
        Dense centerline graph returned by graph builder.
    node : integer
        Node of the centerline graph.
    access : string
        Access site for thrombectomy configuration. Can be either "femoral" or "radial".

    Returns
    -------
    points : numpy.array or array-like object, shape [3, 3].
        Position of the points used for local feature extraction.

    """
    # Endpoints (degree == 1)
    # For endpoints, we gather information from the endpoint node's relationship to its neighbor (node_end)
    if local_graph.degree[node] == 1:
        # First we search the neighboring node and we get the candidate centerline points for segment feature extraction
        for node_end in local_graph.neighbors(node):
            break
        # Since we are analyzing an endpoint segment, we have to check if the analyzed node is an endpoint or a startpoint
        # We can use the node hierarchy computed for the dense graph for that
        # If the node hierarchy is greater than the alternative node, then we need to keep the distal part of the centerline segment for the feature extraction
        if local_graph.nodes[node][f"hierarchy {access}"] > local_graph.nodes[node_end][f"hierarchy {access}"]: 
            # To compute the curvature, we take the position of the last three nodes (we compute curvature at a scale of node distances)
            for node_end_2 in local_graph.neighbors(node_end):
                if node_end_2 != node:
                    break
            points = np.array([
                local_graph.nodes[node_end_2]["pos"],
                local_graph.nodes[node_end]["pos"],
                local_graph.nodes[node]["pos"]
            ])
        # Otherwise (it is a startpoint), we keep the proximal part
        else:
            # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
            for node_end_2 in local_graph.neighbors(node_end):
                if node_end_2 != node and local_graph.nodes[node_end_2][f"hierarchy {access}"] > local_graph.nodes[node_end][f"hierarchy {access}"]:
                    break
            points = np.array([
                local_graph.nodes[node]["pos"],
                local_graph.nodes[node_end]["pos"],
                local_graph.nodes[node_end_2]["pos"]
            ])

    # Normal segment (degree == 2)
    elif local_graph.degree[node] == 2:
        # This is the regular segment case. First we recognize the proximal and distal nodes depending on the node hierarchical index
        # Curvature is computed with preferably 5 point (if available) for nodes with degree == 2
        points = np.array([local_graph.nodes[node]["pos"]])
        for node_aux in local_graph.neighbors(node):
            if local_graph.nodes[node_aux][f"hierarchy {access}"] < local_graph.nodes[node][f"hierarchy {access}"]:
                node_prox = node_aux
                # To compute the curvature, we take the position of the two more proximal nodes (if available) (we compute curvature at a scale of node distances)
                points = np.insert(points, 0, [local_graph.nodes[node_prox]["pos"]], axis = 0)
                if local_graph.degree(node_prox) == 2:
                    for node_prox_2 in local_graph.neighbors(node_prox):
                        if node_prox_2 != node and local_graph.nodes[node_prox_2][f"hierarchy {access}"] < local_graph.nodes[node_prox][f"hierarchy {access}"]:
                            points = np.insert(points, 0, [local_graph.nodes[node_prox_2]["pos"]], axis = 0)
            elif local_graph.nodes[node_aux][f"hierarchy {access}"] > local_graph.nodes[node][f"hierarchy {access}"]:
                node_dist = node_aux
                # To compute the curvature, we take the position of the two more distal nodes (if available) (we compute curvature at a scale of node distances)
                points = np.append(points, [local_graph.nodes[node_dist]["pos"]], axis = 0)
                if local_graph.degree(node_dist) == 2:
                    for node_dist_2 in local_graph.neighbors(node_dist):
                        if node_dist_2 != node and local_graph.nodes[node_dist_2][f"hierarchy {access}"] > local_graph.nodes[node_dist][f"hierarchy {access}"]:
                            points = np.append(points, [local_graph.nodes[node_dist_2]["pos"]], axis = 0)

    # Multi-furcations (degree > 2)
    elif local_graph.degree[node] > 2:
        # We treat multi-furcation as endpoints (since there is no preference a priori for the path that needs to be taken). We only look at the preceeding node
        # First we search the neighboring nodes and we get the candidate centerline points for segment feature extraction from the preceeding node
        # To find the preceeding node, we check the hierarchical order (there should be one node with a lower hierarchical index than the bifurcation point)
        # We have to initialize the segment arrays just in case we are in the rare event of a multifurcation with hierarchy = 0
        node_prox = None
        for node_aux in local_graph.neighbors(node):
            if local_graph.nodes[node_aux][f"hierarchy {access}"] < local_graph.nodes[node][f"hierarchy {access}"]:
                node_prox = node_aux
                break
        # Multi-furcation as start node. We just take the first neighbor to compute all variables. In this case, we invert the roles of node and node_prox
        if node_prox is None:
            original_node = node
            for node_aux in local_graph.neighbors(node):
                node_prox = original_node
                node = node_aux
                # We just take a look at any neighbor node
                break
            
            # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
            for node_aux_2 in local_graph.neighbors(node_aux):
                if node_aux_2 != node:
                    break
            points = np.array([
                local_graph.nodes[node]["pos"],
                local_graph.nodes[node_aux]["pos"],
                local_graph.nodes[node_aux_2]["pos"]
            ])

            # At the end, we set node back to its original value
            node = original_node

        else:
            # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
            for node_aux_2 in local_graph.neighbors(node_aux):
                if node_aux_2 != node and local_graph.nodes[node_aux_2][f"hierarchy {access}"] < local_graph.nodes[node_aux][f"hierarchy {access}"]:
                    break
            points = np.array([
                local_graph.nodes[node]["pos"],
                local_graph.nodes[node_aux]["pos"],
                local_graph.nodes[node_aux_2]["pos"]
            ])
    return points

def sanity_check(local_graph, access="femoral"):
    """
    Sanity check for feature extraction. Checks for nan or inf values in the features and replaces them with the value 
    of the closest node.

    Parameters
    ----------
    local_graph : network.Graph
        Dense centerline graph.
    access : string
        Access site for thrombectomy configuration. Can be either "femoral" or "radial".

    Returns
    -------

    """
    # We finally check all nodes not to have any nan or inf values
    # If present, we choose the value from the neighboring nodes
    # We first check fist node (there is only one node with hierarchy equal to 0). This way, we ensure no error propagation and
    # that all nodes will not have nan or inf feature values:
    for node in local_graph:
        if local_graph.nodes[node][f"hierarchy {access}"] == 0:
            break

    num_max_iterations = 200
    for feature_key in local_graph.nodes[node][f"features {access}"].keys():
        # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
        current_node = node
        iteration = 0
        while math.isnan(local_graph.nodes[node][f"features {access}"][feature_key]) or math.isinf(local_graph.nodes[node][f"features {access}"][feature_key]):
            for neighbor in local_graph.neighbors(current_node):
                # Now we choose first following node to also include first node
                if local_graph.nodes[current_node][f"hierarchy {access}"] < local_graph.nodes[neighbor][f"hierarchy {access}"] and not math.isnan(local_graph.nodes[neighbor][f"features {access}"][feature_key]) and not math.isinf(local_graph.nodes[neighbor][f"features {access}"][feature_key]):
                    local_graph.nodes[node][f"features {access}"][feature_key] = local_graph.nodes[neighbor][f"features {access}"][feature_key]
                    break
            # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
            # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
            current_node = neighbor
            iteration += 1
            if iteration > num_max_iterations:
                local_graph.nodes[node][f"features {access}"][feature_key] = 0 # From experience, we see that this only happens with blanking in node 0 (very rare)
                                                                                    # We will just hard-code it to 0
                # raise Exception("A suitable neighbor could not be found for feature {} in node {}".format(feature_key, node))
    
    # Now we check all other nodes (differently from looking at the first node, we look at nodes with lower hierarchy)
    for node in local_graph:
        if local_graph.nodes[node][f"hierarchy {access}"] > 0:
            for feature_key in local_graph.nodes[node][f"features {access}"].keys():
                # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
                current_node = node
                iteration = 0
                while math.isnan(local_graph.nodes[node][f"features {access}"][feature_key]) or math.isinf(local_graph.nodes[node][f"features {access}"][feature_key]):
                    for neighbor in local_graph.neighbors(current_node):
                        # Now we choose first following node to also include first node
                        if local_graph.nodes[current_node][f"hierarchy {access}"] > local_graph.nodes[neighbor][f"hierarchy {access}"] and not math.isnan(local_graph.nodes[neighbor][f"features {access}"][feature_key]) and not math.isinf(local_graph.nodes[neighbor][f"features {access}"][feature_key]):
                            local_graph.nodes[node][f"features {access}"][feature_key] = local_graph.nodes[neighbor][f"features {access}"][feature_key]
                            break
                    # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
                    # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
                    current_node = neighbor
                    iteration += 1
                    if iteration > num_max_iterations:
                        local_graph.nodes[node][f"features {access}"][feature_key] = 0 # From experience, we see that this only happens with blanking in node 0 (very rare)
                                                                                            # We will just hard-code it to 0
                        # raise Exception("A suitable neighbor could not be found for feature {} in node {}".format(feature_key, node))

def add_cumulative_features(local_graph, access="femoral"):
    """
    Computes accumulative features for a centerline graph. That can only be computed after 
    all other features have been computed. Updates the graph in place.

    Parameters
    ----------
    local_graph : network.Graph
        Dense centerline graph after featurization.
    access : string
        Access site for thrombectomy configuration. Can be either "femoral" or "radial".

    Returns
    -------

    """
    # If we are now performing final feature extraction for supersegment treatment, we compute accumulative features that rely on a proper hierarchical ordering (need subgraph union)
    # Accumulative features
    for hierarchy in range(get_max_hierarchy(local_graph, access) + 1):
        nodes_hierarchy = [node for node in local_graph if local_graph.nodes[node][f"hierarchy {access}"] == hierarchy]
        for node in nodes_hierarchy:
            # For the first node, we just set to 0
            if hierarchy == 0:
                local_graph.nodes[node][f"features {access}"]["accumulated length from access"] = 0
            else:
                neighbor_found = False
                for neighbor in local_graph.neighbors(node):
                    if local_graph.nodes[node][f"hierarchy {access}"] > local_graph.nodes[neighbor][f"hierarchy {access}"]:
                        # We add the distance between nodes instead (marked by empty indices vector in edge features)
                        local_graph.nodes[node][f"features {access}"]["accumulated length from access"] = local_graph.nodes[neighbor][f"features {access}"]["accumulated length from access"] + np.linalg.norm(local_graph.nodes[node]["pos"] - local_graph.nodes[neighbor]["pos"])
                        neighbor_found = True
                        break
                if not neighbor_found:
                    local_graph.nodes[node][f"features {access}"]["accumulated length from access"] = 0

def get_max_hierarchy(local_graph, access="femoral"):
    """ 
    Iterates over nodes to find max hierarchy for each access configuration.

    Parameters
    ----------
    local_graph : network.Graph
        Dense centerline graph returned by graph builder.
    access : string
        Access site for thrombectomy configuration. Can be either "femoral" or "radial".

    Returns
    -------
    max_hierarchy : integer
        Maximum hierarchical index for `access`.
    
    """
    max_hierarchy = 0
    for node in local_graph:
        if local_graph.nodes[node][f"hierarchy {access}"] > max_hierarchy:
            max_hierarchy = local_graph.nodes[node][f"hierarchy {access}"]
                   
    return max_hierarchy

def compute_spherical_angles(vec):
    """
    Returns spherical angles of a vector in cartesian coordinates.

    Parameters
    ----------
    vec : numpy.array or array-like object, shape [3].
        Three-dimensional vector between two cartesian points.

    Returns
    -------
    polar : float
        Polar angle in spherical coordinates. Contained between -pi / 2 and pi / 2.
    azimuth : float
        Azimuth angle in spherical coordinates. Contained between -pi and pi.

    """
    x, y, z = vec
    # We compute the module of the vector
    module = np.linalg.norm(vec)

    # For the polar angle, we consider the case when z could be 0
    if abs(z) < 1e-5:
        # If any x or y is different than 0, polar is pi / 2
        if abs(x) > 1e-5 or abs(y) > 1e-5:
            polar = 0
        # Otherwise, we are in the case when vec == [0, 0, 0]
        else:
            polar = math.nan
    # Otherwise compute polar angle normally
    else: 
        polar = np.sign(z) * math.pi / 2 - math.atan((x ** 2 + y ** 2) ** 0.5 / z) 

    # Consider the case when x and y are 0 (azimuth undefined)
    if abs(x) < 1e-5 and abs(y) < 1e-5:
        azimuth = math.nan
    # Otherwise, use the general formula
    else:
        azimuth = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))

    return module, polar, azimuth

def compute_curvature_and_torsion(curve, node_idx):
    """
    Computes curvature and torsion along a 3D curve.

    Parameters
    ----------
    curve : (N, 3) numpy array
        Ordered points of the curve.
    node_idx : int
        Index of the node at which curvature and torsion are returned.

    Returns
    -------
    curvature : float
        Curvature at node_idx.
    torsion : float
        Torsion at node_idx.
    """
    # First derivative dr/dt
    drdt = np.gradient(curve, axis=0)
    # dt/ds = 1 / |dr/dt|
    speed = np.linalg.norm(drdt, axis=1)
    dtds = 1.0 / speed
    # Unit tangent vector T = dr/ds
    tangent = drdt * dtds[:, None]
    # dT/dt
    dTdt = np.gradient(tangent, axis=0)
    # dT/ds = (dT/dt) * (dt/ds)
    dTds = dTdt * dtds[:, None]
    # Curvature κ = |dT/ds|
    curvature = np.linalg.norm(dTds, axis=1)
    # Unit normal vector N
    normal = dTds / np.linalg.norm(dTds, axis=1)[:, None]
    # Binormal vector B = T × N
    binormal = np.cross(tangent, normal)
    # dB/dt
    dBdt = np.gradient(binormal, axis=0)
    # dB/ds
    dBds = dBdt * dtds[:, None]
    # Torsion τ = - dB/ds · N
    torsion = -np.sum(dBds * normal, axis=1)

    return curvature[node_idx], torsion[node_idx]

def find_point_id(point, model_coordinates):
    """
    Given a point in space and a set of coordinates from a volume model,
    it finds the index of the model coordinates closest to the point of interest.

    Parameters
    ----------
    point : numpy.array or array-like object, shape [3].
        Point of interest.
    model_coordinates : numpy.array or array-like object.
        Array of coordinates from a volume model.

    Returns
    -------
    index : integer
        Index of the model coordinates closest to the point of interest.

    """
    # We compute the norm of the distance from the point to all model points and 
    # get the argument of the closest
    index = np.argmin(np.linalg.norm(model_coordinates - point, axis = 1))
    
    return index