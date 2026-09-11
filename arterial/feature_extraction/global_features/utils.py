#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import numpy as np

def get_aortic_arch_node_coordinates(centerline_graph):
    """
    Gets all coordinates of the aortic arch to be used as reference for
    segment orientation.

    Parameters
    ----------
    centerline_graph: networkx.Graph
        Dense centerline graph with vessel types and node position as node 
        attributes.
    
    Returns
    -------
    aortic_arch_coordinates : numpy.array
        Array with all node coordinates of all aortic arch nodes.
    
    """
    # Initialize array
    aortic_arch_coordinates = np.ndarray([0, 3], dtype = float)
    # Iterate over all nodes
    for node in centerline_graph:
        # Collect node posiiton of those with AA vessel type
        if centerline_graph.nodes[node]["vessel_type_name"] == "AA":
            aortic_arch_coordinates = np.append(aortic_arch_coordinates, [centerline_graph.nodes[node]["pos"]], axis = 0)

    return aortic_arch_coordinates

def get_aortic_arch_type(centerline_graph):
    """
    Gets aortic arch type. The aortic arch type is computed by comparing
    the axial distance between the aortic arch (AA) apex (A) and the brachiocephalic
    trunk (BT) origin (B) with the left common carotid artery (LCCA) proximal diameter 
    (D). The ratio between both determines the AA type (1, 2 or 3):

    $$ ratio = (A - B) / D $$

    * If ratio < 1: aortic_arch_type = 1
    * If ratio > 1 and ratio < 2: aortic_arch_type = 2
    * If ratio > 2: aortic_arch_type = 3
    
    Parameters
    ----------
    centerline_graph: networkx.Graph
        Dense centerline graph with vessel types and node position as node 
        attributes.
    
    Returns
    -------
    aortic_arch_type : integer
        Aortic arch type. Can be 1, 2 o 3.
        
    """
    # Get all AA coordinates
    aortic_arch_coordinates = get_aortic_arch_node_coordinates(centerline_graph)
    # Initialize most proximal distance of BT and LCCA to AA with very high 
    # values to make sure these options are found
    closest_bt_node_distance, closest_lcca_node_distance = 100, 100
    # Initialize A at 0 and B and D as None
    A = 0
    B, D = None, None
    # Search for A, B and D
    for node in centerline_graph:
        if centerline_graph.nodes[node]["vessel_type_name"] == "AA" and centerline_graph.nodes[node]["pos"][2] + centerline_graph.nodes[node]["radius"] > A:
            # A will approximately be the position of the highest AA point plus its radius
            A = centerline_graph.nodes[node]["pos"][2] + centerline_graph.nodes[node]["radius"]
        if centerline_graph.nodes[node]["vessel_type_name"] == "BT" and centerline_graph.nodes[node]["features femoral"]["blanking"] == 0 and np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1)) < closest_bt_node_distance:
            # B will be the BT node closest to the AA without blanking
            B = centerline_graph.nodes[node]["pos"][2]
            closest_bt_node_distance = np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1))
        if centerline_graph.nodes[node]["vessel_type_name"] == "LCCA" and centerline_graph.nodes[node]["features femoral"]["blanking"] == 0 and np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1)) < closest_lcca_node_distance:
            # D will be computed at the LCCA node closest to the AA without blanking
            D = 2 * centerline_graph.nodes[node]["radius"]
            closest_lcca_node_distance = np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1))

    # Finally, compute the ratio to determine the AA type
    if A > 0 and B is not None and D is not None:
        value = np.abs(A - B) / D
        if value < 1:
            aortic_arch_type = 1
        elif value < 2:
            aortic_arch_type = 2
        else:
            aortic_arch_type = 3
    else:
        aortic_arch_type = 1

    return aortic_arch_type

def get_bovine_arch(centerline_graph):
    """
    Gets bovine arch presence. Gets most proximal LCCA node and detects segments in 
    contact at origin. If the BT is one of the vessels in contact at the LCCA origin,
    it detects the bovine aortic arch. Otherwise, it does not.

    Parameters
    ----------
    centerline_graph: networkx.Graph
        Dense centerline graph with vessel types and node position as node 
        attributes.
    
    Returns
    -------
    bovine_arch : integer
        Can be 0 if bovine arch is not detected or 1 if it is.
    
    """
    # Get all AA coordinates
    aortic_arch_coordinates = get_aortic_arch_node_coordinates(centerline_graph)
    # Initialize most proximal distance of LCCA to AA with very high 
    # values to make sure these options are found
    closest_lcca_node_distance = 100
    # Initialize the closest LCCA node at None
    closest_lcca_node = None
    # Search for the closest LCCA node
    for node in centerline_graph:
        if centerline_graph.nodes[node]["vessel_type_name"] == "LCCA" and np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1)) < closest_lcca_node_distance:
            closest_lcca_node = node
            closest_lcca_node_distance = np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1))

    # If found, and not artificial (placed at graph unification), get vessel types in contacts
    if closest_lcca_node is not None:
        vessel_type_names_in_contact = []
        for neighbor in centerline_graph.neighbors(closest_lcca_node):
            for src, dst in centerline_graph.edges(neighbor):
                if not centerline_graph[src][dst]["is_artificial"]:
                    vessel_type_names_in_contact.append(centerline_graph[src][dst]["vessel_type_name"])
        # If the BT (type 2) is found in contact, bovine arch is detected
        if "BT" in vessel_type_names_in_contact:
            bovine_arch = 1
        else:
            bovine_arch = 0
    else:
        bovine_arch = 0

    return bovine_arch

def get_arsa(centerline_graph):
    """
    Gets presence of aberrant RSA (ARSA). Gets most proximal RSA and LSA nodes,
    taking the AA as reference. If the AA point in contact at the RSA origin
    has a smaller hierarchy than the LSA's, ARSA is detected.

    *Potential improvements:
    - Add extra criteria to make it more reliable. 
    - RSA linked to an AA point closer to feamoral startpoint than LSA?
    - Get AA closest node: point A
    - Get point A hierarchy
    - Repeat computation with LSA: point B
    - Get pointy B hierarchy
    - If hierarchy B > hierarchy A, and AA in vessel_type_names_in_contact, ARSA. Else, not ARSA*

    Parameters
    ----------
    centerline_graph: networkx.Graph
        Dense centerline graph with vessel types and node position as node 
        attributes.

    Returns
    -------
    arsa : integer
        Can be 0 if ARSA is not detected or 1 if it is.
    
    """
    # Get all AA coordinates
    aortic_arch_coordinates = get_aortic_arch_node_coordinates(centerline_graph)
    # Initialize most proximal distance of RSA and LSA to AA with very high 
    # values to make sure these options are found
    closest_rsa_node_distance, closest_lsa_node_distance = 100, 100
    # Initialize the closest RSA and LSA nodes at None
    closest_rsa_node, closest_lsa_node = None, None
    # Iterate over all graph nodes to find RSA and LSA origin nodes
    for node in centerline_graph:
        if centerline_graph.nodes[node]["vessel_type_name"] == "RSA" and np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1)) < closest_rsa_node_distance:
            closest_rsa_node = node
            closest_rsa_node_distance = np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1))
        if centerline_graph.nodes[node]["vessel_type_name"] == "LSA" and np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1)) < closest_lsa_node_distance:
            closest_lsa_node = node
            closest_lsa_node_distance = np.amin(np.linalg.norm(aortic_arch_coordinates - centerline_graph.nodes[node]["pos"], axis = 1))
    # If RSA origin is found, get segments in contact of non-artificial edges
    if closest_rsa_node is not None:
        vessel_type_names_in_contact = []
        for neighbor in centerline_graph.neighbors(closest_rsa_node):
            for src, dst in centerline_graph.edges(neighbor):
                if not centerline_graph[src][dst]["is_artificial"]:
                    vessel_type_names_in_contact.append(centerline_graph[src][dst]["vessel_type_name"])
        # If AA is in contact of the RSA origin and the closest RSA node has a smaller hierarchy 
        # than the closest LSA node, ARSA is detected
        if "AA" in vessel_type_names_in_contact and closest_lsa_node is not None and centerline_graph.nodes[closest_rsa_node]["hierarchy femoral"] < centerline_graph.nodes[closest_lsa_node]["hierarchy femoral"]:
            arsa = 1
        else:
            arsa = 0
    else:
        arsa = 0

    return arsa