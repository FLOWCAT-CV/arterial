#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import numpy as np
import nibabel as nib

from vtk.util.numpy_support import vtk_to_numpy

from arterial.feature_extraction.local_features.utils import featurize_node, sanity_check, add_cumulative_features

def perform_local_feature_extraction(local_graph, cta_array, cta_affine, branch_model):
    """
    Function for local graph featurization. Inputs a networkx.Graph from a case and 
    returns the same graph with node attributes for both femoral and radial accesses.

    Local features include:
    * Node position
    * Radius
    * Curvature
    * Relative position to neighbor nodes
    * segment distance
    * Directional features
    * Other features (CTA intensity, blanking)
    * Vessel type

    Parameters
    ----------
    local_graph : networkx.Graph
        Dense centerline graph returned by graph builder.
    cta_array : np.ndarray
        3D array of the CTA image.
    cta_affine : np.ndarray
        Affine matrix of the CTA image.
    branch_model : vtk.vtkPolyData
        Branch model of the centerline graph.

    Returns
    -------
    local_graph : networkx.Graph
        Featurized centerline graph with node attributes for both femoral and radial 
        accesses.

    """
    # Depending on the orientation of the image, we have to define the corner voxel coordinates and the flipping array
    orientation = nib.aff2axcodes(cta_affine)
    if orientation == ('R', 'A', 'S'):
        lpi_corner_voxel_coordinates = np.array([0, 0, 0])
    elif orientation == ('L', 'A', 'S'):
        lpi_corner_voxel_coordinates = np.array([cta_array.shape[0] - 1, 0, 0])
    elif orientation == ('L', 'P', 'S'):
        lpi_corner_voxel_coordinates = np.array([cta_array.shape[0] - 1, cta_array.shape[1] - 1, 0])

    # Compute lpi corner coordinates in real world coordinates, with the same orientation as the image
    lpi_corner_coordinates = np.dot(cta_affine, np.append(lpi_corner_voxel_coordinates, 1))[:3]
    # Pool branch_model point points. Get blanking for each point
    branch_model_coordinates = np.ndarray([branch_model.GetNumberOfPoints(), 3])
    blanking = np.ndarray([branch_model.GetNumberOfPoints()])
    radius_branch_model = vtk_to_numpy(branch_model.GetPointData().GetArray("MaximumInscribedSphereRadius"))
    accumulated_number_of_points = 0
    for idx in range(branch_model.GetNumberOfCells()):         
        for idx2 in range(branch_model.GetCell(idx).GetNumberOfPoints()):
            branch_model_coordinates[idx2 + accumulated_number_of_points] = branch_model.GetCell(idx).GetPoints().GetPoint(idx2) - lpi_corner_coordinates
            blanking[idx2 + accumulated_number_of_points] = vtk_to_numpy(branch_model.GetCellData().GetArray("Blanking"))[idx]
        accumulated_number_of_points += branch_model.GetCell(idx).GetNumberOfPoints()

    # Compute local features from both accesses
    for access in ["femoral", "radial"]:
        # If edges were artificially added for hierarchical ordering, remove them before feature extraction
        for src, dst in local_graph.graph["subgraphs_union_edges"]:
            local_graph.remove_edge(src, dst)
        # Also, add an extra attribute to identify which ones are artificial
        for src, dst in local_graph.edges:
            local_graph[src][dst]["is_artificial"] = False
        # We need to iterate over all graph nodes and generalize the feature extraction process depending on the degree of the node
        for node in local_graph:
            featurize_node(local_graph, node, access, radius_branch_model, branch_model_coordinates, blanking, cta_array, cta_affine, lpi_corner_coordinates)

        # If we had removed edges in the beggining, we add them again to compute accumulated features
        for src, dst in local_graph.graph["subgraphs_union_edges"]:
            local_graph.add_edge(src, dst, cell_id = local_graph.nodes[dst]["cell_id"])
            local_graph[src][dst]["vessel_type"] = local_graph.nodes[dst]["vessel_type"]
            local_graph[src][dst]["vessel_type_name"] = local_graph.nodes[dst]["vessel_type_name"]
            local_graph[src][dst]["is_artificial"] = True
            # Empty indices to identify artificial edges
            local_graph[src][dst]["indices"] = np.array([])

        # Checks for nan features, imputing those by the closes node with a valid feature
        sanity_check(local_graph, access)
        # Compute cumulative features
        add_cumulative_features(local_graph, access)

        # We also add the vessel label as node feature
        for node in local_graph:
            local_graph.nodes[node][f"features {access}"]["vessel_type"] = local_graph.nodes[node]["vessel_type"]

    return local_graph