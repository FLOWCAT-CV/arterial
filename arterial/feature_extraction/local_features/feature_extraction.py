#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import vtk
import math

import numpy as np
import nibabel as nib

from vtk.util.numpy_support import vtk_to_numpy

from arterial.feature_extraction.local_features.utils import get_max_hierarchy, compute_spherical_angles, compute_curvature_and_torsion, find_point_id
from arterial.io.load_and_save_operations import save_pickle

def perform_local_feature_extraction(case_dir, centerline_graph):
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

    This function overwrites the existing dense centerline graph:

    >>> case_dir/graph.pickle

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Dense centerline graph returned by graph builder.

    Returns
    -------
    centerline_graph : networkx.Graph
        Featurized centerline graph with node attributes for both femoral and radial 
        accesses.

    """
    # Load nifti to define data, image_shape and aff
    if os.path.isfile(os.path.join(case_dir, "{}_cta.nii.gz".format(os.path.basename(case_dir)))):
        nifti = nib.load(os.path.join(case_dir, "{}_cta.nii.gz".format(os.path.basename(case_dir))))
    else:
        nifti = nib.load(os.path.join(case_dir, "{}_extracranial_vessels_segmentation.nii.gz".format(os.path.basename(case_dir))))
    cta_array_data = nifti.get_fdata()
    image_shape = cta_array_data.shape
    aff = nifti.affine
    # Depending on the orientation of the image, we have to define the corner voxel coordinates and the flipping array
    orientation = nib.aff2axcodes(aff)
    if orientation == ('R', 'A', 'S'):
        lpi_corner_voxel_coordinates = np.array([0, 0, 0])
    elif orientation == ('L', 'A', 'S'):
        lpi_corner_voxel_coordinates = np.array([image_shape[0] - 1, 0, 0])
    elif orientation == ('L', 'P', 'S'):
        lpi_corner_voxel_coordinates = np.array([image_shape[0] - 1, image_shape[1] - 1, 0])

    # Compute lpi corner coordinates in real world coordinates, with the same orientation as the image
    lpi_corner_coordinates = np.dot(aff, np.append(lpi_corner_voxel_coordinates, 1))[:3]
    # Compute translation from affine matrix
    translation = np.transpose(aff[:3, 3])
    # Load branch_model
    vtk_poly_data_reader = vtk.vtkPolyDataReader()
    vtk_poly_data_reader.SetFileName(os.path.join(case_dir, "branch_model.vtk"))
    vtk_poly_data_reader.Update()
    branch_model = vtk_poly_data_reader.GetOutput()
    # Pool branch_model point points. Get blanking for each point
    branch_model_coordinates = np.ndarray([branch_model.GetNumberOfPoints(), 3])
    blanking = np.ndarray([branch_model.GetNumberOfPoints()])
    radius_branch_model = vtk_to_numpy(branch_model.GetPointData().GetArray("Radius"))
    accumulated_number_of_points = 0
    for idx in range(branch_model.GetNumberOfCells()):         
        for idx2 in range(branch_model.GetCell(idx).GetNumberOfPoints()):
            branch_model_coordinates[idx2 + accumulated_number_of_points] = branch_model.GetCell(idx).GetPoints().GetPoint(idx2) - lpi_corner_coordinates
            blanking[idx2 + accumulated_number_of_points] = vtk_to_numpy(branch_model.GetCellData().GetArray("Blanking"))[idx]
        accumulated_number_of_points += branch_model.GetCell(idx).GetNumberOfPoints()

    # Get subgraph union edges, should be at global features of centerline graph
    subgraphs_union_edges = centerline_graph.graph["subgraphs_union_edges"]

    for access in ["femoral", "radial"]:
        # If edges were artificially added for hierarchical ordering, remove them before feature extraction
        for src, dst in subgraphs_union_edges:
            centerline_graph.remove_edge(src, dst)
        # Also, add an extra attribute to identify which ones are artificial
        for src, dst in centerline_graph.edges:
            centerline_graph[src][dst]["is_artificial"] = False
        # We need to iterate over all graph nodes and generalize the feature extraction process depending on the degree of the node
        for node in centerline_graph:
            # Endpoints (degree == 1)
            # For endpoints, we gather information from the endpoint node's relationship to its neighbor (node_end)
            if centerline_graph.degree[node] == 1:
                # We get the position of the node
                node_pos = centerline_graph.nodes[node]["pos"]
                # First we search the neighboring node and we get the candidate centerline points for segment feature extraction
                for node_end in centerline_graph.neighbors(node):
                    break
                # Since we are analyzing an endpoint segment, we have to check if the analyzed node is an endpoint or a startpoint
                # We can use the node hierarchy computed for the dense graph for that
                # If the node hierarchy is greater than the alternative node, then we need to keep the distal part of the centerline segment for the feature extraction
                if centerline_graph.nodes[node][f"hierarchy {access}"] > centerline_graph.nodes[node_end][f"hierarchy {access}"]: 
                    # To compute the curvature, we take the position of the last three nodes (we compute curvature at a scale of node distances)
                    for node_end_2 in centerline_graph.neighbors(node_end):
                        if node_end_2 != node:
                            break
                    points = np.array([
                        centerline_graph.nodes[node_end_2]["pos"],
                        centerline_graph.nodes[node_end]["pos"],
                        centerline_graph.nodes[node]["pos"]
                    ])
                # Otherwise (it is a startpoint), we keep the proximal part
                else:
                    # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
                    for node_end_2 in centerline_graph.neighbors(node_end):
                        if node_end_2 != node and centerline_graph.nodes[node_end_2][f"hierarchy {access}"] > centerline_graph.nodes[node_end][f"hierarchy {access}"]:
                            break
                    points = np.array([
                        centerline_graph.nodes[node]["pos"],
                        centerline_graph.nodes[node_end]["pos"],
                        centerline_graph.nodes[node_end_2]["pos"]
                    ])

            # Normal segment (degree == 2)
            elif centerline_graph.degree[node] == 2:
                # We get the node position    
                node_pos = centerline_graph.nodes[node]["pos"]
                # This is the regular segment case. First we recognize the proximal and distal nodes depending on the node hierarchical index
                # Curvature is computed with preferably 5 point (if available) for nodes with degree == 2
                points = np.array([centerline_graph.nodes[node]["pos"]])
                for node_aux in centerline_graph.neighbors(node):
                    if centerline_graph.nodes[node_aux][f"hierarchy {access}"] < centerline_graph.nodes[node][f"hierarchy {access}"]:
                        node_prox = node_aux
                        # To compute the curvature, we take the position of the two more proximal nodes (if available) (we compute curvature at a scale of node distances)
                        points = np.insert(points, 0, [centerline_graph.nodes[node_prox]["pos"]], axis = 0)
                        if centerline_graph.degree(node_prox) == 2:
                            for node_prox_2 in centerline_graph.neighbors(node_prox):
                                if node_prox_2 != node and centerline_graph.nodes[node_prox_2][f"hierarchy {access}"] < centerline_graph.nodes[node_prox][f"hierarchy {access}"]:
                                    points = np.insert(points, 0, [centerline_graph.nodes[node_prox_2]["pos"]], axis = 0)
                    elif centerline_graph.nodes[node_aux][f"hierarchy {access}"] > centerline_graph.nodes[node][f"hierarchy {access}"]:
                        node_dist = node_aux
                        # To compute the curvature, we take the position of the two more distal nodes (if available) (we compute curvature at a scale of node distances)
                        points = np.append(points, [centerline_graph.nodes[node_dist]["pos"]], axis = 0)
                        if centerline_graph.degree(node_dist) == 2:
                            for node_dist_2 in centerline_graph.neighbors(node_dist):
                                if node_dist_2 != node and centerline_graph.nodes[node_dist_2][f"hierarchy {access}"] > centerline_graph.nodes[node_dist][f"hierarchy {access}"]:
                                    points = np.append(points, [centerline_graph.nodes[node_dist_2]["pos"]], axis = 0)

            # Multi-furcations (degree > 2)
            elif centerline_graph.degree[node] > 2:
                # We get the node position    
                node_pos = centerline_graph.nodes[node]["pos"]
                # We treat multi-furcation as endpoints (since there is no preference a priori for the path that needs to be taken). We only look at the preceeding node
                # First we search the neighboring nodes and we get the candidate centerline points for segment feature extraction from the preceeding node
                # To find the preceeding node, we check the hierarchical order (there should be one node with a lower hierarchical index than the bifurcation point)
                # We have to initialize the segment arrays just in case we are in the rare event of a multifurcation with hierarchy = 0
                node_prox = None
                for node_aux in centerline_graph.neighbors(node):
                    if centerline_graph.nodes[node_aux][f"hierarchy {access}"] < centerline_graph.nodes[node][f"hierarchy {access}"]:
                        node_prox = node_aux
                        break
                # Multi-furcation as start node. We just take the first neighbor to compute all variables. In this case, we invert the roles of node and node_prox
                if node_prox is None:
                    original_node = node
                    for node_aux in centerline_graph.neighbors(node):
                        node_prox = original_node
                        node = node_aux
                        # We just take a look at any neighbor node
                        break
                    
                    # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
                    for node_aux_2 in centerline_graph.neighbors(node_aux):
                        if node_aux_2 != node:
                            break
                    points = np.array([
                        centerline_graph.nodes[node]["pos"],
                        centerline_graph.nodes[node_aux]["pos"],
                        centerline_graph.nodes[node_aux_2]["pos"]
                    ])

                    # At the end, we set node and node_pos back to their original value
                    node = original_node
                    node_pos = centerline_graph.nodes[node]["pos"]

                else:
                    # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
                    for node_aux_2 in centerline_graph.neighbors(node_aux):
                        if node_aux_2 != node and centerline_graph.nodes[node_aux_2][f"hierarchy {access}"] < centerline_graph.nodes[node_aux][f"hierarchy {access}"]:
                            break
                    points = np.array([
                        centerline_graph.nodes[node]["pos"],
                        centerline_graph.nodes[node_aux]["pos"],
                        centerline_graph.nodes[node_aux_2]["pos"]
                    ])

            # Get the index of the node among the points
            pos = centerline_graph.nodes[node]["pos"]
            node_idx = np.argmin(np.linalg.norm(points - pos, axis = 1))

            # Feature extraction           
            centerline_graph.nodes[node][f"features {access}"] = {}
            
            # Node position
            # We have to compute the ijk positions for normalization purposes (either this or subtract the translation from the affine matrix)
            centerline_graph.nodes[node][f"features {access}"]["pos r"] = node_pos[0]
            centerline_graph.nodes[node][f"features {access}"]["pos a"] = node_pos[1]
            centerline_graph.nodes[node][f"features {access}"]["pos s"] = node_pos[2]
            
            # Radius features
            centerline_graph.nodes[node][f"features {access}"]["radius"] = radius_branch_model[find_point_id(node_pos, branch_model_coordinates)]
            # Segment length
            centerline_graph.nodes[node][f"features {access}"]["segment length"] = sum([np.linalg.norm(points[idx - 1] - points[idx]) for idx in range(1, len(points))]) / len(points)
            # Curvature and torsion (perhaps we could use the data from )
            if len(points) == 3:
                curvature, torsion = compute_curvature_and_torsion(points, 1)
            else:
                curvature, torsion = compute_curvature_and_torsion(points, node_idx)
            centerline_graph.nodes[node][f"features {access}"]["curvature"] = curvature
            centerline_graph.nodes[node][f"features {access}"]["torsion"] = torsion

            # Directional features
            module, polar, azimuth = compute_spherical_angles(points[-1] - points[0])
            centerline_graph.nodes[node][f"features {access}"]["direction module"] = module
            centerline_graph.nodes[node][f"features {access}"]["direction polar"] = polar
            centerline_graph.nodes[node][f"features {access}"]["direction azimuth"] = azimuth
            # Other features
            centerline_graph.nodes[node][f"features {access}"]["blanking"] = blanking[find_point_id(node_pos, branch_model_coordinates)]
            # Tranform to ijk
            node_pos_ijk = np.matmul(np.linalg.inv(aff), np.append(node_pos + lpi_corner_coordinates, 1.0))[:3]
            centerline_graph.nodes[node][f"features {access}"]["HU intensity"] = cta_array_data[np.round(node_pos_ijk).astype(int)[0], np.round(node_pos_ijk).astype(int)[1], np.round(node_pos_ijk).astype(int)[2]]

        # If we had removed edges in the beggining, we add them again to compute accumulated features
        for src, dst in subgraphs_union_edges:
            centerline_graph.add_edge(src, dst, cell_id = centerline_graph.nodes[dst]["cell_id"])
            centerline_graph[src][dst]["vessel_type"] = centerline_graph.nodes[dst]["vessel_type"]
            centerline_graph[src][dst]["vessel_type_name"] = centerline_graph.nodes[dst]["vessel_type_name"]
            centerline_graph[src][dst]["is_artificial"] = True
            # Empty indices to identify artificial edges
            centerline_graph[src][dst]["indices"] = np.array([])

        # We finally check all nodes not to have any nan or inf values
        # If present, we choose the value from the neighboring nodes
        # We first check fist node (there is only one node with hierarchy equal to 0). This way, we ensure no error propagation and
        # that all nodes will not have nan or inf feature values:
        for node in centerline_graph:
            if centerline_graph.nodes[node][f"hierarchy {access}"] == 0:
                break

        num_max_iterations = 200
        for feature_key in centerline_graph.nodes[node][f"features {access}"].keys():
            # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
            current_node = node
            iteration = 0
            while math.isnan(centerline_graph.nodes[node][f"features {access}"][feature_key]) or math.isinf(centerline_graph.nodes[node][f"features {access}"][feature_key]):
                for neighbor in centerline_graph.neighbors(current_node):
                    # Now we choose first following node to also include first node
                    if centerline_graph.nodes[current_node][f"hierarchy {access}"] < centerline_graph.nodes[neighbor][f"hierarchy {access}"] and not math.isnan(centerline_graph.nodes[neighbor][f"features {access}"][feature_key]) and not math.isinf(centerline_graph.nodes[neighbor][f"features {access}"][feature_key]):
                        centerline_graph.nodes[node][f"features {access}"][feature_key] = centerline_graph.nodes[neighbor][f"features {access}"][feature_key]
                        break
                # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
                # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
                current_node = neighbor
                iteration += 1
                if iteration > num_max_iterations:
                    centerline_graph.nodes[node][f"features {access}"][feature_key] = 0 # From experience, we see that this only happens with blanking in node 0 (very rare)
                                                                                        # We will just hard-code it to 0
                    print("A suitable neighbor could not be found for feature {} in node {}".format(feature_key, node))
                    # raise Exception("A suitable neighbor could not be found for feature {} in node {}".format(feature_key, node))
        
        # Now we check all other nodes (differently from looking at the first node, we look at nodes with lower hierarchy)
        for node in centerline_graph:
            if centerline_graph.nodes[node][f"hierarchy {access}"] > 0:
                for feature_key in centerline_graph.nodes[node][f"features {access}"].keys():
                    # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
                    current_node = node
                    iteration = 0
                    while math.isnan(centerline_graph.nodes[node][f"features {access}"][feature_key]) or math.isinf(centerline_graph.nodes[node][f"features {access}"][feature_key]):
                        for neighbor in centerline_graph.neighbors(current_node):
                            # Now we choose first following node to also include first node
                            if centerline_graph.nodes[current_node][f"hierarchy {access}"] > centerline_graph.nodes[neighbor][f"hierarchy {access}"] and not math.isnan(centerline_graph.nodes[neighbor][f"features {access}"][feature_key]) and not math.isinf(centerline_graph.nodes[neighbor][f"features {access}"][feature_key]):
                                centerline_graph.nodes[node][f"features {access}"][feature_key] = centerline_graph.nodes[neighbor][f"features {access}"][feature_key]
                                break
                        # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
                        # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
                        current_node = neighbor
                        iteration += 1
                        if iteration > num_max_iterations:
                            centerline_graph.nodes[node][f"features {access}"][feature_key] = 0 # From experience, we see that this only happens with blanking in node 0 (very rare)
                                                                                                # We will just hard-code it to 0
                            print("A suitable neighbor could not be found for feature {} in node {}".format(feature_key, node))
                            # raise Exception("A suitable neighbor could not be found for feature {} in node {}".format(feature_key, node))

        # If we are now performing final feature extraction for supersegment treatment, we compute accumulative features that rely on a proper hierarchical ordering (need subgraph union)
        # # Accumulative features
        for hierarchy in range(get_max_hierarchy(centerline_graph, access) + 1):
            for node in centerline_graph:
                if centerline_graph.nodes[node][f"hierarchy {access}"] == hierarchy:
                    # For the first node, we just set to 0
                    if hierarchy == 0:
                        centerline_graph.nodes[node][f"features {access}"]["accumulated length from access"] = 0
                    else:
                        for node_aux in centerline_graph.neighbors(node):
                            if centerline_graph.nodes[node][f"hierarchy {access}"] > centerline_graph.nodes[node_aux][f"hierarchy {access}"]:
                                # We add the distance between nodes instead (marked by empty indices vector in edge features)
                                centerline_graph.nodes[node][f"features {access}"]["accumulated length from access"] = centerline_graph.nodes[node_aux][f"features {access}"]["accumulated length from access"] + np.linalg.norm(centerline_graph.nodes[node]["pos"] - centerline_graph.nodes[node_aux]["pos"])
                                break

        # We also add the vessel label as node feature
        for node in centerline_graph:
            centerline_graph.nodes[node][f"features {access}"]["vessel_type"] = centerline_graph.nodes[node]["vessel_type"]

    # Overwrite centerline_graph
    save_pickle(centerline_graph, os.path.join(case_dir, "dense_graph.pickle"))

    return centerline_graph