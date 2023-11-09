#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import vtk

import numpy as np
import nibabel as nib

from vtk.util.numpy_support import vtk_to_numpy

from arterial.feature_extraction.local_features.utils import featurize_node, sanity_check, add_cumulative_features
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

    # Compute local features from both accesses
    for access in ["femoral", "radial"]:
        # If edges were artificially added for hierarchical ordering, remove them before feature extraction
        for src, dst in centerline_graph.graph["subgraphs_union_edges"]:
            centerline_graph.remove_edge(src, dst)
        # Also, add an extra attribute to identify which ones are artificial
        for src, dst in centerline_graph.edges:
            centerline_graph[src][dst]["is_artificial"] = False
        # We need to iterate over all graph nodes and generalize the feature extraction process depending on the degree of the node
        for node in centerline_graph:
            featurize_node(node, centerline_graph, access, radius_branch_model, branch_model_coordinates, blanking, cta_array_data, aff, lpi_corner_coordinates)

        # If we had removed edges in the beggining, we add them again to compute accumulated features
        for src, dst in centerline_graph.graph["subgraphs_union_edges"]:
            centerline_graph.add_edge(src, dst, cell_id = centerline_graph.nodes[dst]["cell_id"])
            centerline_graph[src][dst]["vessel_type"] = centerline_graph.nodes[dst]["vessel_type"]
            centerline_graph[src][dst]["vessel_type_name"] = centerline_graph.nodes[dst]["vessel_type_name"]
            centerline_graph[src][dst]["is_artificial"] = True
            # Empty indices to identify artificial edges
            centerline_graph[src][dst]["indices"] = np.array([])

        # Checks for nan features, imputing those by the closes node with a valid feature
        sanity_check(centerline_graph, access)
        # Compute cumulative features
        add_cumulative_features(centerline_graph, access)

        # We also add the vessel label as node feature
        for node in centerline_graph:
            centerline_graph.nodes[node][f"features {access}"]["vessel_type"] = centerline_graph.nodes[node]["vessel_type"]

    # Overwrite centerline_graph
    save_pickle(centerline_graph, os.path.join(case_dir, "dense_graph.pickle"))

    return centerline_graph