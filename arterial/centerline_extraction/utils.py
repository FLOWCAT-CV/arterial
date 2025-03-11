#   Copyright 2024 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import numpy as np
import nibabel as nib

from skimage import measure
from scipy import ndimage
from scipy.spatial import cKDTree

from arterial.io.load_and_save_operations import *
    
def build_endpoints_json(endpoint_list, segmentation_affine):
    """
    Builds a JSON object with the endpoints in the format compatible with the Markups module in 3D Slicer.

    Parameters
    ----------
    endpoint_list : vtkPoints
        The list of endpoints.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------
    endpoints_json : dict
        The JSON-serializable dict with the endpoints.

    """
    # For some reason, in order to visualize the endpoints in Slicer correctly and in the right orientation, 
    # we need to invert the coordinate system for the first two axes. I believe this may have something to do
    # with how nibabel reads the orientation of the nifti file compared to sitk
    coordinate_system = ''.join(str(axis) for axis in nib.orientations.aff2axcodes(segmentation_affine))
    if "R" in coordinate_system:
        coordinate_system = coordinate_system.replace("R", "L")
    else:
        coordinate_system = coordinate_system.replace("L", "R")
    if "A" in coordinate_system:
        coordinate_system = coordinate_system.replace("A", "P")
    else:
        coordinate_system = coordinate_system.replace("P", "A")
    endpoints_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": coordinate_system,
                "coordinateUnits": "mm",
                "locked": "false",
                "fixedNumberOfControlPoints": "false",
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 0,
                "controlPoints": [],
                "measurements": [],
                "display": {
                    "visibility": "true",
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": "false",
                    "pointLabelsVisibility": "false",
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 2.5,
                    "useGlyphScale": "true",
                    "sliceProjection": "false",
                    "sliceProjectionUseFiducialColor": "true",
                    "sliceProjectionOutlinedBehindSlicePlane": "false",
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": "false",
                    "translationHandleVisibility": "true",
                    "rotationHandleVisibility": "true",
                    "scaleHandleVisibility": "true",
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }

    # Pass from vtkPoints to list
    endpoint_list_ = [list(endpoint_list.GetPoint(idx)) for idx in range(endpoint_list.GetNumberOfPoints())]

    for idx, endpoint in enumerate(endpoint_list_):
        endpoints_json["markups"][0]["controlPoints"].append(
            {
                "id": str(idx + 1),
                "label": "Endpoints-1",
                "description": "",
                "associatedNodeID": "",
                "position": list(endpoint),
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": "false",
                "locked": "false",
                "visibility": "true",
                "positionStatus": "defined"
            }
        )
        endpoints_json["markups"][0]["lastUsedControlPointNumber"] += 1

    return endpoints_json

def get_bounding_box_limits_3d(array):
    """
    Computes bounding box (only z axis) of a numpy array (expects an array 
    with zeros as background).

    Parameters
    ----------
    array : numpy.array or array-like object
        3D numpy binary (0, 1) array.

    Returns
    -------
    min_lr : integer
        Lower bound on axis x, LR (in voxel coordinates).
    max_lr : integer
        Upper bound on axis x, LR (in voxel coordinates).
    min_pa : integer
        Lower bound on axis y, PA (in voxel coordinates).
    max_pa : integer
        Upper bound on axis y, PA (in voxel coordinates).
    min_is : integer
        Lower bound on axis z, IS (in voxel coordinates).
    max_is : integer
        Upper bound on axis z, IS (in voxel coordinates).

    """
    axis_left_right = np.any(array, axis=(0, 1))
    axis_posterior_anterior = np.any(array, axis=(0, 2))
    axis_inferior_superior = np.any(array, axis=(1, 2))

    min_lr, max_lr = np.where(axis_left_right)[0][[0, -1]]
    min_pa, max_pa = np.where(axis_posterior_anterior)[0][[0, -1]]
    min_is, max_is = np.where(axis_inferior_superior)[0][[0, -1]]

    return min_lr, max_lr, min_pa, max_pa, min_is, max_is

def volume_sanity_check(segmentation_array, segmentation_affine):
    """
    Check that the volume of the segmentation is within the expected range.
    Otherwise, image will be read as an artifact and an error will be raised.

    Parameters
    ----------
    segmentation_array : numpy.array or array-like object
        3D numpy binary (0, 1) array.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------

    """
    # Compute volume of the bouding box taking into account voxel size
    min_lr, max_lr, min_pa, max_pa, min_is, max_is = get_bounding_box_limits_3d(segmentation_array)
    # Compute segmentation volume taking into account voxel size
    voxel_size = np.abs(np.prod([segmentation_affine[idx, idx] for idx in range(3)]))
    segmentation_volume = np.sum(segmentation_array > 0) * voxel_size
    bouding_box_volume = (max_lr - min_lr) * (max_pa - min_pa) * (max_is - min_is) * voxel_size
    if segmentation_volume < 3.5e4: # Empirically tested
        raise ValueError("Segmentation volume is too small: {:.2f} mm3".format(segmentation_volume))
    if bouding_box_volume < 2.5e6: # Empirically tested
        raise ValueError("Bounding box volume is too small: {:.2f} mm3".format(bouding_box_volume))
    if bouding_box_volume > 3.5e7: # Empirically tested
        raise ValueError("Bounding box volume is too large: {:.2f} mm3".format(bouding_box_volume))
    if segmentation_volume < 4e4 and bouding_box_volume < 4e6: # Empirically tested
        raise ValueError("Combination of segmentation volume and bounding box volume is too small: \nSegmentation volume: {:.2f} mm3 \nBounding box volume: {:.2f}".format(segmentation_volume, bouding_box_volume))

def robust_endpoint_relocation(endpoint_vtk_points, segmentation_array, segmentation_affine, window_size = 5, larger_window_for_aa_startpoint=False):
    """
    Relocates automatically detected endpoints to the center of mass of the closest component
    inside a local region around the endpoint (defined by n).

    Takes the endpoint position, converts it to voxel coordinates with the affine matrix, then defines a region  
    of (2 * n) ^ 3 voxels centered around the endpoint. Then components inside the local region are treated 
    as separate objects. The minimum distance from these objects to the endpoint is computed, and from 
    these, the object with the smallest distance to the endpoint is chosen to compute the centroid, which
    is converted back to RAS with the affine matrix.

    Parameters
    ----------
    endpoint : numpy.array or array-like object 
        Position of the endpoint in RAS coordinates.
    segmentation_array : numpy.array or array-like object
        Numpy array corresponding to the masked_volume_node.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.
    window_size : int 
        Defines the size of the region around the endpoint that is analyzed for this method.
        New endpoint location will be searched within a cubic box of size 2 * n around the 
        originial endpoint location.
    larger_window_for_aa_startpoint : bool, optional
        Whether to use a larger window for the first endpoint (AA startpoint). The default is False.

    Returns
    -------
    new_endpoint : numpy.array or array-like object
        New position of the endpoint.

    """
    # Invert the affine matrix
    segmentation_affine_inv = np.linalg.inv(segmentation_affine)
    max_iterations = 5
    for endpoint_idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        endpoint = endpoint_vtk_points.GetPoint(endpoint_idx)
        # Compute endpoint ijk coordinates with affine matrix
        i, j, k = np.round(np.matmul(segmentation_affine_inv, np.append(endpoint, 1.0))[:3]).astype(int)
        original_i, original_j, original_k = i, j, k
        valid_endpoint = False
        window_size_  = window_size
        iteration = 0
        while not valid_endpoint and iteration < max_iterations:
            # Define limits of the region of interest
            if larger_window_for_aa_startpoint and endpoint_idx == 0:
                i_min, i_max = np.clip([i - (window_size_ + 15), i + (window_size_ + 15)], 0, segmentation_array.shape[0])
                j_min, j_max = np.clip([j - (window_size_ + 15), j + (window_size_ + 15)], 0, segmentation_array.shape[1])
                k_min, k_max = np.clip([k - (window_size_ + 15), k + (window_size_ + 15)], 0, segmentation_array.shape[2])
            else:
                i_min, i_max = np.clip([i - window_size_, i + window_size_], 0, segmentation_array.shape[0])
                j_min, j_max = np.clip([j - window_size_, j + window_size_], 0, segmentation_array.shape[1])
                k_min, k_max = np.clip([k - window_size_, k + window_size_], 0, segmentation_array.shape[2])
            # Mask the segmentation_array (only region of interest)
            masked_segmentation = segmentation_array[i_min:i_max, j_min:j_max, k_min:k_max]
            # Divide into different connected components
            label_mask = measure.label(masked_segmentation, connectivity=1)
            unique_labels = np.unique(label_mask)[1:] 

            if unique_labels.size == 0:
                # Default to the original coordinates if no labels were found
                window_size_ += 5
                iteration += 1
            else:
                if unique_labels.size > 1:
                    # Only perform distance transformation when necessary
                    distances = np.ndarray([len(unique_labels), label_mask.shape[0], label_mask.shape[1], label_mask.shape[2]])
                    for idx_unique_label, unique_label in enumerate(unique_labels):
                        # Select label mask for the current unique label
                        label_mask_unique_label = label_mask == unique_label
                        # Invert the mask to compute the distance transform
                        inverted_label_mask = label_mask_unique_label == 0
                        distances[idx_unique_label, :] = ndimage.distance_transform_edt(inverted_label_mask, return_distances=True, return_indices=False)
                    # Now collect the distances at the center of the region of interest
                    distances = distances[:, window_size, window_size, window_size]
                    # Select the nearest label (that with the lowest value in the distance map)
                    nearest_label = unique_labels[np.argmin(distances)]
                    properties = measure.regionprops((label_mask == nearest_label).astype(int))
                    centroid = properties[0].centroid + np.array([i_min, j_min, k_min])
                elif unique_labels.size == 1:
                    # If only one label, use its centroid directly
                    properties = measure.regionprops(label_mask.astype(int), label_mask == unique_labels[0])
                    centroid = properties[0].centroid + np.array([i_min, j_min, k_min])
                # Check if the new endpoint is within the segmentation (array at ijk is non-zero)
                if segmentation_array[int(centroid[0]), int(centroid[1]), int(centroid[2])] > 0.5:
                    valid_endpoint = True
                else:
                    # Recenter centroid for the next iteration 
                    i, j, k = centroid.astype(int)
                    iteration += 1
            # If no valid endpoint was found after max_iterations, default to the originally found centroid
            if not valid_endpoint and iteration == max_iterations:
                centroid = np.array([original_i, original_j, original_k])
            
        # Return the new position of the endpoint in RAS coordinates
        endpoint_vtk_points.SetPoint(endpoint_idx, np.matmul(segmentation_affine, np.append(centroid, 1.0))[:3])

    print("Endpoint relocation completed. Number of endpoints: ", endpoint_vtk_points.GetNumberOfPoints())

    return endpoint_vtk_points

def aortic_arch_endpoint_check(endpoint_vtk_points, segmentation_array, segmentation_affine):
    """
    Checks that both ens of the aortic arch (AA), if present, have one associated endpoint.
    To do that, it looks at the bottom slice of the volume and analyzes the presence 
    of large connected components. Once it has recognized all large connected components,
    it checks if any endpoint is at an Euclidean distance of less than 50 mm with respect to the
    center of mass of the bottom islands. 
    
    Paremeters
    ----------
    endpoints_node : vtkMRMLMarkupsFiducialNode
        MRML node with all endpoints from the automatic endpoint detection.
    segmentation_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 85% of the segmentation's bounding box.
    segmentation_affine : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------
    endpoints_node : vtkMRMLMarkupsFiducialNode
        Updated MRML node with all endpoints from the automatic endpoint detection.

    """
    # We define a distance factor in case we are dealing with images of a different resolution
    # We always assume we have close-to-isotropic voxels. 0.43 corresponds to the reference voxel size
    # used to empirically define certain reference values
    factor = abs(0.43 / segmentation_affine[0, 0])

    # For AA island validation (number of foreground voxels in the bottom slice)
    # Empirically, we found that 500 is a good threshold for a voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    voxel_size = np.prod([segmentation_affine[idx, idx] for idx in range(3)])
    threshold_counts = abs(round(500 * (reference_voxel_size / voxel_size)))

    # For AA endpoints check (distance from bottom slice in mm)
    threshold_distance_for_aa_centroid = 30 # mm

    # Divide into different connected components of the bottom slice
    label_mask = measure.label(segmentation_array[:, :, 0])
    properties = measure.regionprops(label_mask.astype(int), label_mask.astype(int))
    
    # Get rid of all components below the threshold_counts
    # This is done because we expect here to only have bottom slices of the
    # ascending and descending aorta. This way we get rid of any other component
    _, counts = np.unique(label_mask, return_counts=True)
    delete_idx = []
    for idx, count in enumerate(counts):
        if count < threshold_counts:
            delete_idx.append(idx - 1)
    properties = list(np.delete(properties, delete_idx))
    
    # Access and store the coordinates of centroids in RAS coordinates
    # Notice that we set the S coordinate to 1.0 for all centroids
    aa_centroids_to_be_found = np.zeros(shape = (len(properties), 3))
    aa_candidate_dict = {}
    for idx_centroid, prop in enumerate(properties):
        aa_centroids_to_be_found[idx_centroid] = np.matmul(segmentation_affine, np.append(np.array(prop.centroid), [1.0, 1.0]))[:3] # Result in RAS coordinates
        aa_candidate_dict[idx_centroid] = {}
        aa_candidate_dict[idx_centroid]["found_aa_centroid_candidate"] = False
        aa_candidate_dict[idx_centroid]["endpoints_idx"] = []
        aa_candidate_dict[idx_centroid]["endpoints"] = []
        aa_candidate_dict[idx_centroid]["distances"] = []

    # Compute distance from each endpoint to all centroids of components in the bottom slice
    # The goal is to check that each component (generallly there should be 2) has one endpoint
    # nearby
    for endpoint_idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        endpoint = endpoint_vtk_points.GetPoint(endpoint_idx)
        for idx_centroid, centroid in enumerate(aa_centroids_to_be_found):
            # If a connnected component is found close to an endpoint, we accept it as correctly placed
            # We remove the AA centroid from the list of aa_centroids as a way of saying "this one is found" 
            distance_to_aa_centroid = np.linalg.norm(centroid - endpoint)
            if distance_to_aa_centroid < threshold_distance_for_aa_centroid:
                aa_candidate_dict[idx_centroid]["found_aa_centroid_candidate"] = True
                aa_candidate_dict[idx_centroid]["endpoints_idx"].append(endpoint_idx)
                aa_candidate_dict[idx_centroid]["endpoints"].append(endpoint)
                aa_candidate_dict[idx_centroid]["distances"].append(distance_to_aa_centroid)

    # We add a filter to ensure that we only keep the closest endpoint to each centroid, and discard the rest
    # for this, we remove the furthest one from the corresponding aa_candidate
    aa_centroids_to_be_found = np.delete(aa_centroids_to_be_found, [idx_centroid for idx_centroid in aa_candidate_dict.keys() if aa_candidate_dict[idx_centroid]["found_aa_centroid_candidate"]], axis=0)
    endpoints_to_remove = []
    for idx_centroid in aa_candidate_dict.keys():
        # print(aa_candidate_dict[idx_centroid])
        if aa_candidate_dict[idx_centroid]["found_aa_centroid_candidate"]:
            if len(aa_candidate_dict[idx_centroid]["endpoints_idx"]) > 1:
                # We keep te closest endpoint to the candidate, we remove the rest
                closest_endpoint_idx = aa_candidate_dict[idx_centroid]["endpoints_idx"][np.argmin(aa_candidate_dict[idx_centroid]["distances"])]
                endpoints_to_remove += [idx for idx in aa_candidate_dict[idx_centroid]["endpoints_idx"] if idx != closest_endpoint_idx]

    # Remove endpoints that are not the closest to the centroids. The only way to remove points from a vtkPoints object
    # is to create a new one and copy the points that we want to keep
    new_endpoints = vtk.vtkPoints()
    for idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        if idx in endpoints_to_remove: continue
        new_endpoints.InsertNextPoint(endpoint_vtk_points.GetPoint(idx))
    endpoint_vtk_points = new_endpoints

    # If any aa_centroids_to_be_found survive, it means that no enpoints were found close by
    if len(aa_centroids_to_be_found) > 0:
        print("{} AA islands do not have associated endpoints".format(len(aa_centroids_to_be_found)))
        # This way, we convert the remaining centroids to endpoints
        for centroid in aa_centroids_to_be_found:
            print("Adding endpoint at", centroid)
            endpoint_vtk_points.InsertNextPoint(centroid)

    # Now all that's left is to ensure that the startpoint is placed at the descending aorta
    # (most proximal point from femoral access in endovascular interventions)

    # Select distal AA endpoint as startpoint (in some cases, the distal LSA endpoint is closer to the origin)
    # The criteria will be to choose the AA endpoint (at < 50 mm from bottom slice) that is closest to the reference point
    # Check every other point's distance to origin (ijk)
    # Reference point set at [350, 0, 0] in LAS coordinates
    if nib.orientations.aff2axcodes(segmentation_affine) == ("R", "A", "S"):
        aa_reference_voxel_coordinates = np.array([150.0 * factor, 0.0, 0.0])
    elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "A", "S"):
        aa_reference_voxel_coordinates = np.array([350.0 * factor, 0.0, 0.0])
    elif nib.orientations.aff2axcodes(segmentation_affine) == ("L", "P", "S"):
        aa_reference_voxel_coordinates = np.array([350.0 * factor, label_mask.shape[1], 0.0])
    aa_reference_ras_coordinates = np.dot(segmentation_affine, np.append(aa_reference_voxel_coordinates, 1))[:3]

    # We store the distance to the reference point for each endpoint (in mm)
    distance_to_reference = []
    for endpoint_idx in range(endpoint_vtk_points.GetNumberOfPoints()):
        endpoint = endpoint_vtk_points.GetPoint(endpoint_idx)
        distance_to_reference.append(np.linalg.norm(endpoint - aa_reference_ras_coordinates))

    # Get order from closest to furthest
    sorted_distance_idx = np.argsort(distance_to_reference)
    startpoint_candidate = endpoint_vtk_points.GetPoint(sorted_distance_idx[0])
    # Check if the endpoint at 0 is at the distal AA
    if sorted_distance_idx[0] == 0:
        print("Original startpoint is at distal AA")
    # If it is not, set next closest endpoint to reference as startpoint if it is closer to bottom slice
    else:
        print("New startpoint ({}): {}".format(sorted_distance_idx[0], startpoint_candidate))
        endpoint_vtk_points.SetPoint(sorted_distance_idx[0], endpoint_vtk_points.GetPoint(0))
        endpoint_vtk_points.SetPoint(0, startpoint_candidate)
    
    return endpoint_vtk_points

def consolidate_points(centerline_model, threshold=1e-2):
    """
    Maps all points that are within a threshold distance of each other to a single reference
    point, so that centerlines that overlap actually overlap (i.e. share the same points).

    Parameters
    ----------
    centerline_model : vtk.vtkPolyData
        Centerline model.
    threshold : float, optional
        Threshold distance for grouping points. The default is 1e-2.

    Returns
    -------
    index_map : dict
        Dictionary that maps original point indices to the representative point index.

    """
    points = np.array([centerline_model.GetPoint(idx) for idx in range(centerline_model.GetNumberOfPoints())])
    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, r=threshold)

    # groups is a nested list containing groups of centerline point ids that are within threshold distance of each other
    # Each entry idx (a list of point ids) represents all the point ids that are grouped with that point

    # Map original indices to new consolidated indices
    index_map = {}
    for idx, group in enumerate(groups):
        if not group:
            print(f"Empty group ({idx})")
        representative_index = group[0]  # Take the first point in group as representative. It will be the smallest index of the group, because they are always sorted
        for index in group:
            index_map[index] = representative_index # index_map is a dictionary that related point_ids with the representative point_id

    return index_map

def update_new_centerline_model(centerline_model, index_map):
    """
    Updates the polydata structure by consolidating points that are within a 
    threshold distance of each other. Basically assigns the same position and point data values
    exactly to all points within the threshold distance, keeping the original number of points 
    and cell connectivity.

    Parameters
    ----------
    centerline_model : vtk.vtkPolyData
        Centerline model.
    index_map : dict
        Dictionary that maps original point indices to the representative point index.

    Returns
    -------
    new_centerline_model : vtk.vtkPolyData
        Updated centerline model with consolidated points.
        
    """
    # Get original points and point data arrays
    original_points = centerline_model.GetPoints()
    num_point_arrays = centerline_model.GetPointData().GetNumberOfArrays()
    
    # Create new points and point data structures
    new_points = vtk.vtkPoints()
    new_point_arrays = [vtk.vtkDoubleArray() for _ in range(num_point_arrays)]
    
    # Copy the attributes and names of the original point data arrays
    for i in range(num_point_arrays):
        array = centerline_model.GetPointData().GetArray(i)
        new_point_arrays[i].SetName(array.GetName())
        new_point_arrays[i].SetNumberOfComponents(array.GetNumberOfComponents())

    # Mapping of old indices to new indices after consolidation
    new_index_map = {}

    # Ensure that each representative index has a new index
    for representative_index in index_map.values():
        if representative_index not in new_index_map:
            # Add point to new_points, and save the new index
            new_point_idx = new_points.InsertNextPoint(original_points.GetPoint(representative_index))
            new_index_map[representative_index] = new_point_idx
            # Copy data for this point
            for i in range(num_point_arrays):
                original_array = centerline_model.GetPointData().GetArray(i)
                value = [original_array.GetComponent(representative_index, j) for j in range(original_array.GetNumberOfComponents())]
                new_point_arrays[i].InsertNextTuple(value)

    new_centerline_model = vtk.vtkPolyData()
    new_centerline_model.SetPoints(new_points)
    for new_array in new_point_arrays:
        new_centerline_model.GetPointData().AddArray(new_array)

    # Remap the cells
    new_cells = vtk.vtkCellArray()
    for i in range(centerline_model.GetNumberOfCells()):
        cell = centerline_model.GetCell(i)
        new_cell_points = vtk.vtkIdList()
        for j in range(cell.GetNumberOfPoints()):
            original_index = cell.GetPointId(j)
            representative_index = index_map[original_index]
            new_index = new_index_map[representative_index]
            new_cell_points.InsertNextId(new_index)
        new_cells.InsertNextCell(new_cell_points)
    
    new_centerline_model.SetLines(new_cells)

    return new_centerline_model

def clean_centerline(centerline_model, startpoint, threshold=1e-3):
    """
    Applies the consolidate_points and update_polydata functions to clean the centerline model.
    The result is a centerline model with consolidated points, i.e., centerlines that overlap with
    points that share the exact position and data values. This helps simplify postprocessing steps.

    Also, we remove cells with less than 3 points and cells that do not start at the startpoint.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Centerline model.
    threshold : float, optional
        Threshold distance for grouping points. The default is 1e-3.

    Returns
    -------
    new_polydata : vtk.vtkPolyData
        Updated centerline model with consolidated points.
        
    """
    index_map = consolidate_points(centerline_model, threshold)

    print(f"Found {centerline_model.GetNumberOfCells()} centerline cells")

    # Remove cells with less than 3 points
    number_of_removed_cells = 0
    for idx in range(centerline_model.GetNumberOfCells()):
        cell = centerline_model.GetCell(idx)
        if np.linalg.norm(np.array(cell.GetPoints().GetPoint(0)) - startpoint) > 30:
            centerline_model.DeleteCell(idx)
            number_of_removed_cells += 1
        elif cell.GetNumberOfPoints() <= 2:
            centerline_model.DeleteCell(idx)
            number_of_removed_cells += 1

    centerline_model.RemoveDeletedCells()
    print(f"Removed {number_of_removed_cells} cells (less than 3 points or not starting at startpoint)")

    return update_new_centerline_model(centerline_model, index_map)

# def compute_frenet_serret(centerline_poly_data):
#     """
#     Uses the vtkParallelTransportFrame custom filter from
#     Slicer to compute tangent, normal and binormal vectors from 
#     the Frenet-Serret frame for each point of the centerline model.

#     Parameters
#     ----------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model.
    
#     Returns
#     -------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model with tangent, normal and binormal vectors 
#         computed for each centerline point as point data.
    
#     """
#     curve_coordinate_system_generator = slicer.vtkParallelTransportFrame()
#     curve_coordinate_system_generator.SetInputData(centerline_poly_data)
#     curve_coordinate_system_generator.Update()

#     return curve_coordinate_system_generator.GetOutput()

# def compute_curvature_and_torsion(centerline_poly_data):
#     """
#     Compute cruvature and torsion using the Frenet-Serret moving frame.
#     Assumes that Frenet-Serret vectors are available as point data in the 
#     input centerline_poly_data object. Stores curvature and torsion as
#     point data arrays in the centerline_poly_data object. Also computes 
#     a smoothed out version of the curvature using a Savitzky-Golay filter.

#     Parameters
#     ----------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model.

#     Returns
#     ------
#     centerline_poly_data : vtk.vtkPolyData
#         Centerline model with curvature, torsion and filtered_curvature 
#         as additional point data arrays.

#     References:
#     [1]     
    
#     """
#     # Load Frenet-Serret vectors as numpy arrays for each point
#     coordinates = np.ndarray([centerline_poly_data.GetNumberOfPoints(), 3])
#     for point_idx in range(centerline_poly_data.GetNumberOfPoints()):
#         coordinates[point_idx] = centerline_poly_data.GetPoints().GetPoint(point_idx)
#     tangents = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Tangents"))
#     normals = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Normals"))
#     binormals = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Binormals"))
    
#     # Compute curvature and torsion
#     curvature = np.linalg.norm(np.gradient(tangents, axis = 0), axis = 1)
#     torsion = (- np.gradient(binormals, axis = 0) * normals).sum(axis = 1)
#     # Compute smoothed curvature
#     filtered_curvature = savgol_filter(curvature, window_length=10, polyorder=3, mode='nearest')
    
#     # Add new point data arrays
#     centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(curvature))
#     centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Curvature")
#     centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(torsion))
#     centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Torsion")
#     centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(filtered_curvature))
#     centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Filtered curvature")

#     return centerline_poly_data