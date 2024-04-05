#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
 
import os
import slicer
import vtk

import numpy as np
import nibabel as nib

from skimage import measure
from scipy import ndimage

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.signal import savgol_filter

def get_bounding_box_limits_3d(img):
    """
    Computes bounding box (only z axis) of a numpy array (expects an array 
    with zeros as background).

    Parameters
    ----------
    img : numpy.array or array-like object
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
    axis_left_right = np.any(img, axis=(0, 1))
    axis_posterior_anterior = np.any(img, axis=(0, 2))
    axis_inferior_superior = np.any(img, axis=(1, 2))

    min_lr, max_lr = np.where(axis_left_right)[0][[0, -1]]
    min_pa, max_pa = np.where(axis_posterior_anterior)[0][[0, -1]]
    min_is, max_is = np.where(axis_inferior_superior)[0][[0, -1]]

    return min_lr, max_lr, min_pa, max_pa, min_is, max_is

def volume_sanity_check(segmentation_nifti):
    """
    Check that the volume of the segmentation is within the expected range.
    Otherwise, image will be read as an artifact and an error will be raised.

    Parameters
    ----------
    case_dir : string or path-like object   
        Path to case directory.

    """
    data = segmentation_nifti.get_fdata()
    # Compute volume of the bouding box taking into account voxel size
    min_lr, max_lr, min_pa, max_pa, min_is, max_is = get_bounding_box_limits_3d(data)
    # Compute segmentation volume taking into account voxel size
    segmentation_volume = np.sum(data > 0) * np.prod(segmentation_nifti.header.get_zooms())
    bouding_box_volume = (max_lr - min_lr) * (max_pa - min_pa) * (max_is - min_is) * np.prod(segmentation_nifti.header.get_zooms())
    if segmentation_volume < 4.5e4: # Empirically tested
        raise ValueError("Segmentation volume is too small: {:.2f} mm3".format(segmentation_volume))
    if bouding_box_volume < 3.5e6: # Empirically tested
        raise ValueError("Bounding box volume is too small: {:.2f} mm3".format(bouding_box_volume))
    if bouding_box_volume > 3.5e7: # Empirically tested
        raise ValueError("Bounding box volume is too large: {:.2f} mm3".format(bouding_box_volume))
    if segmentation_volume < 5e4 and bouding_box_volume < 6e6: # Empirically tested
        raise ValueError("Combination of segmentation volume and bounding box volume is too small: \nSegmentation volume: {:.2f} mm3 \nBounding box volume: {:.2f}".format(segmentation_volume, bouding_box_volume))

def aortic_arch_endpoint_check(endpoints_node, masked_volume_array, aff):
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
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 85% of the segmentation's bounding box.
    aff : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------
    endpoints_node : vtkMRMLMarkupsFiducialNode
        Updated MRML node with all endpoints from the automatic endpoint detection.

    """
    # We define a distance factor in case we are dealing with images of a different resolution
    # We always assume we have close-to-isotropic voxels. 0.43 corresponds to the reference voxel size
    # used to empirically define certain reference values
    factor = abs(0.43 / aff[0, 0])
    # For AA island validation (number of foreground voxels in the bottom slice)
    # Empirically, we found that 500 is a good threshold for a voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    voxel_size = np.prod([aff[idx, idx] for idx in range(3)])
    threshold_counts = abs(round(500 * (reference_voxel_size / voxel_size)))
    # For AA endpoints check (distance from bottom slice)
    threshold_distance = 50 * 0.4 / aff[2, 2]

    # Divide into different connected components of the bottom slice
    label_mask = measure.label(masked_volume_array[0])
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
    centroids = np.zeros(shape = (len(properties), 3))
    for idx, prop in enumerate(properties):
        centroids[idx] = np.matmul(aff, np.append(np.array(prop.centroid)[[1, 0]], [1.0, 1.0]))[:3]

    # Compute distance from each endpoint to all centroids of components in the bottom slice
    # The goal is to check that each component (generallly there should be 2) has one endpoint
    # nearby
    for idx in range(endpoints_node.GetNumberOfControlPoints()):
        endpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(idx))
        delete_idx = None
        for idx_centroids, centroid in enumerate(centroids):
            # If a connnected component is found close to an endpoint, we accept it as correctly placed
            if np.linalg.norm(centroid - endpoint) < threshold_distance: # Threshold at 50 mm
                delete_idx = idx_centroids
        if delete_idx is not None:
            centroids = np.delete(centroids, delete_idx, axis=0)

    # If any connected components survive, it means that no enpoints were found close by
    if len(centroids) > 0:
        print("{} AA islands do not have associated endpoints".format(len(centroids)))
        # This way, we convert the remaining centroinds to endpoints
        for centroid in centroids:
            print("Adding endpoint at", centroid)
            print()
            endpoints_node.AddControlPoint(np.array(centroid))

    # Now all that's left is to ensure that the startpoint is placed at the descending aorta
    # (most proximal point from femoral access in endovascular interventions)
            
    # Select distal AA endpoint as startpoint (in some cases, the distal LSA endpoint is closer to the origin)
    # The criteria will be to choose the AA endpoint (at < 50 mm from bottom slice) that is closest to the reference point
    # Check every other point's distance to origin (ijk)
    distance_to_reference = []
    for idx in range(endpoints_node.GetNumberOfControlPoints()):
        endpoint = np.matmul(np.linalg.inv(aff), np.append(np.array(endpoints_node.GetCurvePoints().GetPoint(idx)), 1.0))[:3]
        # Reference point set at [350, 0, 0] in LAS coordinates
        if nib.orientations.aff2axcodes(aff) == ("R", "A", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([150.0 * factor, 0.0, 0.0])))
        elif nib.orientations.aff2axcodes(aff) == ("L", "A", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([350.0 * factor, 0.0, 0.0])))
        elif nib.orientations.aff2axcodes(aff) == ("L", "P", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([350.0 * factor, label_mask.shape[1], 0.0])))

    # Get order from closest to furthest
    sorted_distance_idx = np.argsort(distance_to_reference)
    for idx in sorted_distance_idx:
        startpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(idx))
        if np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] > threshold_distance:
            print("Startpoint {} found is not in the AA region".format(idx))
            pass
        else:
            # Make sure that startpoint is close to the bottom slice
            if np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] < threshold_distance and idx == 0:
                print("Original startpoint is at distal AA")
                break
            # If it is not, set next closest endpoint to reference as startpoint if it is closer to bottom slice
            elif np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] < threshold_distance and idx != 0:
                print("New startpoint ({}): {}".format(idx, startpoint))
                endpoints_node.SetNthControlPointPosition(idx, endpoints_node.GetCurvePoints().GetPoint(0))
                endpoints_node.SetNthControlPointPosition(0, startpoint)
                break
            else: 
                pass
    
    return endpoints_node

def ica_endpoint_check(endpoints_node, masked_volume_array, aff):
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
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 85% of the segmentation's bounding box.
    aff : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.

    Returns
    -------
    endpoints_node : vtkMRMLMarkupsFiducialNode
        Updated MRML node with all endpoints from the automatic endpoint detection.

    """
    # We define a distance factor in case we are dealing with images of a different resolution
    # We always assume we have close-to-isotropic voxels. 0.43 corresponds to the reference voxel size
    # used to empirically define certain reference values
    factor = abs(0.43 / aff[0, 0])
    # For AA island validation (number of foreground voxels in the bottom slice)
    # Empirically, we found that 500 is a good threshold for a voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    voxel_size = np.prod([aff[idx, idx] for idx in range(3)])
    threshold_counts = abs(round(500 * (reference_voxel_size / voxel_size)))
    # For AA endpoints check (distance from bottom slice)
    threshold_distance = 50 * 0.4 / aff[2, 2]

    # Divide into different connected components of the bottom slice
    label_mask = measure.label(masked_volume_array[0])
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
    centroids = np.zeros(shape = (len(properties), 3))
    for idx, prop in enumerate(properties):
        centroids[idx] = np.matmul(aff, np.append(np.array(prop.centroid)[[1, 0]], [1.0, 1.0]))[:3]

    # Compute distance from each endpoint to all centroids of components in the bottom slice
    # The goal is to check that each component (generallly there should be 2) has one endpoint
    # nearby
    for idx in range(endpoints_node.GetNumberOfControlPoints()):
        endpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(idx))
        delete_idx = None
        for idx_centroids, centroid in enumerate(centroids):
            # If a connnected component is found close to an endpoint, we accept it as correctly placed
            if np.linalg.norm(centroid - endpoint) < threshold_distance: # Threshold at 50 mm
                delete_idx = idx_centroids
        if delete_idx is not None:
            centroids = np.delete(centroids, delete_idx, axis=0)

    # If any connected components survive, it means that no enpoints were found close by
    if len(centroids) > 0:
        print("{} ICA islands do not have associated endpoints".format(len(centroids)))
        # This way, we convert the remaining centroinds to endpoints
        for centroid in centroids:
            print("Adding endpoint at", centroid)
            print()
            endpoints_node.AddControlPoint(np.array(centroid))

    # Now all that's left is to ensure that the startpoint is placed at the descending aorta
    # (most proximal point from femoral access in endovascular interventions)
            
    # Select distal AA endpoint as startpoint (in some cases, the distal LSA endpoint is closer to the origin)
    # The criteria will be to choose the AA endpoint (at < 50 mm from bottom slice) that is closest to the reference point
    # Check every other point's distance to origin (ijk)
    distance_to_reference = []
    for idx in range(endpoints_node.GetNumberOfControlPoints()):
        endpoint = np.matmul(np.linalg.inv(aff), np.append(np.array(endpoints_node.GetCurvePoints().GetPoint(idx)), 1.0))[:3]
        # Reference point set at [350, 0, 0] in LAS coordinates
        if nib.orientations.aff2axcodes(aff) == ("L", "A", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([350.0 * factor, label_mask.shape[1], 0.0])))
        elif nib.orientations.aff2axcodes(aff) == ("L", "P", "S"):
            distance_to_reference.append(np.linalg.norm(endpoint - np.array([350.0 * factor, 0.0, 0.0])))
    print(distance_to_reference)
    # Get order from closest to furthest
    sorted_distance_idx = np.argsort(distance_to_reference)
    for idx in sorted_distance_idx:
        startpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(idx))
        if np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] > threshold_distance:
            print("Startpoint {} found is not in the ICA region".format(idx))
            pass
        else:
            # Make sure that startpoint is close to the bottom slice
            if np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] < threshold_distance and idx == 0:
                print("Original startpoint is at right ICA")
                break
            # If it is not, set next closest endpoint to reference as startpoint if it is closer to bottom slice
            elif np.matmul(np.linalg.inv(aff), np.append(startpoint, 1.0))[2] < threshold_distance and idx != 0:
                print("New startpoint ({}): {}".format(idx, startpoint))
                endpoints_node.SetNthControlPointPosition(idx, endpoints_node.GetCurvePoints().GetPoint(0))
                endpoints_node.SetNthControlPointPosition(0, startpoint)
                break
            else: 
                pass
    
    return endpoints_node

def robust_end_point_detection(endpoint, segmentation, aff, n = 15):
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
    segmentation : numpy.array or array-like object
        Numpy array corresponding to the masked_volume_node.
    aff : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.
    n : integer 
        Defines the size of the region around the endpoint that is analyzed for this method.
        New endpoint location will be searched within a cubic box of size 2 * n around the 
        originial endpoint location.

    Returns
    -------
    new_endpoint : numpy.array or array-like object
        New position of the endpoint.

    """
    # Compute endpointRAS coordinates with affine matrix
    R, A, S = np.round(np.matmul(np.linalg.inv(aff), np.append(endpoint, 1.0))[:3]).astype(int)
    
    # Mask the segmentation (only region of interest)
    masked_segmentation = segmentation[np.max([0, S - n]): np.min([segmentation.shape[0], S + n]), 
                                       np.max([0, A - n]): np.min([segmentation.shape[1], A + n]),
                                       np.max([0, R - n]): np.min([segmentation.shape[2], R + n])]
    
    # Divide into different connected components
    label_mask = measure.label(masked_segmentation)
    # We sort label values and ignore the background
    labels = np.sort(np.unique(label_mask))
    labels = np.delete(labels, np.where([labels == 0]))
    # Pass masked groups to one-hot encoding
    label_mask_one_hot = np.zeros([len(labels), label_mask.shape[0], label_mask.shape[1], label_mask.shape[2]], dtype=np.uint8)
    for idx, label in enumerate(labels):
        label_mask_one_hot[idx][label_mask == label] = 1

    # Invert the masks
    inverted_label_mask_one_hot = np.ones_like(label_mask_one_hot) - label_mask_one_hot
    
    # Get distance transform for each and get only closest component of the inverted masks
    # Distance transforms encode the distance of all foreground voxels
    # to the closest background element
    distance_labels = np.empty_like(labels, dtype=float)
    for idx in range(len(labels)):
        distance_labels[idx] = ndimage.distance_transform_edt(inverted_label_mask_one_hot[idx])[inverted_label_mask_one_hot.shape[1] // 2][inverted_label_mask_one_hot.shape[2] // 2][inverted_label_mask_one_hot.shape[3] // 2]
    # We keep only the closest component to the original endpoint
    mask = np.zeros_like(segmentation)
    mask[np.max([0, S - n]): np.min([segmentation.shape[0], S + n]), 
         np.max([0, A - n]): np.min([segmentation.shape[1], A + n]),
         np.max([0, R - n]): np.min([segmentation.shape[2], R + n])] = label_mask_one_hot[np.argmin(distance_labels)]
    
    # Get the centroid of the foregroud region and turn it into the new endpoint
    properties = measure.regionprops(mask.astype(int), mask.astype(int))
    center_of_mass = np.array(properties[0].centroid)[[2, 1, 0]]
    
    # Return the new position of the endpoint in RAS coordinates
    return np.matmul(aff, np.append(center_of_mass, 1.0))[:3]

def inspect_circular_centerlines(centerline_poly_data, surface_model, segmentation_node, segment_id, aff):
    """ 
    Analyzes surface model and centerline model to recognize large surface areas without associated centerline.
    This can happen due to either the presence of circular segments (VMTK does not contemplate this possibility) or due
    to suboptimal segmentation at some point of the volume, causing a vessel to be too thin for the centerline to 
    be correctly extracted.

    If a centerline-less segment is found, it performs a special centerline extraction based on endpoint detection
    over the segmnent of interest and the previously extracted centerline model.

    Parameters
    ---------- 
    centerline_poly_data : vtk.vtkPolyData
        Centerline model in vtkPolyData form for segment_id segment.
    surface_model : vtkMRMLModelNode
        Surface model as vtkMRMLModelNode.
    segmentation_node : vtkMRMLSegmentationNode
        MRML segmentation node.
    segment_id : integer
        Segment identifier in the segmentation_node.
    masked_volume_array : numpy.array
        Binary array of the segmentation mask after removal of the foreground voxels
        of the upper 80% of the segmentation's bounding box.
    aff : numpy.array
        Affine matrix from RAS to ijk tranformation.

    Returns
    -------
    centerline_poly_data : vtk.vtkPolyData
        New centerline model in vtkPolyData form for segment_id segment.
    
    """
    print("Inspecting circular segments in centerline...")
    # Check for potential circular centerlines
    decimation_factor = 10
    # Minimum fraction of surface model points at a distance larger than 2 * radius to consider presence of centerlineless segment
    threshold_radius_ratio = 0.05
    # Distance in ijk units to propagate a single centerlineless segment
    threshold_distance_prop = 5
    # Minimum number of surface model points to consider unique propagated segment as centerlineless
    # voxel size of 0.43 * 0.43 * 0.4 mm^3
    reference_voxel_size = 0.07385254 # = 0.43 * 0.43 * 0.4
    # Get voxel size from image
    voxel_size = abs(aff[0, 0] * aff[1, 1] * aff[2, 2])
    # Compute approximate number of voxels
    threshold_counts = round(500 * (reference_voxel_size / voxel_size))
    # threshold_counts = 500
    
    # First we pool all centerline points and their associated radius
    centerline_positions_array = np.ndarray([centerline_poly_data.GetNumberOfPoints(), 3])
    radius_array = vtk.util.numpy_support.vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Radius"))

    for idx in range(centerline_poly_data.GetNumberOfPoints()):
        centerline_positions_array[idx] = np.array(centerline_poly_data.GetPoint(idx))

    # Then, we pool all surface model points (we take the average position of each cell)
    surface_model_positions_array = np.ndarray([surface_model.GetNumberOfCells(), 3])

    for cell_idx in range(surface_model.GetNumberOfCells()):
        average_position_aux = np.ndarray([3, 3])
        for idx in range(3):
            average_position_aux[idx] = np.array([surface_model.GetCell(cell_idx).GetPoints().GetPoint(idx)])
        surface_model_positions_array[cell_idx] = np.mean(average_position_aux, axis = 0)

    # Now we compute the distance of each surface model point to the closest centerline point. We also store the associated radius
    surface_model_feature_array = np.ndarray([surface_model.GetNumberOfCells(), 2])
    # We decimate the surface model points for computation speed. We can do it aggressively and results are not significantly affected
    decimated_surface_model_feature_array = surface_model_feature_array[::decimation_factor, :]
    decimated_surface_model_positions_array = surface_model_positions_array[::decimation_factor, :]

    for idx, point in enumerate(decimated_surface_model_positions_array):
        decimated_surface_model_feature_array[idx][0] = np.linalg.norm(centerline_positions_array[np.argmin(np.linalg.norm(centerline_positions_array - point, axis = 1))] - point, axis = 0)
        decimated_surface_model_feature_array[idx][1] = radius_array[np.argmin(np.linalg.norm(centerline_positions_array - point, axis = 1))]

    # In order to identify if there is a potential significant segment without associated centerline, we can check the percentage of surface model cells at a large distance to any centerline point
    distance_radius_ratio = decimated_surface_model_feature_array[:, 0] / decimated_surface_model_feature_array[:, 1]
    
    print("% of surface model points at > 2 * radius to any centerline: {:.2f}".format(100 * len(distance_radius_ratio[distance_radius_ratio > 2]) / len(distance_radius_ratio)))
    
    if len(distance_radius_ratio[distance_radius_ratio > 2]) / len(distance_radius_ratio) < threshold_radius_ratio:
        print("No significant segments without centerline were found")
        print()
        return centerline_poly_data
        
    else:
        print("A significant segment without centerline was identified")
        print()

        # Label all surface model cells with no associated centerline (at a distance > 2 * radius)
        labels = np.zeros([len(decimated_surface_model_feature_array)])
        for idx in range(len(decimated_surface_model_feature_array)):
            if decimated_surface_model_feature_array[idx, 0] / decimated_surface_model_feature_array[idx, 1] > 2:
                labels[idx] = 1

        # We have to filter out potential islands that may have been incorrectly formed by clustering
        # Search for number of connected components with labels != 0
        # Establish a lower threshold for the number of connected triangles with label != 0
        decimated_surface_model_positions_array_ones = decimated_surface_model_positions_array[labels == 1]

        # We have designed a region growing algorithm to cluster the multiple islands that may have been formed, taking the Euclidean distance as the base metric
        # The algorithm does as follows:
        #     1) Selects an initial point (we take that with the lowest z coordinate)
        #     2) Assigns a label to that point
        #     3) From there, it computes the distance from this to all other points and attributes the same label to those at a distance smaller than 10 mm
        #     4) Checks if the next closest point is at a distance smaller than 10 mm to any point of the current cluster
        #     5) If so, that point passes as the initial seed and the process is repeated until we find that the closest point to the cluster is at a distance larger than 10 mm
        #     6) When this condition fails, the closes point to the cluster is taken as the initial seed for the next cluster, and the clustering label gets updated
        #     7) The algorithm stops when all points have been assigned to a clustering label
        label_idx = 1
        label_to_array = np.zeros([len(decimated_surface_model_positions_array_ones)])
        initial_point = decimated_surface_model_positions_array_ones[np.argmin(decimated_surface_model_positions_array_ones[:, 2])]
        label_to_array[np.argmin(decimated_surface_model_positions_array_ones[:, 2])] = label_idx

        # The algorithm will continue until all points belong to any cluster
        while 0 in np.unique(label_to_array):
            distances_array = []
            # Checks distance to all unassigned points
            for idx, point in enumerate(decimated_surface_model_positions_array_ones):
                if label_to_array[idx] == 0:
                    distance = np.linalg.norm(initial_point - point)
                    # If distance is smaller than 10 mm, assign to same cluster
                    if distance < threshold_distance_prop:
                        label_to_array[idx] = label_idx
                    # Otherwise, keep distance to current cluster seed
                    else:
                        distances_array.append(distance)

            if len(distances_array) > 0:
                # Convert closest point to new seed
                initial_point = decimated_surface_model_positions_array_ones[label_to_array == 0][np.argmin(distances_array)]
                # If distance from new seed to prior cluster is less than 10 mm, keep same clustering label
                if np.amin(np.linalg.norm(decimated_surface_model_positions_array_ones[label_to_array == label_idx] - initial_point, axis=1)) < threshold_distance_prop:
                    pass
                # Otherwise, start a new cluster
                else:
                    label_idx += 1
                label_to_array[np.argmin(np.linalg.norm(decimated_surface_model_positions_array_ones - initial_point, axis=1))] = label_idx

        # To decide wether to keep an initially identified cluster or not, we can look at the counts of the islands
        # If counts are less than a given thresholds, it means that the island is very small and it is probably an error (remember that this value is decimated by a factor, so this means that the undecimated cutoff is much larger)
        values, counts = np.unique(label_to_array, return_counts=True)
        # Clear circular segment model node
        circular_segment_model_node = None
        for val_idx, value in enumerate(values):
            labels_copy = labels.copy()
            idx_aux = 0
            # If counts are larger than threshold_counts, it means it is probably a well-identified segment, then we keep it
            # Otherwise, we discard it as it is probably a sparse island, incorrecly clustered
            if counts[val_idx] > threshold_counts:
                print("Segment with label {}, found with {} counts ({:.2f} % of total counts). \n Creating closed surface model and importing into Slicer...".format(int(value), counts[val_idx], 100 * counts[val_idx] / len(distance_radius_ratio)))
                for idx, label_cluster in enumerate(labels):
                    # For originally clustered points
                    if label_cluster == 1: 
                        if label_to_array[idx_aux] != value:
                            labels_copy[idx] = 0
                        idx_aux += 1

                # Finally, we assign a label to all surface model points that were initially ignored
                # We attibute the label of the closest labelled point
                labels_array = np.ndarray([surface_model.GetNumberOfCells()])
                for idx, point in enumerate(surface_model_positions_array):
                    labels_array[idx] = labels_copy[np.argmin(np.linalg.norm(decimated_surface_model_positions_array - point, axis = 1))]

                labeled_decimated_surface_model = vtk.vtkPolyData()
                labeled_decimated_surface_model.DeepCopy(surface_model)
                labeled_decimated_surface_model.GetCellData().AddArray(vtk.util.numpy_support.numpy_to_vtk(labels_array))
                labeled_decimated_surface_model.GetCellData().GetArray(0).SetName("Label")

                # In order to compute centerlines for the centerline-less vessel, 
                # we split it from the rest of the arterial tree in the form of a closed surface model
                # To do that:
                    # 1) Create an open surface model
                    # 2) Load onto Slicer
                    # 3) Perform centerline extraction

                # Pass cell labels for points
                labels_point_array = np.zeros([surface_model.GetNumberOfPoints()])
                for cell_idx in range(surface_model.GetNumberOfCells()):
                    for idx in range(surface_model.GetCell(cell_idx).GetNumberOfPoints()):
                        labels_point_array[surface_model.GetCell(cell_idx).GetPointId(idx)] = labels_array[cell_idx]

                # Initialize the vtkPoints and the vtkCellArray objects for the circular_segmentModel
                points_circular_segment_model = vtk.vtkPoints()
                cell_array_circular_segment_model = vtk.vtkCellArray()

                point_id_array = np.arange(surface_model.GetNumberOfPoints())[labels_point_array == 1]
                cell_id_array = np.arange(surface_model.GetNumberOfCells())[labels_array == 1]

                for idx in point_id_array:
                    points_circular_segment_model.InsertNextPoint(surface_model.GetPoint(idx))

                # Insert the cells with the corresponding label=1 to the new vtkCellArray
                for idx in cell_id_array:
                    cell = vtk.vtkTriangle()
                    cell.GetPointIds().SetNumberOfIds(surface_model.GetCell(idx).GetNumberOfPoints())
                    for idx2 in range(surface_model.GetCell(idx).GetNumberOfPoints()):
                        if surface_model.GetCell(idx).GetPointId(idx2) in point_id_array:
                            cell.GetPointIds().SetId(idx2, int(np.where(point_id_array == surface_model.GetCell(idx).GetPointId(idx2))[0]))
                        else:
                            point_id_array = np.append(point_id_array, surface_model.GetCell(idx).GetPointId(idx2))
                            points_circular_segment_model.InsertNextPoint(surface_model.GetPoint(surface_model.GetCell(idx).GetPointId(idx2)))
                            cell.GetPointIds().SetId(idx2, int(np.where(point_id_array == surface_model.GetCell(idx).GetPointId(idx2))[0]))
                    cell_array_circular_segment_model.InsertNextCell(cell)

                # First we create a new vtkPolyData for the labelled surface model, only including labelled cells
                circular_segment = vtk.vtkPolyData()
                circular_segment.SetPoints(points_circular_segment_model)
                circular_segment.SetPolys(cell_array_circular_segment_model)

                # We can to compute the normals for all mesh triangles
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(circular_segment)
                normals.SetFeatureAngle(80)
                normals.AutoOrientNormalsOn()
                normals.UpdateInformation()
                normals.Update()
                circular_segment = normals.GetOutput()
                # Clean the vtkPolyData. This allows edges to be correctly identified
                clean_poly_data = vtk.vtkCleanPolyData()
                clean_poly_data.SetInputData(circular_segment)
                clean_poly_data.Update()
                clean_circular_segment = clean_poly_data.GetOutput()

                # Associate it with a MRML model node to load it into the Slicer context
                circular_segment_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
                circular_segment_model_node.SetAndObservePolyData(clean_circular_segment)

                points_numpy = np.ndarray([points_circular_segment_model.GetNumberOfPoints(), 3])
                for point_idx in range(points_circular_segment_model.GetNumberOfPoints()):
                    points_numpy[point_idx] = points_circular_segment_model.GetPoint(point_idx)

                # For now we limit the circular centerline analysis to extracranial vessels 
                # For that, we only look at segments with their center of mass at the lower 80% of the image
                center_of_mass = np.mean(points_numpy, axis = 0)
                min_is = centerline_positions_array[:, 2].min()
                max_is = centerline_positions_array[:, 2].max()

                if center_of_mass[2] - min_is < 0.80 * (max_is - min_is):
                    print(" Performing circular centerline extraction")
                    # Apply the centerline extraction algorithm for circular segments
                    centerline_poly_data = extract_centerline_circular_segment(circular_segment_model_node, centerline_poly_data, segmentation_node, segment_id, aff)
                else:
                    print(" Ignoring segment (it is most likely a cerebral vessel, not fined tuned for circular centerline extraction)")
                print()

        return centerline_poly_data
    
def extract_centerline_circular_segment(input_surface_model_node, centerline_poly_data, segmentation_node, segment_id, aff): 
    """ 
    Special centerline extraction for circular segments.

    This methods is only applied when a circular segment is detected. The closed surface model obtained from
    the circular segment inspection is used to obtain the start- and endpoints for the centerline extraction. 
    In addition, the startpoint from the original segment is used to generate a smooth centerline with the same 
    format as the rest of the centerline cells from the original centerline_poly_data. 
    
    The startpoint will always correspond to the proximal end of the circular segment (relative to the original startpoint). 
    The distal end's endpoint is used to recognize the alternative path's centerline, in order to get the actual endpoint of 
    the centerline cell. Then, two centerlines should be extracted:
        1) One that joins the proximal end of the circular segment with the endpoint of the alternative segment (except last 4 points,
        ignored for robustness).
        2) One that joins the proximal end of the circular segment with the startpoint of the overall segment's centerline model.
        
    The 2nd segment is inverted and then joint to the 1st, and the resulting centerline cell is added to the original 
    centerline_poly_data as a new centerline cell. Then, for a correct extraction of the branch and clipped models, a new centerline cell 
    is created by continuing the alternate segment's cell and the last 4 points of the circular centerline cell.

    Paremeters
    ----------
    input_surface_model_node : vtkMRMLModelNode
        Surface model node corresponding to the centerline-less segment.
    centerline_poly_data : vtk.vtkPolyData
        Centerline model in vtkPolyData form for segment_id segment.
    segmentation_node : vtkMRMLSegmentationNode
        MRML segmentation node.
    segment_id : integer
        Segment identifier in the segmentation_node.
    aff : numpy.array or array-like object. Shape: 4 x 4
        Affine matrix corresponding to the nifti file. RAS to ijk transformation.
    
    Returns
    -------
    final_centerline_poly_data : vtk.vtkPolyData
        Final centerline segment with the new circular segment cells added to centerline_poly_data.
    
    """
    print("Extracting circular segment...")
    # Set up extract centerline widget
    extract_centerline_widget = None
    parameter_node = None
    extract_centerline_widget = slicer.modules.extractcenterline.widgetRepresentation().self()
    # Set up parameter node
    parameter_node = slicer.mrmlScene.GetSingletonNode("ExtractCenterline", "vtkMRMLScriptedModuleNode")
    extract_centerline_widget.setParameterNode(parameter_node)
    extract_centerline_widget.setup()
    # Update from GUI to get segmentation_node as inputSurfaceNode
    extract_centerline_widget.updateParameterNodeFromGUI()
    extract_centerline_widget._parameterNode.SetNodeReferenceID("InputSurface", input_surface_model_node.GetID())

    # Autodetect endpoints
    print("Automatic endpoint extraction...")
    extract_centerline_widget.onAutoDetectEndPoints()
    extract_centerline_widget.updateGUIFromParameterNode()
        
    # Set network node reference to original segment
    extract_centerline_widget._parameterNode.SetNodeReferenceID("InputSurface", segmentation_node.GetID())
    extract_centerline_widget.ui.inputSegmentSelectorWidget.setCurrentSegmentID(segmentation_node.GetSegmentation().GetNthSegmentID(segment_id))

    # Get volume node array from segmentation node
    label_map_volume_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentation_node, label_map_volume_node)
    segmentation_array = slicer.util.arrayFromVolume(label_map_volume_node)

    # Get affine matrix from segmentation labelMapVolumeNode
    vtk_aff = vtk.vtkMatrix4x4()
    aff_eye = np.eye(4)
    label_map_volume_node.GetIJKToRASMatrix(vtk_aff)
    vtk_aff.DeepCopy(aff_eye.ravel(), vtk_aff)

    # Get endpoints node
    endpoints_node = slicer.util.getNode(extract_centerline_widget._parameterNode.GetNodeReferenceID("EndPoints"))
    startpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(0))
    # Remove all nodes except for the startpoint and the one that is further away. We do this because sometimes there are several 
    # endpoints detected close to the origin of the centerline-less segment
    while endpoints_node.GetNumberOfControlPoints() > 2: 
        distances_endpoints = []
        for idx in range(1, endpoints_node.GetNumberOfControlPoints()):
            distances_endpoints.append(np.linalg.norm(startpoint - np.array(endpoints_node.GetCurvePoints().GetPoint(idx))))
        endpoints_node.RemoveNthControlPoint(np.argmin(distances_endpoints) + 1)
        
    print("Relocating endpoints for robust centerline extraction...")
    # Relocate endpoints for robust centerline extraction 
    for idx in range(endpoints_node.GetNumberOfControlPoints()):
        endpoint = np.array(endpoints_node.GetCurvePoints().GetPoint(idx))
        new_endpoint = robust_end_point_detection(endpoint, segmentation_array, aff_eye) # Center of mass of closest component method
        # Search for alternative centerline to pass closes centerline point to detected endpoint as new endpoint
        if idx > 0:
            closest_cell_points = np.ndarray([centerline_poly_data.GetNumberOfCells()])
            for cell_idx in range(centerline_poly_data.GetNumberOfCells()):
                centerline_cell_points = np.ndarray([centerline_poly_data.GetCell(cell_idx).GetNumberOfPoints(), 3])
                for point_idx in range(centerline_poly_data.GetCell(cell_idx).GetNumberOfPoints()):
                    centerline_cell_points[point_idx] = centerline_poly_data.GetCell(cell_idx).GetPoints().GetPoint(point_idx)
                closest_cell_points[cell_idx]= np.amin(np.linalg.norm(new_endpoint - centerline_cell_points, axis = 1))
            new_endpoint = centerline_poly_data.GetCell(np.argmin(closest_cell_points)).GetPoints().GetPoint(centerline_poly_data.GetCell(np.argmin(closest_cell_points)).GetNumberOfPoints() - 1)
        endpoints_node.SetNthControlPointPosition(idx, new_endpoint[0],
                                                    new_endpoint[1],
                                                    new_endpoint[2])
    
    # We add startpoint of original segment as endpoint
    startpoint = centerline_poly_data.GetCell(0).GetPoints().GetPoint(0)
    endpoints_node.AddControlPoint(np.array(startpoint))

    # Now, the startpoint of the centerline-less endpoints node will be the detected endpoint closest to the startpoint of the
    # original centerline_poly_data. We should have two additional endpoints: one corresponding to the centerline point from 
    # the original centerline_poly_data closest to the robust endpoint detected at the other end of the centerline-less segment
    # and another one at the startpoint of the original centerline_poly_data
    print("Extracting centerline...")
    # Create new model node for the centerline model
    centerline_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    # Set centerline node to new empty node
    extract_centerline_widget._parameterNode.SetNodeReferenceID("CenterlineModel", centerline_model_node.GetID())
    extract_centerline_widget.onApplyButton()

    print("Checking for floating centerlines (errors)...")
    # Check if all centerlines depart from the same origin. Dismiss the ones that do not, they are most likely floating
    circular_centerline_poly_data = centerline_model_node.GetPolyData()

    # Decimate centerline model
    # decimator = vtk.vtkDecimatePolylineFilter()
    # decimator.SetTargetReduction(0.5)
    # decimator.AddInputData(circular_centerline_poly_data)
    # decimator.Update()
    # circular_centerline_poly_data = decimator.GetOutput()

    # Declare empty arrays
    cells_id_array = np.ndarray([circular_centerline_poly_data.GetNumberOfCells()], dtype=int)
    cells_first_coordinate_array = np.ndarray([circular_centerline_poly_data.GetNumberOfCells(), 3])

    # Iterate over cells to extract cell_ids, positions and radii. Store lengths of cells
    for cell_id in range(circular_centerline_poly_data.GetNumberOfCells()):
        cells_id_array[cell_id] = cell_id
        cell = vtk.vtkGenericCell()
        circular_centerline_poly_data.GetCell(cell_id, cell)
        cells_first_coordinate_array[cell_id] = np.matmul(aff, np.append(cell.GetPoints().GetPoint(0), 1.0))[:3]

    unique_cells_first_coordinate_array, counts = np.unique(cells_first_coordinate_array, return_counts=True, axis=0)

    # Get all those that do not start at the startpoint, and remove them
    remove_floating = []
    for idx in range(len(unique_cells_first_coordinate_array)):
        if idx != np.argmax(counts):
            for idx2 in range(circular_centerline_poly_data.GetNumberOfCells()):
                if (unique_cells_first_coordinate_array[idx] == cells_first_coordinate_array[idx2]).all(): remove_floating.append(idx2)

    if len(remove_floating) > 0:
        print("Found floating centerlines: {}. Suspending circular centerline extraction".format(remove_floating))
        return centerline_poly_data
    else:
        print("No errors found")

    print("Building final centerline segment")
    # We join both cells to create the final cell of the original centerline_poly_data 
    # Then we add it to the centerline_poly_data as a new vtkPolyLine cell
    
    # Initialize the new vtkPoints with those from the original centerline_poly_data
    final_points = vtk.vtkPoints()
    final_points.DeepCopy(centerline_poly_data.GetPoints())
    # Get the numpy arrays for the point_data arrays
    radius_numpy = vtk.util.numpy_support.vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Radius"))
    edge_array_numpy = vtk.util.numpy_support.vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("EdgeArray"))
    edge_p_coord_array_numpy = vtk.util.numpy_support.vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("EdgePCoordArray"))
    # We start a new vtkCellArrray with the cells from the original centerline_poly_data
    final_cell_array = vtk.vtkCellArray()
    for cell_idx in range(centerline_poly_data.GetNumberOfCells()):
        final_cell_array.InsertNextCell(centerline_poly_data.GetCell(cell_idx))
    
    # Now, we join both cells from the circular_centerline_poly_data as one with the correct order
    # We have to search for the exact preceeding centerline segment from the original centerline_poly_data
    # First store all centerlines from the original centerline_poly_data in numpy arrays for speed
    centerline_cell_arrays = np.ndarray([centerline_poly_data.GetNumberOfCells()], dtype = object)
    for centerline_cell_idx in range(centerline_poly_data.GetNumberOfCells()):
        centerline_cell_arrays[centerline_cell_idx] = np.ndarray([centerline_poly_data.GetCell(centerline_cell_idx).GetNumberOfPoints(), 3])
        for centerline_point_idx in range(centerline_poly_data.GetCell(centerline_cell_idx).GetNumberOfPoints()):
            centerline_cell_arrays[centerline_cell_idx][centerline_point_idx] = centerline_poly_data.GetCell(centerline_cell_idx).GetPoints().GetPoint(centerline_point_idx)

    # Then, for each circular centerline point, search for closest point to each centerline_poly_data cell and get the first under 0.1 mm    
    for circular_centerline_prox_point_idx in range(circular_centerline_poly_data.GetCell(1).GetNumberOfPoints()):
        point = circular_centerline_poly_data.GetCell(1).GetPoints().GetPoint(circular_centerline_prox_point_idx)
        for centerline_cell_idx in range(centerline_poly_data.GetNumberOfCells()):
            distances = np.linalg.norm(point - centerline_cell_arrays[centerline_cell_idx], axis = 1)
            if np.amin(distances) < 0.1:
                centerline_point_idx = np.argmin(distances)
                break # centerline_cell_idx, centerline_point_idx correspond to the original centerline_poly_data
                      # circular_centerline_prox_point_idx - 1 will be the first point used from the circular_centerline
                      # We will concatenate both
        if np.amin(distances) < 0.1:
            break
            
    # We can then search for the distal branch of the centerline model. Branching and clipping present problems if nothing else is done
    # The idea is to split the distal end into a new centerline cell. Then, for each centerline point, search for closest point to each 
    # centerline_poly_data cell (now centerline_cell_idx_2) and get the first under 0.1 mm    
    for circular_centerline_dist_point_idx in range(1, circular_centerline_poly_data.GetCell(0).GetNumberOfPoints()):
        point = circular_centerline_poly_data.GetCell(0).GetPoints().GetPoint(circular_centerline_dist_point_idx)
        for centerline_cell_idx_2 in range(centerline_poly_data.GetNumberOfCells()):
            distances = np.linalg.norm(point - centerline_cell_arrays[centerline_cell_idx_2], axis = 1)
            if np.amin(distances) < 0.1:
                centerline_point_idx_2 = np.argmin(distances)
                break # centerline_cell_idx_2, centerline_point_idx_2 correspond to the preceeding centerline
                      # circular_centerline_dist_point_idx - 1 will be the first point used from the circular_centerline
                      # We will concatenate both
        if np.amin(distances) < 0.1:
            break
    
    # We define the necessary objects for the circular centerline cell
    circular_centerline_poly_line = vtk.vtkPolyLine()
    circular_centerline_poly_line_points = vtk.vtkPoints()
    circular_centerline_poly_line_point_ids = []
    # We have to be coherent with the point Ids of the new cell
    previous_number_of_points = centerline_poly_data.GetNumberOfPoints()
    idx_aux = 0
    
    # We build the final centerline_poly_data
    for cell_idx in range(circular_centerline_poly_data.GetNumberOfCells() - 1, -1, -1):
        circular_centerline_cell = circular_centerline_poly_data.GetCell(cell_idx)
        # Start from the second cell, in inverse order
        if cell_idx == 1:
            # We first add the first segment fromt he original centerline_poly_data
            for point_idx in range(centerline_point_idx):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circular_centerline_poly_line_points.InsertNextPoint(centerline_poly_data.GetCell(centerline_cell_idx).GetPoints().GetPoint(point_idx))
                final_points.InsertNextPoint(centerline_poly_data.GetCell(centerline_cell_idx).GetPoints().GetPoint(point_idx))
                circular_centerline_poly_line_point_ids.append(previous_number_of_points + idx_aux)
                idx_aux += 1
                # Append point_data values to the numpy arrays
                radius_numpy = np.append(radius_numpy, centerline_poly_data.GetPointData().GetArray("Radius").GetValue(centerline_poly_data.GetCell(centerline_cell_idx).GetPointId(point_idx)))
                edge_array_numpy = np.append(edge_array_numpy, np.array([centerline_poly_data.GetPointData().GetArray("EdgeArray").GetTuple(centerline_poly_data.GetCell(centerline_cell_idx).GetPointId(point_idx))]), axis=0)
                edge_p_coord_array_numpy = np.append(edge_p_coord_array_numpy, centerline_poly_data.GetPointData().GetArray("EdgePCoordArray").GetValue(centerline_poly_data.GetCell(centerline_cell_idx).GetPointId(point_idx)))
            # We then add the rest of the second cell (1) of the circular_centerline_poly_data. We go backwards because we want to invert the order
            for point_idx in range(circular_centerline_prox_point_idx - 1, -1, -1):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circular_centerline_poly_line_points.InsertNextPoint(circular_centerline_cell.GetPoints().GetPoint(point_idx))
                final_points.InsertNextPoint(circular_centerline_cell.GetPoints().GetPoint(point_idx))
                circular_centerline_poly_line_point_ids.append(previous_number_of_points + idx_aux)
                idx_aux += 1
                # Append point_data values to the numpy arrays
                radius_numpy = np.append(radius_numpy, circular_centerline_poly_data.GetPointData().GetArray("Radius").GetValue(circular_centerline_cell.GetPointId(point_idx)))
                edge_array_numpy = np.append(edge_array_numpy, np.array([circular_centerline_poly_data.GetPointData().GetArray("EdgeArray").GetTuple(circular_centerline_cell.GetPointId(point_idx))]), axis=0)
                edge_p_coord_array_numpy = np.append(edge_p_coord_array_numpy, circular_centerline_poly_data.GetPointData().GetArray("EdgePCoordArray").GetValue(circular_centerline_cell.GetPointId(point_idx)))

        elif cell_idx == 0:
            for point_idx in range(circular_centerline_dist_point_idx - 3):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circular_centerline_poly_line_points.InsertNextPoint(circular_centerline_cell.GetPoints().GetPoint(point_idx))
                final_points.InsertNextPoint(circular_centerline_cell.GetPoints().GetPoint(point_idx))
                circular_centerline_poly_line_point_ids.append(previous_number_of_points + idx_aux)
                idx_aux += 1
                # Append point_data values to the numpy arrays
                radius_numpy = np.append(radius_numpy, circular_centerline_poly_data.GetPointData().GetArray("Radius").GetValue(circular_centerline_cell.GetPointId(point_idx)))
                edge_array_numpy = np.append(edge_array_numpy, np.array([circular_centerline_poly_data.GetPointData().GetArray("EdgeArray").GetTuple(circular_centerline_cell.GetPointId(point_idx))]), axis=0)
                edge_p_coord_array_numpy = np.append(edge_p_coord_array_numpy, circular_centerline_poly_data.GetPointData().GetArray("EdgePCoordArray").GetValue(circular_centerline_cell.GetPointId(point_idx)))
            # Create a new cell with the poits up to centerline_cell_idx_2 and a few points (e.g., 2-10. We take 4) from the circular segment. This can solve several issues in branch model, clipped model and segments_array computation
            # We define the necessary objects for the circular centerline cell
            circular_centerline_poly_line_2 = vtk.vtkPolyLine()
            circular_centerline_poly_line_points_2 = vtk.vtkPoints()
            circular_centerline_poly_line_point_ids_2 = []
            for point_idx in range(centerline_point_idx_2):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circular_centerline_poly_line_points_2.InsertNextPoint(centerline_poly_data.GetCell(centerline_cell_idx_2).GetPoints().GetPoint(point_idx))
                final_points.InsertNextPoint(centerline_poly_data.GetCell(centerline_cell_idx_2).GetPoints().GetPoint(point_idx))
                circular_centerline_poly_line_point_ids_2.append(previous_number_of_points + idx_aux)
                idx_aux += 1
                # Append point_data values to the numpy arrays
                radius_numpy = np.append(radius_numpy, centerline_poly_data.GetPointData().GetArray("Radius").GetValue(centerline_poly_data.GetCell(centerline_cell_idx_2).GetPointId(point_idx)))
                edge_array_numpy = np.append(edge_array_numpy, np.array([centerline_poly_data.GetPointData().GetArray("EdgeArray").GetTuple(centerline_poly_data.GetCell(centerline_cell_idx_2).GetPointId(point_idx))]), axis=0)
                edge_p_coord_array_numpy = np.append(edge_p_coord_array_numpy, centerline_poly_data.GetPointData().GetArray("EdgePCoordArray").GetValue(centerline_poly_data.GetCell(centerline_cell_idx_2).GetPointId(point_idx)))
            for point_idx_aux in range(4, -1, -1):
                # Add points to the cell and the vtkPoints. Keep new pointIds
                circular_centerline_poly_line_points_2.InsertNextPoint(circular_centerline_cell.GetPoints().GetPoint(circular_centerline_dist_point_idx - 4 + point_idx_aux))
                final_points.InsertNextPoint(circular_centerline_cell.GetPoints().GetPoint(circular_centerline_dist_point_idx - 4 + point_idx_aux))
                circular_centerline_poly_line_point_ids_2.append(previous_number_of_points + idx_aux)
                idx_aux += 1
                # Append point_data values to the numpy arrays
                radius_numpy = np.append(radius_numpy, circular_centerline_poly_data.GetPointData().GetArray("Radius").GetValue(circular_centerline_cell.GetPointId(circular_centerline_dist_point_idx - 4 + point_idx_aux)))
                edge_array_numpy = np.append(edge_array_numpy, np.array([circular_centerline_poly_data.GetPointData().GetArray("EdgeArray").GetTuple(circular_centerline_cell.GetPointId(circular_centerline_dist_point_idx - 4 + point_idx_aux))]), axis=0)
                edge_p_coord_array_numpy = np.append(edge_p_coord_array_numpy, circular_centerline_poly_data.GetPointData().GetArray("EdgePCoordArray").GetValue(circular_centerline_cell.GetPointId(circular_centerline_dist_point_idx - 4 + point_idx_aux)))                           
    
    # Build final cell for the circular centerline and ad it to the rest in the vtkCellArray
    circular_centerline_poly_line.Initialize(len(circular_centerline_poly_line_point_ids), circular_centerline_poly_line_point_ids, circular_centerline_poly_line_points)
    final_cell_array.InsertNextCell(circular_centerline_poly_line)
    circular_centerline_poly_line_2.Initialize(len(circular_centerline_poly_line_point_ids_2), circular_centerline_poly_line_point_ids_2, circular_centerline_poly_line_points_2)
    final_cell_array.InsertNextCell(circular_centerline_poly_line_2)
    # Build a new vtkPolyData object and add the vtkPoints and the vtkCellArray objects
    final_centerline_poly_data = vtk.vtkPolyData()
    final_centerline_poly_data.SetPoints(final_points)
    final_centerline_poly_data.SetLines(final_cell_array)
    # Add the PointData
    final_centerline_poly_data.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(radius_numpy))
    final_centerline_poly_data.GetPointData().GetArray(0).SetName("Radius")
    final_centerline_poly_data.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(edge_array_numpy))
    final_centerline_poly_data.GetPointData().GetArray(1).SetName("EdgeArray")
    final_centerline_poly_data.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(edge_p_coord_array_numpy))
    final_centerline_poly_data.GetPointData().GetArray(2).SetName("EdgePCoordArray")
    
    return final_centerline_poly_data

def compute_frenet_serret(centerline_poly_data):
    """
    Uses the vtkParallelTransportFrame custom filter from
    Slicer to compute tangent, normal and binormal vectors from 
    the Frenet-Serret frame for each point of the centerline model.

    Parameters
    ----------
    centerline_poly_data : vtk.vtkPolyData
        Centerline model.
    
    Returns
    -------
    centerline_poly_data : vtk.vtkPolyData
        Centerline model with tangent, normal and binormal vectors 
        computed for each centerline point as point data.
    
    """
    curve_coordinate_system_generator = slicer.vtkParallelTransportFrame()
    curve_coordinate_system_generator.SetInputData(centerline_poly_data)
    curve_coordinate_system_generator.Update()

    return curve_coordinate_system_generator.GetOutput()

def compute_curvature_and_torsion(centerline_poly_data):
    """
    Compute cruvature and torsion using the Frenet-Serret moving frame.
    Assumes that Frenet-Serret vectors are available as point data in the 
    input centerline_poly_data object. Stores curvature and torsion as
    point data arrays in the centerline_poly_data object. Also computes 
    a smoothed out version of the curvature using a Savitzky-Golay filter.

    Parameters
    ----------
    centerline_poly_data : vtk.vtkPolyData
        Centerline model.

    Returns
    ------
    centerline_poly_data : vtk.vtkPolyData
        Centerline model with curvature, torsion and filtered_curvature 
        as additional point data arrays.

    References:
    [1]     
    
    """
    # Load Frenet-Serret vectors as numpy arrays for each point
    coordinates = np.ndarray([centerline_poly_data.GetNumberOfPoints(), 3])
    for point_idx in range(centerline_poly_data.GetNumberOfPoints()):
        coordinates[point_idx] = centerline_poly_data.GetPoints().GetPoint(point_idx)
    tangents = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Tangents"))
    normals = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Normals"))
    binormals = vtk_to_numpy(centerline_poly_data.GetPointData().GetArray("Binormals"))
    
    # Compute curvature and torsion
    curvature = np.linalg.norm(np.gradient(tangents, axis = 0), axis = 1)
    torsion = (- np.gradient(binormals, axis = 0) * normals).sum(axis = 1)
    # Compute smoothed curvature
    filtered_curvature = savgol_filter(curvature, window_length=10, polyorder=3, mode='nearest')
    
    # Add new point data arrays
    centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(curvature))
    centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Curvature")
    centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(torsion))
    centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Torsion")
    centerline_poly_data.GetPointData().AddArray(numpy_to_vtk(filtered_curvature))
    centerline_poly_data.GetPointData().GetArray(centerline_poly_data.GetPointData().GetNumberOfArrays() - 1).SetName("Filtered curvature")

    return centerline_poly_data