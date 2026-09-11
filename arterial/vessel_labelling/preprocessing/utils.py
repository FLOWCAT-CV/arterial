#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import numpy as np

def _unit_vector(vector):
    """
    Normalises a vector, returning zeros for a (near) zero-length input.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector to normalise.

    Returns
    -------
    unit : numpy.ndarray
        vector / ||vector||, or zeros when the norm is below 1e-6.

    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else np.zeros_like(vector, dtype=float)


def extract_features_for_labelling(simple_centerline_graph):
    """
    Extracts segment-level features for labelling.

    Parmeters
    ---------
    simple_centerline_graph : networkx.Graph
        Simple centerline graph constructed from the centerline_centerline_segments_array.

    Returns
    -------
    simple_centerline_graph : networkx.Graph
        Simple featurized centerline graph.

    """
    def relative_length(segment_coordinates):
        """
        Computes relative length from a segment. The relative length is defined as the ratio 
        between the Euclidean distance between two endpoints of a centerline segment divided
        by the actual length of the centerline, computed as the line integral between both
        endpoints.
    
        Parmeters
        ---------
        segment_coordinates : numpy.array or array-like object
            Array containing all 3D coordinates of a centerline segment.

        Returns
        -------
        relative_length : float
            Result of the relative length computation.

        """
        def distance_along_centerline(centerline):
            """
            Auxiliary function to perform the numerical line integral to compute the length of 
            a centerline segment.
        
            Parmeters
            ---------
            centerline : numpy.array or array-like object
                Array containing all 3D coordinates of a centerline segment.

            Returns
            -------
            distance : float
                Result of the distance computation.

            """
            distance = 0
            for idx in range(1, len(centerline)):
                distance += np.linalg.norm(centerline[idx] - centerline[idx - 1])
                
            return distance
            
        if np.linalg.norm(segment_coordinates[-1] - segment_coordinates[0]) < 1e-6:
            return 0.0
        else:
            euclidean_distance = np.linalg.norm(segment_coordinates[-1] - segment_coordinates[0])
            centerline_distance = distance_along_centerline(segment_coordinates)
            return euclidean_distance / centerline_distance

    for src, dst in simple_centerline_graph.edges:
        coordinate_array = simple_centerline_graph[src][dst]["coordinate_array"]
        radius_array = simple_centerline_graph[src][dst]["radius_array"]

        simple_centerline_graph[src][dst]["pos"] = np.mean(coordinate_array, axis = 0)
        
        # Build edge feature array and dict
        simple_centerline_graph[src][dst]["features_dict"] = {}
        simple_centerline_graph[src][dst]["features_dict"]["mean radius"] = np.mean(radius_array)
        simple_centerline_graph[src][dst]["features_dict"]["proximal radius"] = radius_array[0]
        simple_centerline_graph[src][dst]["features_dict"]["distal radius"] = radius_array[-1]
        simple_centerline_graph[src][dst]["features_dict"]["proximal/distal radius ratio"] = radius_array[0] / radius_array[-1] if radius_array[-1] > 1e-6 else 0.0
        simple_centerline_graph[src][dst]["features_dict"]["minimum radius"] = np.amin(radius_array)
        simple_centerline_graph[src][dst]["features_dict"]["maximum radius"] = np.amax(radius_array)
        simple_centerline_graph[src][dst]["features_dict"]["distance"] = np.linalg.norm(coordinate_array[-1] - coordinate_array[0])
        simple_centerline_graph[src][dst]["features_dict"]["relative length"] = relative_length(coordinate_array)
        simple_centerline_graph[src][dst]["features_dict"]["direction r"] = _unit_vector(coordinate_array[-1] - coordinate_array[0])[0]
        simple_centerline_graph[src][dst]["features_dict"]["direction a"] = _unit_vector(coordinate_array[-1] - coordinate_array[0])[1]
        simple_centerline_graph[src][dst]["features_dict"]["direction s"] = _unit_vector(coordinate_array[-1] - coordinate_array[0])[2]
        # To compute departure angle, we will be computing the direction of the first 10 mm of the segment from the bifurcation point
        idx = 1
        while np.linalg.norm(coordinate_array[idx] - coordinate_array[0]) < 10 and idx < len(coordinate_array) - 1:
            idx += 1
        simple_centerline_graph[src][dst]["features_dict"]["departure angle r"] = _unit_vector(coordinate_array[idx] - coordinate_array[0])[0]
        simple_centerline_graph[src][dst]["features_dict"]["departure angle a"] = _unit_vector(coordinate_array[idx] - coordinate_array[0])[1]
        simple_centerline_graph[src][dst]["features_dict"]["departure angle s"] = _unit_vector(coordinate_array[idx] - coordinate_array[0])[2]
        simple_centerline_graph[src][dst]["features_dict"]["number of points"] = len(coordinate_array)
        simple_centerline_graph[src][dst]["features_dict"]["proximal bifurcation position r"] = coordinate_array[0][0]
        simple_centerline_graph[src][dst]["features_dict"]["proximal bifurcation position a"] = coordinate_array[0][1]
        simple_centerline_graph[src][dst]["features_dict"]["proximal bifurcation position s"] = coordinate_array[0][2]
        simple_centerline_graph[src][dst]["features_dict"]["distal bifurcation position r"] = coordinate_array[-1][0]
        simple_centerline_graph[src][dst]["features_dict"]["distal bifurcation position a"] = coordinate_array[-1][1]
        simple_centerline_graph[src][dst]["features_dict"]["distal bifurcation position s"] = coordinate_array[-1][2]
        simple_centerline_graph[src][dst]["features_dict"]["pos r"] = np.mean(coordinate_array, axis = 0)[0]
        simple_centerline_graph[src][dst]["features_dict"]["pos a"] = np.mean(coordinate_array, axis = 0)[1]
        simple_centerline_graph[src][dst]["features_dict"]["pos s"] = np.mean(coordinate_array, axis = 0)[2]
        # Now the array
        simple_centerline_graph[src][dst]["features"] = np.array(list(simple_centerline_graph[src][dst]["features_dict"].values()))

    return simple_centerline_graph