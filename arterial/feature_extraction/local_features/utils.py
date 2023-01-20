#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import math

import numpy as np

def get_max_hierarchy(centerline_graph, access = "femoral"):
    """ 
    Iterates over nodes to find max hierarchy for each access configuration.

    Parameters
    ----------
    centerline_graph : network.Graph
        Dense centerline graph returned by graph builder.
    access : string
        Access site for thrombectomy configuration. Can be either "femoral" or "radial".

    Returns
    -------
    max_hierarchy : integer
        Maximum hierarchical index for `access`.
    
    """
    max_hierarchy = 0
    for node in centerline_graph:
        if centerline_graph.nodes[node][f"hierarchy {access}"] > max_hierarchy:
            max_hierarchy = centerline_graph.nodes[node][f"hierarchy {access}"]
                   
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
    Computes curvature along a curve.

    Parameters
    ----------
    curve : numpy.array or array-like object.
        Array containing the coordinates of a set of ordered points from
        a curve.

    Returns
    -------
    radius_of_curvature : numpy.array or array-like object.
        Array containing the radius of curvature at each point of the curve.

    """
    # Change the sign of the first coordinate of the curve points to make pass it to a positively oriented system
    curve[:, 0] = - curve[:, 0]
    # Compute the time tangent of the curve
    time_tangent = np.gradient(curve, axis = 0)
    # Compute the inverse of the derivative of the arclength wrt time
    dtds = 1 / np.linalg.norm(time_tangent, axis = 1)
    # Compute the tangent vector at all points of the curve
    tangent = time_tangent * np.expand_dims(dtds, axis = 1)
    # The curvature is simply the norm of the derivative of the tangent wrt arclength
    curvature = np.linalg.norm(np.gradient(tangent, axis = 0), axis = 1)
    # Compute the differential of the tangent
    diff_tangent = np.gradient(tangent, axis = 0)
    # Compute the non-normalized normal vector at every point
    normal = diff_tangent
    # Normalize to 1
    normal = normal / np.expand_dims(np.linalg.norm(normal, axis = 1), axis = 1)
    # Compute the binormal vector
    binormal = np.cross(tangent, normal)
    # The torsion is given by the Frenet-Serret formulas
    torsion = (- np.gradient(binormal, axis = 0) * np.expand_dims(dtds, axis = 1) * normal).sum(axis = 1)

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