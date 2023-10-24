#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

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
            
        euclidean_distance = np.linalg.norm(segment_coordinates[-1] - segment_coordinates[0])
        centerline_distance = distance_along_centerline(segment_coordinates)

        return euclidean_distance / centerline_distance

    for src, dst in simple_centerline_graph.edges:
        coordinate_array = simple_centerline_graph[src][dst]["coordinate_array"]
        radius_array = simple_centerline_graph[src][dst]["radius_array"]

        simple_centerline_graph[src][dst]["pos"] = np.sum(coordinate_array, axis = 0) / len(coordinate_array)
        
        # Build edge feature array
        simple_centerline_graph[src][dst]["features"] = {}
        simple_centerline_graph[src][dst]["features"]["mean radius"] = np.mean(radius_array)
        simple_centerline_graph[src][dst]["features"]["proximal radius"] = radius_array[0]
        simple_centerline_graph[src][dst]["features"]["distal radius"] = radius_array[-1]
        simple_centerline_graph[src][dst]["features"]["proximal/distal radius ratio"] = radius_array[0] / radius_array[-1]
        simple_centerline_graph[src][dst]["features"]["minimum radius"] = np.amin(radius_array)
        simple_centerline_graph[src][dst]["features"]["maximum radius"] = np.amax(radius_array)
        simple_centerline_graph[src][dst]["features"]["distance"] = np.linalg.norm(coordinate_array[-1] - coordinate_array[0])
        simple_centerline_graph[src][dst]["features"]["relative length"] = relative_length(coordinate_array)
        simple_centerline_graph[src][dst]["features"]["direction r"] = ((coordinate_array[-1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[-1] - coordinate_array[0]))[0]
        simple_centerline_graph[src][dst]["features"]["direction a"] = ((coordinate_array[-1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[-1] - coordinate_array[0]))[1]
        simple_centerline_graph[src][dst]["features"]["direction s"] = ((coordinate_array[-1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[-1] - coordinate_array[0]))[2]
        simple_centerline_graph[src][dst]["features"]["departure angle r"] = ((coordinate_array[1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[1] - coordinate_array[0]))[0]
        simple_centerline_graph[src][dst]["features"]["departure angle a"] = ((coordinate_array[1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[1] - coordinate_array[0]))[1]
        simple_centerline_graph[src][dst]["features"]["departure angle s"] = ((coordinate_array[1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[1] - coordinate_array[0]))[2]
        simple_centerline_graph[src][dst]["features"]["number of points"] = len(coordinate_array)
        simple_centerline_graph[src][dst]["features"]["proximal bifurcation position r"] = coordinate_array[0][0]
        simple_centerline_graph[src][dst]["features"]["proximal bifurcation position a"] = coordinate_array[0][1]
        simple_centerline_graph[src][dst]["features"]["proximal bifurcation position s"] = coordinate_array[0][2]
        simple_centerline_graph[src][dst]["features"]["distal bifurcation position r"] = coordinate_array[-1][0]
        simple_centerline_graph[src][dst]["features"]["distal bifurcation position a"] = coordinate_array[-1][1]
        simple_centerline_graph[src][dst]["features"]["distal bifurcation position s"] = coordinate_array[-1][2]
        simple_centerline_graph[src][dst]["features"]["pos r"] = np.sum(coordinate_array, axis = 0)[0] / len(coordinate_array)
        simple_centerline_graph[src][dst]["features"]["pos a"] = np.sum(coordinate_array, axis = 0)[1] / len(coordinate_array)
        simple_centerline_graph[src][dst]["features"]["pos s"] = np.sum(coordinate_array, axis = 0)[2] / len(coordinate_array)

    return simple_centerline_graph

def make_graph_plot(case_dir, graph, filename, label = None):
    """
    Makes matplotlib.pyplot figure of the coronal plane of a networkx graph.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    graph : networkx.Graph
        Graph that we want to plot.
    filename : string
        Fine name of the final image. Make sure to add a valid extension (e.g. .png, .eps, etc)
    label : string
        Edge attribute to be printed at the center of each graph edge.

    Returns
    -------

    """
    # Generate plot of dense graph for quick visualization
    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()

    # In order to place the nodes in the visualization of the graph in a sagittal view, 
    # we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_p = {}
    for n in graph.nodes():
        node_pos_dict_p[n] = [-graph.nodes(data=True)[n]["pos"][0], graph.nodes(data=True)[n]["pos"][2]]

    if label is not None:
        edge_labels = nx.get_edge_attributes(graph, label)
        nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
        nx.draw_networkx_edge_labels(graph, node_pos_dict_p, edge_labels = edge_labels, ax=ax)
    else:
        nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
        
    # Add labels to the plot
    plt.text(1, 0.5, 'L', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, 0.5, 'R', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, 1, 'S', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
    plt.text(0.5, 0, 'I', horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)

    plt.savefig(os.path.join(case_dir, filename))
    plt.close()