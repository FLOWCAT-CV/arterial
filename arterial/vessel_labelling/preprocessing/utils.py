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
        
        # Build edge feature array and dict
        simple_centerline_graph[src][dst]["features_dict"] = {}
        simple_centerline_graph[src][dst]["features_dict"]["mean radius"] = np.mean(radius_array)
        simple_centerline_graph[src][dst]["features_dict"]["proximal radius"] = radius_array[0]
        simple_centerline_graph[src][dst]["features_dict"]["distal radius"] = radius_array[-1]
        simple_centerline_graph[src][dst]["features_dict"]["proximal/distal radius ratio"] = radius_array[0] / radius_array[-1]
        simple_centerline_graph[src][dst]["features_dict"]["minimum radius"] = np.amin(radius_array)
        simple_centerline_graph[src][dst]["features_dict"]["maximum radius"] = np.amax(radius_array)
        simple_centerline_graph[src][dst]["features_dict"]["distance"] = np.linalg.norm(coordinate_array[-1] - coordinate_array[0])
        simple_centerline_graph[src][dst]["features_dict"]["relative length"] = relative_length(coordinate_array)
        simple_centerline_graph[src][dst]["features_dict"]["direction r"] = ((coordinate_array[-1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[-1] - coordinate_array[0]))[0]
        simple_centerline_graph[src][dst]["features_dict"]["direction a"] = ((coordinate_array[-1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[-1] - coordinate_array[0]))[1]
        simple_centerline_graph[src][dst]["features_dict"]["direction s"] = ((coordinate_array[-1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[-1] - coordinate_array[0]))[2]
        simple_centerline_graph[src][dst]["features_dict"]["departure angle r"] = ((coordinate_array[1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[1] - coordinate_array[0]))[0]
        simple_centerline_graph[src][dst]["features_dict"]["departure angle a"] = ((coordinate_array[1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[1] - coordinate_array[0]))[1]
        simple_centerline_graph[src][dst]["features_dict"]["departure angle s"] = ((coordinate_array[1] - coordinate_array[0]) / np.linalg.norm(coordinate_array[1] - coordinate_array[0]))[2]
        simple_centerline_graph[src][dst]["features_dict"]["number of points"] = len(coordinate_array)
        simple_centerline_graph[src][dst]["features_dict"]["proximal bifurcation position r"] = coordinate_array[0][0]
        simple_centerline_graph[src][dst]["features_dict"]["proximal bifurcation position a"] = coordinate_array[0][1]
        simple_centerline_graph[src][dst]["features_dict"]["proximal bifurcation position s"] = coordinate_array[0][2]
        simple_centerline_graph[src][dst]["features_dict"]["distal bifurcation position r"] = coordinate_array[-1][0]
        simple_centerline_graph[src][dst]["features_dict"]["distal bifurcation position a"] = coordinate_array[-1][1]
        simple_centerline_graph[src][dst]["features_dict"]["distal bifurcation position s"] = coordinate_array[-1][2]
        simple_centerline_graph[src][dst]["features_dict"]["pos r"] = np.sum(coordinate_array, axis = 0)[0] / len(coordinate_array)
        simple_centerline_graph[src][dst]["features_dict"]["pos a"] = np.sum(coordinate_array, axis = 0)[1] / len(coordinate_array)
        simple_centerline_graph[src][dst]["features_dict"]["pos s"] = np.sum(coordinate_array, axis = 0)[2] / len(coordinate_array)
        # Now the array
        simple_centerline_graph[src][dst]["features"] = np.array(list(simple_centerline_graph[src][dst]["features_dict"].values()))

    return simple_centerline_graph

def make_graph_plot(case_dir, graph, filename = None, label = None, subplot = None):
    """
    Makes matplotlib.pyplot figure of the coronal plane of a networkx graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that we want to plot.
    label : string
        Edge attribute to be printed at the center of each graph edge.
    subplot : matplotlib.axes.Axes
        Subplot on which to draw the graph.

    Returns
    -------

    """
    if subplot is None:
        fig, ax = plt.subplots(figsize=[5, 10])
    else:
        ax = subplot

    # In order to place the nodes in the visualization of the graph in a sagittal view, 
    # we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_p = {}
    for n in graph.nodes():
        node_pos_dict_p[n] = [-graph.nodes(data=True)[n]["pos"][0], graph.nodes(data=True)[n]["pos"][2]]

    # Draw nodes
    nx.draw_networkx_nodes(graph, node_pos_dict_p, node_size=20, ax=ax)

    # Create a list of colors
    colors = plt.cm.seismic(np.linspace(0, 1, len(graph.edges())))

    # Draw edges
    for (src, dst, data), color in zip(graph.edges(data=True), colors):
        if 'coordinate_array' in data:
            ax.plot(-data['coordinate_array'][:, 0], data['coordinate_array'][:, 2], color = color, linewidth = 1)
            
            # Print label at the center of the segments array
            if label is not None and label in data:
                mid_point_idx = len(data['coordinate_array']) // 2
                mid_point = data['coordinate_array'][mid_point_idx]
                # We add a small displacement to the left to the label to better fit the center of the line
                ax.text(-mid_point[0] - 3, mid_point[2], str(data[label]), color = color, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha = 0.8))
        else:
            # If no coordinate_array is present, just draw a line between the two nodes
            if label is not None:
                edge_labels = nx.get_edge_attributes(graph, label)
                nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
                nx.draw_networkx_edge_labels(graph, node_pos_dict_p, edge_labels=edge_labels, ax=ax)
            else:
                nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
            break

    # Add labels to the plot
    plt.text(0.99, 0.5, 'L', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    plt.text(0.01, 0.5, 'R', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.text(0.5, 0.99, 'S', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.text(0.5, 0.01, 'I', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    if subplot is None:
        plt.show()
        plt.savefig(os.path.join(case_dir, filename))
        plt.close()

# def make_graph_plot(case_dir, graph, filename, label = None):
#     """
#     Makes matplotlib.pyplot figure of the coronal plane of a networkx graph.

#     Parameters
#     ----------
#     case_dir : string or path-like object
#         Path to case directory. 
#     graph : networkx.Graph
#         Graph that we want to plot.
#     filename : string
#         Fine name of the final image. Make sure to add a valid extension (e.g. .png, .eps, etc)
#     label : string
#         Edge attribute to be printed at the center of each graph edge.

#     Returns
#     -------

#     """
#     # Generate plot of dense graph for quick visualization
#     _ = plt.figure(figsize = [5, 10])
#     ax = plt.gca()

#     # In order to place the nodes in the visualization of the graph in a sagittal view, 
#     # we use L and S coordinates (the view will be from the coronal plane, P axis)
#     node_pos_dict_p = {}
#     for n in graph.nodes():
#         node_pos_dict_p[n] = [-graph.nodes(data=True)[n]["pos"][0], graph.nodes(data=True)[n]["pos"][2]]

#     if label is not None:
#         edge_labels = nx.get_edge_attributes(graph, label)
#         nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
#         nx.draw_networkx_edge_labels(graph, node_pos_dict_p, edge_labels = edge_labels, ax=ax)
#     else:
#         nx.draw(graph, node_pos_dict_p, node_size=20, ax=ax)
        
#     # Add labels to the plot
#     plt.text(1, 0.5, 'L', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
#     plt.text(0, 0.5, 'R', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
#     plt.text(0.5, 1, 'S', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
#     plt.text(0.5, 0, 'I', horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)

#     plt.savefig(os.path.join(case_dir, filename))
#     plt.close()