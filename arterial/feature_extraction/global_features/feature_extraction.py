#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

import pickle

from arterial.feature_extraction.global_features.utils import get_aortic_arch_type, get_bovine_arch, get_arsa
from arterial.io.load_and_save_operations import save_pickle

def perform_global_feature_extraction(case_dir, centerline_graph):
    """
    Performs extraction of global features over the dense graph.

    These include:
    * Aortic arch type
    * Bovine arch presence
    * ARSA presence

    These will be stored in the .graph attribute (type dict) of the networkx.Graph.

    This function overwrites the existing dense centerline graph:

    >>> case_dir/graph.pickle

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    centerline_graph : networkx.Graph
        Centerline graph containing with a node sampled at every `SAMPLE_NODE_EVERY_MM` milimiters,
        with ordered indices inidicating catheterization direction from different femoral and radial
        accesses.

    Returns
    -------
    centerline_graph : networkx.Graph
        Featurized centerline graph with global (graph) attributes.
    
    """
    # Extract aortic arch type
    centerline_graph.graph["aortic_arch_type"] = get_aortic_arch_type(centerline_graph)
    # Extract presence of bovine arch
    centerline_graph.graph["bovine_arch"] = get_bovine_arch(centerline_graph)
    # Extract presence of aberrant RSA
    centerline_graph.graph["arsa"] = get_arsa(centerline_graph)
    # Overwrite centerline_graph
    save_pickle(centerline_graph, os.path.join(case_dir, "dense_graph.pickle"))

    return centerline_graph