#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

import os

from arterial.feature_extraction.global_features.utils import get_aortic_arch_type, get_bovine_arch, get_arsa

def perform_global_feature_extraction(local_graph):
    """
    Performs extraction of global features over the dense graph.

    These include:
    * Aortic arch type
    * Bovine arch presence
    * ARSA presence

    These will be stored in the .graph attribute (type dict) of the networkx.Graph.

    Parameters
    ----------
    local_graph : networkx.Graph
        Centerline graph containing with a node sampled at every `SAMPLE_NODE_EVERY_MM` milimiters,
        with ordered indices inidicating catheterization direction from different femoral and radial
        accesses.

    Returns
    -------
    local_graph : networkx.Graph
        Featurized centerline graph with global (graph) attributes.
    
    """
    # Extract aortic arch type
    local_graph.graph["aortic_arch_type"] = get_aortic_arch_type(local_graph)
    # Extract presence of bovine arch
    local_graph.graph["bovine_arch"] = get_bovine_arch(local_graph)
    # Extract presence of aberrant RSA
    local_graph.graph["arsa"] = get_arsa(local_graph)

    return local_graph