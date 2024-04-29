#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.feature_extraction.mapping.utils import supersegment_prediction, build_supersegments

def extract_arterial_mapping(local_graph):
    """
    Maps all possible catheter pathways and creates subgraphs for all
    of them. These are referred to as supersegments. A supersegment is 
    generated for each thrombectomy configuration, involving three different
    binary variables for a total of 8 different combinations:
        Access: femoral or radial.
        Laterality: right or left.
        Antero-posterior: anterior or posterior.

    First, all possible pathways for each access are extracted. Then, 
    every configuration of the case of interest is linked to a reference 
    configuration, which exists for all possible 8 configurations. A subgraph
    is created for each configuration.

    Parameters
    ----------
    local_graph : networkx.Graph
        Featurized and united centerline graph. 

    Returns
    -------
    supersegments : dict
        Dictionary containing all supersegments.
    
    """
    # Compute all catheter pathways (supersegments)
    predicted_configurations = supersegment_prediction(local_graph)
    # Associate each supersegment to a reference configuration and build subgraphs
    supersegments = build_supersegments(local_graph, predicted_configurations)

    return supersegments