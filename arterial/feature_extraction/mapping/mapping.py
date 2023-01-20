#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.feature_extraction.mapping.utils import supersegment_prediction, supersegment_built, select_configuration

def extract_arterial_mapping(case_dir, centerline_graph):
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

    If information for a past intervention is found in the form of a json file, 
    the used configuration is identified.

    Saves supersegments (graphs and image) as:

    >>> case_dir/supersegments/{configuration_name}.pickle
    >>> case_dir/supersegments.png

    And if a past intervention configuration for a patient exists, also saves it (graph and image) as:

    >>> case/dir/thrombectomy_configuration/supersegment.pickle
    >>> case/dir/thrombectomy_configuration/supersegment.png

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory.
    centerline_graph : networkx.Graph
        Featurized and united centerline graph. 

    Returns
    -------
    
    """
    # Compute all catheter pathways (supersegments)
    predicted_configurations = supersegment_prediction(centerline_graph)
    # Associate each supersegment to a reference configuration and build subgraphs
    supersegment_built(case_dir, centerline_graph, predicted_configurations)
    # If a patient_configuration.json file is found, identify supersegment
    # corresponding to used configuration
    if os.path.isfile(os.path.join(case_dir, "patient_configuration.json")):
        select_configuration(case_dir, centerline_graph)