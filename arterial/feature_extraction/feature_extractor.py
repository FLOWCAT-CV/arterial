#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.feature_extraction.graph_builder import build_centerline_graph
from arterial.feature_extraction.local_features.feature_extraction import perform_local_feature_extraction
from arterial.feature_extraction.segment_features.feature_extraction import perform_segment_feature_extraction
from arterial.feature_extraction.global_features.feature_extraction import perform_global_feature_extraction
from arterial.feature_extraction.mapping.mapping import extract_arterial_mapping

class FeatureExtractor():
    """
    FeatureExtractor class to perform feature extraction at multiple scales
    and mapping of catheter pathways.

    """
    def __init__(self, case_dir):
        """
        Initializes object of the FeatureExtractor class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """ 
        self.case_dir = case_dir

    def build_graph(self):
        """
        Builds dense centerline graph from case_dir/centerline_segments_array.npy and 
        graph_pred.pickle.

        Saves graph and image of unified dense graph as:

        >>> case_dir/graph.pickle
        >>> case_dir/graph.png

        Acts as a wrapper for the arterial.feature_extraction.graph_builder.build_graph() 
        function.

        Parameters
        ----------

        Returns
        -------

        """
        self.centerline_graph = build_centerline_graph(self.case_dir)
    
    def extract_local_features(self):
        """
        Extracts local node features from centerline_graph.

        This function overwrites the existing dense centerline graph:

        >>> case_dir/graph.pickle

        Acts as a wrapper for the arterial.feature_extraction.local_features.feature_extraction.
            perform_local_feature_extraction() function.

        Parameters
        ----------

        Returns
        -------

        """
        self.centerline_graph = perform_local_feature_extraction(self.case_dir, self.centerline_graph)

    def extract_segment_features(self):
        """
        Extracts segment features from centerline_graph.

        This function overwrites the existing centerline graphs:

        >>> case_dir/graph.pickle
        >>> case_dir/graph_simple.pickle

        Creates the following image:

        >>> case_dir/single_segments.png

        Acts as a wrapper for the arterial.feature_extraction.segment_features.feature_extraction.
            perform_segment_feature_extraction() function.

        Parameters
        ----------

        Returns
        -------

        """
        self.centerline_graph = perform_segment_feature_extraction(self.case_dir, self.centerline_graph)

    def extract_global_features(self):
        """
        Extracts global features from centerline_graph.

        This function overwrites the existing dense centerline graph:

        >>> case_dir/graph.pickle

        Acts as a wrapper for the arterial.feature_extraction.global_features.feature_extraction.
            perform_global_feature_extraction() function.

        Parameters
        ----------

        Returns
        -------

        """
        self.centerline_graph = perform_global_feature_extraction(self.case_dir, self.centerline_graph)

    def map_catheter_pathways(self):
        """
        Maps all catheter pathways and saves them in the form of supersegments (networkx.Graph objects).

        This function saves supersegments (graphs and image) as:

        >>> case_dir/supersegments/{configuration_name}.pickle
        >>> case_dir/supersegments.png

        And if a past intervention configuration for a patient exists, also saves it (graph and image) as:

        >>> case/dir/thrombectomy_configuration/supersegment.pickle
        >>> case/dir/thrombectomy_configuration/supersegment.png

        Acts as a wrapper for the arterial.feature_extraction.mapping.mapping.extract_arterial_mapping()
        function.
        
        Parameters
        ----------

        Returns
        -------

        """
        extract_arterial_mapping(self.case_dir, self.centerline_graph)