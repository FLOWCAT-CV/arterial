#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.vessel_labelling.preprocessing.preprocessing import build_simple_centerline_graph
from arterial.vessel_labelling.inference import perform_inference

class VesselLabeller():
    """
    VesselLabeller class to perform vessel labelling over centerline model.   
    
    """
    def __init__(self, case_dir):
        """
        Initializes object of the VesselLabeller class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """

        self.case_dir = case_dir

    def preprocessing(self):
        """
        Runs preprocessing for vessel labelling. Generates a networkx graph where
        vessel segments are represented by edges, and nodes correspond to bifurcations.
        We call these "simple" graphs. Also performs featurization at a segment level.

        At the end of the preprocessing, a graph and an image should be generated:
        
        >>> case_dir/graph_simple.pickle
        >>> case_dir/graph_simple.png

        Acts as a wrapper for the arterial.vessel_labelling.preprocessing.preprocessing 
            make_simple_centerline_graph() function.

        Parameters
        ----------

        Returns
        -------

        """
        build_simple_centerline_graph(self.case_dir)

    def predict(self):
        """
        This method calls perform_inference to perform inference using a trained graph U-Net
        model over the simple graph generated in preprocessing. 
        
        At the end of the vessel type prediction, a graph and an image should be generated:
        
        >>> case_dir/graph_pred.pickle
        >>> case_dir/graph_pred.png
        
        Acts as a wrapper for the for the arterial.vessel_labelling.inference.
            perform_inference() function. 

        Parmeters
        ---------

        Returns
        -------

        """
        perform_inference(self.case_dir)