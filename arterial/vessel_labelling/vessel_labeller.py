#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.vessel_labelling.preprocessing.preprocessing import build_nx_graph_from_segments_array
from arterial.vessel_labelling.inference import perform_inference
from arterial.vessel_labelling.utils import make_graph_plot 
from arterial.io.load_and_save_operations import load_pickle, save_pickle, load_numpy

class VesselLabeller():
    """
    VesselLabeller class to perform vessel labelling over centerline model.   
    
    """
    def __init__(self, case_dir, mode="extracranial_vessels", centerline_segments_array_path=None):
        """
        Initializes object of the VesselLabeller class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory, where outputs will be saved. 
        mode : string, optional
            Mode of the vessel labeller. The default is "extracranial_vessels", it can also be "intracranial_vessels".
        centerline_segments_array_path : string or path-like object, optional
            Path to centerline segments array. If None, it will be set to case_dir/{mode}_centerline_segments_array.npy. 
            The default is None.

        Returns
        -------
        
        """

        self.case_dir = case_dir
        self.mode = mode

        if centerline_segments_array_path is None:
            self.centerline_segments_array_path = os.path.join(self.case_dir, f"{self.mode}_centerline_segments_array.npy")
        else:
            self.centerline_segments_array_path = centerline_segments_array_path
        self.centerline_segments_array = None

        self.segments_graph = None
        self.segments_graph_pred = None

        self.segments_graph_path = os.path.join(self.case_dir, f"{self.mode}_segments_graph.pickle")
        self.segments_graph_pred_path = os.path.join(self.case_dir, f"{self.mode}_segments_graph_pred.pickle")
        self.segments_graph_plot_path = os.path.join(self.case_dir, f"{self.mode}_segments_graph.png")
        self.segments_graph_pred_plot_path = os.path.join(self.case_dir, f"{self.mode}_segments_graph_pred.png")

    def build_segments_graph(self, save=True):
        """
        Runs preprocessing for vessel labelling. Generates a networkx graph where
        vessel segments are represented by edges, and nodes correspond to bifurcations.
        We call these "segments" graphs. Also performs featurization at a segment level.

        Parameters
        ----------
        save : bool, optional
            Whether to save the generated graph. The default is True.
        
        Returns
        -------

        """
        if not os.path.isfile(self.centerline_segments_array_path):
            raise FileNotFoundError(f"Centerline segments array not found in {self.centerline_segments_array_path}")
        
        self.load_centerline_segments_array()
        # Build segments graph from centerline segments array
        print("Building segments graph...")
        self.segments_graph = build_nx_graph_from_segments_array(self.centerline_segments_array)

        if save:
            save_pickle(self.segments_graph, self.segments_graph_path)
            make_graph_plot(self.segments_graph, label="cell_id", output_path=self.segments_graph_plot_path)

    def predict_vessel_types(self, ensemble=True, save=True):
        """
        This method calls perform_inference to perform inference using a trained GATv2 model
        model over the simple graph generated in preprocessing. Inference with an ensemble of models
        is also possible.

        Parmeters
        ---------
        ensemble : bool, optional
            Whether to use an ensemble of models. The default is True.
        save : bool, optional  
            Whether to save the predicted graph. The default is True.

        Returns
        -------

        """
        if self.segments_graph is None:
            if os.path.isfile(self.segments_graph_path):
                self.segments_graph = load_pickle(self.segments_graph_path)
            else:
                raise FileNotFoundError(f"Segments graph not found in {self.segments_graph_path}. Try running self.build_segments_graph() first.")
            
        print("Predicting vessel types...")
        self.segments_graph_pred = perform_inference(self.segments_graph, self.mode, ensemble)

        if save:
            save_pickle(self.segments_graph_path, self.segments_graph_pred_path)
            make_graph_plot(self.segments_graph_pred, label="cell_id", output_path=self.segments_graph_pred_plot_path)

    def load_centerline_segments_array(self):
        """
        Loads centerline segments array from file.

        Parameters
        ----------

        Returns
        -------

        """
        if not os.path.isfile(self.centerline_segments_array_path):
            raise FileNotFoundError(f"Centerline segments array not found in {self.centerline_segments_array_path}")
        
        self.centerline_segments_array = load_numpy(self.centerline_segments_array_path)

    def load_segments_graph(self):
        """
        Loads segments graph from file.

        Parameters
        ----------

        Returns
        -------

        """
        if not os.path.isfile(self.segments_graph_path):
            raise FileNotFoundError(f"Segments graph not found in {self.segments_graph_path}")

        self.segments_graph = load_pickle(self.segments_graph_path)

    def set_centerline_segments_array_path(self, path):
        """
        Sets centerline segments array path.

        Parameters
        ----------
        path : string or path-like object
            Path to centerline segments array. 

        Returns
        -------

        """
        self.centerline_segments_array_path = path
    
    def set_segments_graph_path(self, path):
        """
        Sets segments graph path.

        Parameters
        ----------
        path : string or path-like object
            Path to segments graph. 

        Returns
        -------

        """
        self.segments_graph_path = path
    
    def set_segments_graph_pred_path(self, path):
        """
        Sets predicted graph path.
        
        Parameters
        ----------
        path : string or path-like object
            Path to predicted graph.

        Returns
        -------

        """
        self.segments_graph_pred_path = path
