#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

from arterial.vessel_labelling.preprocessing.preprocessing import build_nx_graph_from_segments_array
from arterial.vessel_labelling.inference import perform_inference
from arterial.vessel_labelling.utils import make_graph_plot
from arterial.io.load_and_save_operations import load_pickle, save_pickle, load_numpy

class VesselLabeller():
    """
    VesselLabeller class to perform vessel labelling over centerline model.   
    
    """
    def __init__(self, 
                 case_dir, 
                 mode="extracranial_vessels", 
                 centerline_segments_array_path=None
                 ):
        """
        Initializes object of the VesselLabeller class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory, where outputs will be saved. 
        mode : string, optional
            Mode of the vessel labeller. The default is "extracranial_vessels", it can also be "intracranial_vessels".
        centerline_segments_array_path : string or path-like object, optional
            Path to centerline segments array. If None, it will be set to `{mode}_centerline_segments_array.npy`, and it
            should be located in the case_dir. The default is None.
        
        """
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."

        self.case_dir = case_dir
        self.mode = mode

        if centerline_segments_array_path is None:
            self.centerline_segments_array_path = os.path.join(self.case_dir, self.mode, "centerline_segments_array.npy")
        else:
            self.centerline_segments_array_path = centerline_segments_array_path
        self.centerline_segments_array = None

        self.segments_graph = None
        self.segments_graph_pred = None

        self.segments_graph_path = os.path.join(self.case_dir, self.mode, "segments_graph.pickle")
        self.segments_graph_pred_path = os.path.join(self.case_dir, self.mode, "segments_graph_pred.pickle")
        self.segments_graph_plot_path = os.path.join(self.case_dir, self.mode, "segments_graph.png")
        self.segments_graph_pred_plot_path = os.path.join(self.case_dir, self.mode, "segments_graph_pred.png")

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
        if save: os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        self._load_centerline_segments_array()
        # Build segments graph from centerline segments array
        print("Building segments graph...")
        self.segments_graph = build_nx_graph_from_segments_array(self.centerline_segments_array)

        if save:
            print(f"Saving segments graph to {self.segments_graph_path}")
            save_pickle(self.segments_graph, self.segments_graph_path)
            print(f"Saving segments graph plot to {self.segments_graph_plot_path}")
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
                self._load_segments_graph()
            else:
                self.build_segments_graph()

        # Perform sanity check on the segments graph
        self.segments_graph_sanity_check()
            
        print("Predicting vessel types...")
        self.segments_graph_pred = perform_inference(self.segments_graph, self.mode, ensemble)

        if save:
            print(f"Saving predicted segments graph to {self.segments_graph_pred_path}")
            save_pickle(self.segments_graph_pred, self.segments_graph_pred_path)
            print(f"Saving predicted segments graph plot to {self.segments_graph_pred_plot_path}")
            make_graph_plot(self.segments_graph_pred, label="vessel_type_name", output_path=self.segments_graph_pred_plot_path)

    def segments_graph_sanity_check(self):
        """
        Performs sanity check on segments graph. Raises an error
        if the graph has less than 2 edges.

        Parameters
        ----------

        Returns
        -------

        """
        if self.segments_graph is None:
            raise ValueError("Segments graph is not loaded or created.")
        if len(self.segments_graph.edges) < 2:
            raise ValueError("Segment graph has less than 2 edges. This is a critical error, processing is interrupted.")

    def _load_centerline_segments_array(self):
        if not os.path.isfile(self.centerline_segments_array_path):
            raise FileNotFoundError(f"Centerline segments array not found in {self.centerline_segments_array_path}")
        
        self.centerline_segments_array = load_numpy(self.centerline_segments_array_path)

    def _load_segments_graph(self):
        if not os.path.isfile(self.segments_graph_path):
            raise FileNotFoundError(f"Segments graph not found in {self.segments_graph_path}")
        self.segments_graph = load_pickle(self.segments_graph_path)

    def _load_segments_graph_pred(self):
        if not os.path.isfile(self.segments_graph_pred_path):
            raise FileNotFoundError(f"Predicted segments graph not found in {self.segments_graph_pred_path}")
        self.segments_graph_pred = load_pickle(self.segments_graph_pred_path)

    def _set_case_dir(self, case_dir):
        if not isinstance(case_dir, str):
            raise ValueError("case_dir should be a string.")
        self.case_dir = case_dir

    def _set_mode(self, mode):
        if mode not in ["extracranial_vessels", "intracranial_vessels"]:
            raise ValueError("mode should be either 'extracranial_vessels' or 'intracranial_vessels'.")
        self.mode = mode

    def _set_centerline_segments_array_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.centerline_segments_array_path = path
    
    def _set_segments_graph_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segments_graph_path = path
    
    def _set_segments_graph_pred_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segments_graph_pred_path = path

    def _set_segments_graph_plot_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segments_graph_plot_path = path

    def _set_segments_graph_pred_plot_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segments_graph_pred_plot_path = path

    def make_segments_graph_plot(self):
        if self.segments_graph is None:
            if os.path.isfile(self.segments_graph_path):
                self.segments_graph = load_pickle(self.segments_graph_path)
            else:
                self.build_segments_graph()
        print(f"Saving segments graph plot to {self.segments_graph_plot_path}")
        make_graph_plot(self.segments_graph, label="cell_id", output_path=self.segments_graph_plot_path)
    
    def make_segments_graph_pred_plot(self):
        if self.segments_graph_pred is None:
            if os.path.isfile(self.segments_graph_pred_path):
                self._load_segments_graph_pred()
            else:
                self.predict_vessel_types()
        print(f"Saving segments predicted graph plot to {self.segments_graph_pred_plot_path}")
        make_graph_plot(self.segments_graph_pred, label="vessel_type_name", output_path=self.segments_graph_pred_plot_path)
