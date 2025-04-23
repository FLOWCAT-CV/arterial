#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.access_prediction.preprocessing.preprocessing import preprocess_supersegment
from arterial.access_prediction.inference import perform_inference
from arterial.access_prediction.utils import make_combined_plot, get_lpi_corner_coordinates, nx_graph_to_vtk_polydata, nx_graph_to_point_dict

from arterial.io.load_and_save_operations import load_pickle, save_json, save_pickle, load_nifti, save_vtkpolydata
from arterial.feature_extraction.utils import make_graph_plot

class AccessPredictor():
    """
    AccessPredictor class to perform preprocessing and inference from dense
    supersegments to access prediction for endovascular treatment.

    """
    def __init__(self,
                 case_dir,
                 cta_nifti_path=None,
                 supersegments_dir_path=None,
                 access=["femoral"],
                 side=["left", "right"]
                ):
        """
        Initializes object of the AccessPredictor class.

        Parameters
        ----------
        case_dir : str
            Path to the directory containing the case data.
        cta_nifti_path : str
            Path to the CTA nifti file. Default is None, and it will be set to `case_dir/cta.nii.gz`.
        access : list
            List of strings with the accesses to be considered. Default is ["femoral"].
        side : list
            List of strings with the sides to be considered. Default is ["left", "right"].

        """
        self.case_dir = case_dir
        if cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        else:
            self.cta_nifti_path = cta_nifti_path
        if supersegments_dir_path is None:
            self.supersegments_dir_path = os.path.join(self.case_dir, "extracranial_vessels", "supersegments")
        else:
            self.supersegments_dir_path = supersegments_dir_path
        self.access = access
        self.side = side

        self.cta_nifti = None

        self.access_prediction_dir_path = os.path.join(self.case_dir, "extracranial_vessels", "access_prediction")

        self.raw_supersegment_dict = {}
        self.preprocessed_supersegment_dict = {}

        self.lpi_corner_coordinates = None

        self.predictions_dict = {}
        self.attention_maps_dict = {}
        self.attention_maps_vtk_dict = {}
        self.attention_maps_point_dict = {}

    def preprocess_supersegments(self, save=True):
        """
        Generates the pickle file to be used for access prediction from supersegments
        extracted with Arterial. Will store raw supersegments, as well as store preprocessed
        supersegments with global features, segment graph and dense graph.

        Preprocessing includes:
        - Reading the raw supersegment files
        - Process the raw supersegment file to get a 1D segment (a nx Graph)
        - Process the 1D segment to get a set of global features (i.e., a set of segment features extracted from the whole supersegment)
        - Process the 1D segment to get a segment graph (a nx Graph), where each node corresponds to a vessel segment
        - Process the 1D segment to get a dense graph (a nx Graph), where each node corresponds to a point of the 1D segment
        - Join all objects into a single dictionary and save it as a pickle file, as well as saving the global features, segment graph and dense graph separately
        - Create a combined plot of the segment graph and dense graph and save it as a png file

        The following files are saved:
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "preprocessed_supersegment_dict.pickle")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "global_features.json")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "segment_supersegment.pickle")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "dense_supersegment.pickle")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "combined_plot.png")

        Parameters
        ----------
        save : bool
            Whether to save the preprocessed supersegments (separately). Default is True.

        Returns
        -------

        """
        for access in self.access:
            for side in self.side:
                print(self.supersegments_dir_path, f"{access} + {side} + anterior.pickle")
                assert os.path.isfile(os.path.join(self.supersegments_dir_path, f"{access} + {side} + anterior.pickle")), f"Supersegments ({access}, {side}) not found"
                print(f"\nPreprocessing {access} {side} supersegment...")
                self.raw_supersegment_dict[(access, side)] = load_pickle(os.path.join(self.supersegments_dir_path, f"{access} + {side} + anterior.pickle"))                  
                self.preprocessed_supersegment_dict[(access, side)] = preprocess_supersegment(self.raw_supersegment_dict[(access, side)], access, side)
                if save:
                    assert len(self.preprocessed_supersegment_dict[(access, side)]["segment_graph"]) > 0, "Segment graph not built"
                    assert self.preprocessed_supersegment_dict[(access, side)]["dense_graph"] is not None, "Dense graph not built"
                    
                    os.makedirs(os.path.join(self.access_prediction_dir_path, f"{access}_{side}"), exist_ok=True)
                    
                    save_json(self.preprocessed_supersegment_dict[(access, side)]["global_features"], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "global_features.json"))
                    save_pickle(self.preprocessed_supersegment_dict[(access, side)]["segment_graph"], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "segment_supersegment.pickle"))
                    save_pickle(self.preprocessed_supersegment_dict[(access, side)]["dense_graph"], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "dense_supersegment.pickle"))
                    save_pickle(self.preprocessed_supersegment_dict[(access, side)], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "preprocessed_supersegment_dict.pickle"))
                    make_combined_plot(side, self.preprocessed_supersegment_dict[(access, side)]["segment_graph"], self.preprocessed_supersegment_dict[(access, side)]["dense_graph"], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "combined_plot.png"))

    def predict_accessibility(self, return_attention_map=True, save=True):
        """
        Calls perform_inference from access_prediction/inference.py to predict access for each supersegment.
        Predictoin is composed by mean and std of the prediction logits across folds. If return_attention_map is True,
        attention maps are also computed as nx graphs, vtkPolyData objects and simple dictionaries with the points and 
        attention weights.

        The following files are saved:
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "access_prediction.json")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map.pickle")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map.png")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map.vtk")
        >>> os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map_points.json")

        Parameters
        ----------
        return_attention_map : bool
            Whether to return attention maps. Default is True.
        save : bool
            Whether to save the predictions. Default is True.

        Returns
        -------
        
        """
        if self.lpi_corner_coordinates is None:
            self.compute_lpi_corner_coordinates()
        for access in self.access:
            for side in self.side:
                if (access, side) not in self.preprocessed_supersegment_dict:
                    self.preprocess_supersegments(save=save)
                print(f"\nPerforming inference for {access} {side} supersegment...")
                self.predictions_dict[(access, side)] = {}
                out = perform_inference(self.preprocessed_supersegment_dict[(access, side)], self.lpi_corner_coordinates, return_attention_map=return_attention_map)
                self.predictions_dict[(access, side)]["mean"] = out[0]
                self.predictions_dict[(access, side)]["std"] = out[1]
                if return_attention_map:
                    self.attention_maps_dict[(access, side)] = out[2]
                    self.attention_maps_vtk_dict[(access, side)] = nx_graph_to_vtk_polydata(self.attention_maps_dict[(access, side)])
                    self.attention_maps_point_dict[(access, side)] = nx_graph_to_point_dict(self.attention_maps_dict[(access, side)])
                if save:
                    save_json(self.predictions_dict[(access, side)], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "access_prediction.json"))
                    if return_attention_map:
                        save_pickle(self.attention_maps_dict[(access, side)], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map.pickle"))
                        make_graph_plot(self.attention_maps_dict[(access, side)], feature="attention_weight", output_path=os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map.png"))
                        save_vtkpolydata(self.attention_maps_vtk_dict[(access, side)], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map.vtk"))
                        save_json(self.attention_maps_point_dict[(access, side)], os.path.join(self.access_prediction_dir_path, f"{access}_{side}", "attention_map_points.json"))

    def compute_lpi_corner_coordinates(self):
        """
        Computes LPI corner coordinate from the CTA nifti to adjust positions 
        of the attention maps, for these to be coherent with 3D models directly derived
        from the segmentation nifti.

        Parameters
        ----------

        Returns
        -------

        """
        if self.cta_nifti is None:
            self.load_cta_nifti()
        self.lpi_corner_coordinates = get_lpi_corner_coordinates(self.cta_nifti)

    def load_cta_nifti(self):
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        
        self.cta_nifti = load_nifti(self.cta_nifti_path)