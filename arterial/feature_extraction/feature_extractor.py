#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
from arterial.feature_extraction.graph_builder import build_local_graph
from arterial.feature_extraction.utils import make_graph_plot
from arterial.feature_extraction.local_features.feature_extraction import perform_local_feature_extraction
from arterial.feature_extraction.segment_features.feature_extraction import perform_segment_feature_extraction
from arterial.feature_extraction.segment_features.utils import plot_single_segments
from arterial.feature_extraction.global_features.feature_extraction import perform_global_feature_extraction
from arterial.feature_extraction.mapping.mapping import extract_arterial_mapping
from arterial.feature_extraction.mapping.utils import make_supersegment_plots
from arterial.io.load_and_save_operations import *

class FeatureExtractor():
    """
    FeatureExtractor class to perform feature extraction at multiple scales
    and mapping of catheter pathways.

    """
    def __init__(self, 
                 case_dir, 
                 mode="extracranial_vessels",
                 sampling_distance_mm=2,
                 cta_nifti_path=None, 
                 centerline_segments_array_path=None, 
                 branch_model_path=None,
                 segments_graph_pred_path=None
                 ):
        """
        Initializes object of the FeatureExtractor class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 
        mode : string, optional
            Mode of the vessel labeller. The default is "extracranial_vessels", it can also be "intracranial_vessels".
        sampling_distance_mm : int or float, optional
            Sampling distance in mm. The default is 2.
        cta_nifti_path : string or path-like object, default = None
            Path to the original CTA nifti file. If not provided, the CTA should be in nifti format, the
            name convention used should be case_dir/cta.nii.gz it should be located in the self.case_dir directory.
        centerline_segments_array_path : string or path-like object, optional
            Path to centerline segments array. If None, it will be set to case_dir/{mode}_centerline_segments_array.npy. 
            The default is None.
        branch_model_path : string or path-like object, optional
            Path to the trained model for branch prediction. If None, it will be set to case_dir/{mode}_branch_model.vtk. The default is None.
        segments_graph_pred_path : string or path-like object, optional
            Path to the predicted segments graph. If None, it will be set to case_dir/{mode}_graph_simple_pred.pickle. The default is None.
        
        """ 
        assert case_dir is not None, "case_dir should be provided as the directory where all results will be saved."
        assert mode in ["extracranial_vessels", "intracranial_vessels"], "mode should be either 'extracranial_vessels' or 'intracranial_vessels'."

        self.case_dir = case_dir
        self.mode = mode
        self.sampling_distance_mm = sampling_distance_mm
        if cta_nifti_path is None:
            self.cta_nifti_path = os.path.join(self.case_dir, "cta.nii.gz")
        else:
            self.cta_nifti_path = cta_nifti_path
        if centerline_segments_array_path is None:
            self.centerline_segments_array_path = os.path.join(self.case_dir, self.mode, "centerline_segments_array.npy")
        else:
            self.centerline_segments_array_path = centerline_segments_array_path
        if branch_model_path is None:
            self.branch_model_path = os.path.join(self.case_dir, self.mode, "branch_model.vtk")
        else:
            self.branch_model_path = branch_model_path
        if segments_graph_pred_path is None:
            self.segments_graph_pred_path = os.path.join(self.case_dir, self.mode, "segments_graph_pred.pickle")
        else:
            self.segments_graph_pred_path = segments_graph_pred_path

        self.cta_nifti = None
        self.cta_array = None
        self.cta_affine = None
        self.centerline_segments_array = None
        self.branch_model = None
        self.segments_graph = None

        self.local_graph = None
        self.local_graph_path = os.path.join(self.case_dir, self.mode, "local_graph.pickle")
        self.local_graph_plot_path = os.path.join(self.case_dir, self.mode, "local_graph.png")

        self.segment_features = None
        self.segments_vessel_type_dict = None
        self.single_segments_dir_path = os.path.join(self.case_dir, self.mode, "single_segments")
        self.single_segments_plot_path = os.path.join(self.case_dir, self.mode, "single_segments.png")

        self.supersegments = None
        self.supersegments_dir_path = os.path.join(self.case_dir, self.mode, "supersegments")
        self.supersegments_plot_path = os.path.join(self.case_dir, self.mode, "supersegments.png")

    def build_local_graph(self, save=True):
        """
        Builds dense centerline graph from case_dir/centerline_segments_array.npy and 
        graph_pred.pickle.

        Parameters
        ----------
        save : bool, optional
            Whether to save the generated graph. The default is True.

        Returns
        -------

        """
        if save: os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        if self.centerline_segments_array is None:
            self.load_centerline_segments_array()
        if self.segments_graph is None:
            self.load_segments_graph_pred()
        self.local_graph = build_local_graph(self.centerline_segments_array, self.segments_graph, self.sampling_distance_mm)

        if save:
            save_pickle(self.local_graph, self.local_graph_path)
            make_graph_plot(self.local_graph, output_path=self.local_graph_plot_path)
    
    def extract_local_features(self, save=True):
        """
        Extracts local node features from centerline_graph.CE

        Parameters
        ----------
        save : bool, optional
            Whether to save the generated graph. The default is True.

        Returns
        -------

        """
        if self.cta_array is None or self.cta_affine is None:
            self.load_cta_nifti()
        if self.branch_model is None:
            self.load_branch_model()
        self.local_graph = perform_local_feature_extraction(self.local_graph, self.cta_array, self.cta_affine, self.branch_model)
        
        if save:
            save_pickle(self.local_graph, self.local_graph_path)

    def extract_segment_features(self, save=True):
        """
        Extracts segment features from centerline_graph. Segment features are stored both in the global 
        features of the self.local_graph as well as in the edges of the segment_graph If save=True, this function overwrites the
        existing centerline graphs:

        Parameters
        ----------

        Returns
        -------

        """
        if save: os.makedirs(self.single_segments_dir_path, exist_ok=True)
        if self.segments_graph is None:
            self.load_segments_graph_pred()
        self.local_graph, self.segments_graph, self.segments_vessel_type_dict = perform_segment_feature_extraction(self.local_graph, self.segments_graph)
        self.segment_features = self.local_graph.graph["segment_features"]

        if save:
            save_pickle(self.local_graph, self.local_graph_path)
            save_pickle(self.segments_graph, self.segments_graph_pred_path)
            for vessel_type in self.segments_vessel_type_dict.keys():
                if self.segments_vessel_type_dict[vessel_type] is not None:
                    save_pickle(self.segments_vessel_type_dict[vessel_type], os.path.join(self.single_segments_dir_path, "{}.pickle".format(vessel_type)))
            plot_single_segments(self.local_graph, self.segments_vessel_type_dict, output_path=self.single_segments_plot_path)

    def extract_global_features(self, save=True):
        """
        Extracts global features from centerline_graph. Stores them in the graph attribute of the networkx.Graph.

        Parameters
        ----------
        save : bool, optional
            Whether to save the generated graph. The default is True.

        Returns
        -------

        """
        self.local_graph = perform_global_feature_extraction(self.local_graph)

        if save:
            save_pickle(self.local_graph, self.local_graph_path)

    def extract_supersegments(self, save=True):
        """
        Maps all catheter pathways corresponding to the all configurations from combining 
        access (femoral, radial), laterality (right, left) or antero-posterior (anterior, posterior)
        and saves them in the form of supersegments (networkx.Graph objects) if save is True.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the generated supersegments. The default is True.

        Returns
        -------

        """
        if save: os.makedirs(self.supersegments_dir_path, exist_ok=True)
        
        self.supersegments = extract_arterial_mapping(self.local_graph)

        if save:
            for config, supersegment in self.supersegments.items():
                save_pickle(supersegment, os.path.join(self.supersegments_dir_path, f"{config[0]} + {config[1]} + {config[2]}.pickle"))
            make_supersegment_plots(self.supersegments, output_path=self.supersegments_plot_path)

    def is_local_featurized(self):
        if "features femoral" in self.local_graph.nodes[0].keys():
            return True
        else:
            return False
    
    def is_segment_featurized(self):
        if "segment_features" in self.local_graph.graph.keys():
            return True
        else:
            return False
    
    def is_global_featurized(self):
        if "aortic_arch_type" in self.local_graph.graph.keys():
            return True
        else:
            return False

    def load_cta_nifti(self):
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        
        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine

    def load_centerline_segments_array(self):
        if not os.path.isfile(self.centerline_segments_array_path):
            raise FileNotFoundError(f"Centerline segments array not found in {self.centerline_segments_array_path}")
        
        self.centerline_segments_array = load_numpy(self.centerline_segments_array_path)

    def load_branch_model(self):
        if not os.path.isfile(self.branch_model_path):
            raise FileNotFoundError(f"Branch model not found in {self.branch_model_path}")

        self.branch_model = load_vtkpolydata(self.branch_model_path)

    def load_segments_graph_pred(self):
        if not os.path.isfile(self.segments_graph_pred_path):
            raise FileNotFoundError(f"Segments graph not found in {self.segments_graph_pred_path}")
        self.segments_graph = load_pickle(self.segments_graph_pred_path)

    def set_case_dir(self, case_dir):
        if not isinstance(case_dir, str):
            raise ValueError("case_dir should be a string.")
        self.case_dir = case_dir
    
    def set_mode(self, mode):
        if mode not in ["extracranial_vessels", "intracranial_vessels"]:
            raise ValueError("mode should be either 'extracranial_vessels' or 'intracranial_vessels'.")
        self.mode = mode

    def set_sampling_distance_mm(self, sampling_distance_mm):
        if not isinstance(sampling_distance_mm, (int, float)):
            raise ValueError("sampling_distance_mm should be an integer or a float.")
        self.sampling_distance_mm = sampling_distance_mm

    def set_cta_nifti_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.cta_nifti_path = path

    def set_centerline_segments_array_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.centerline_segments_array_path = path

    def set_branch_model_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.branch_model_path = path

    def set_segments_graph_pred_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segments_graph_pred_path = path

    def set_local_graph_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.local_graph_path = path

    def set_local_graph_plot_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.local_graph_plot_path = path

    def set_single_segments_dir_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.single_segments_dir_path = path    

    def set_single_segments_plot_path(self, path):  
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.single_segments_plot_path = path   

    def set_supersegments_dir_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.supersegments_dir_path = path
        
    def set_supersegments_plot_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.supersegments_plot_path = path