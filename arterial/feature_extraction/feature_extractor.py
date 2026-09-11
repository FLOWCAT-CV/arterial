#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from arterial.feature_extraction.graph_builder import build_local_graph, build_individual_centerline_graph_from_vtkpolydata
from arterial.feature_extraction.utils import resample_centerline_segments_array, make_graph_plot, centerline_sanity_check
from arterial.feature_extraction.local_features.feature_extraction import perform_local_feature_extraction, perform_local_feature_extraction_individual_centerline
from arterial.feature_extraction.segment_features.feature_extraction import perform_segment_feature_extraction
from arterial.feature_extraction.segment_features.utils import plot_single_segments, extract_segment_features
from arterial.feature_extraction.global_features.feature_extraction import perform_global_feature_extraction
from arterial.feature_extraction.mapping.mapping import extract_arterial_mapping
from arterial.feature_extraction.mapping.utils import make_supersegment_plots
from arterial.feature_extraction.vtk_centerline_geometry.feature_extraction import perform_radius_extraction, perform_curvature_extraction, perform_curve_id_extraction
from arterial.feature_extraction.vtk_centerline_geometry.carotid_analysis import (
    perform_carotid_analysis, perform_intracranial_transition_detection,
)
from arterial.feature_extraction.vtk_centerline_geometry.utils import pickle_to_vtk
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

        # For intracranial vessel analysis
        self.individual_centerlines_dir_path = os.path.join(self.case_dir, self.mode, "individual_centerlines")
        self.individual_centerline_graph = None

        # Cached cranium distance transform for the intracranial-transition detection
        # in `add_carotid_analysis` / `add_intracranial_transition`. Lazily populated.
        self._cranium_distance_transform = None
        self._cranium_distance_transform_path = os.path.join(self.case_dir, self.mode, "cranium_distance_transform.npy")

    def build_local_graph(self, resample=True, save=True):
        """
        Builds dense centerline graph from case_dir/centerline_segments_array.npy and 
        graph_pred.pickle.

        Parameters
        ----------
        resample : bool, optional
            Whether to resample the centerline segments array. The default is True.
        save : bool, optional
            Whether to save the generated graph. The default is True.

        Returns
        -------

        """
        if save: os.makedirs(os.path.join(self.case_dir, self.mode), exist_ok=True)
        if self.centerline_segments_array is None:
            self._load_centerline_segments_array()
        if resample:
            from time import time
            start_time = time()
            print(f"Resampling centerline segments array to {self.sampling_distance_mm} mm")
            self.centerline_segments_array = resample_centerline_segments_array(self.centerline_segments_array, self.sampling_distance_mm)
            print(f"Centerline segments array resampled ({time() - start_time:.2f} seconds)")
        if self.segments_graph is None:
            self._load_segments_graph_pred()
        self.local_graph = build_local_graph(self.centerline_segments_array, self.segments_graph, self.sampling_distance_mm)

        if save:
            print(f"Saving local graph to {self.local_graph_path}")
            save_pickle(self.local_graph, self.local_graph_path)
            print(f"Saving local graph plot to {self.local_graph_plot_path}")
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
            self._load_cta_nifti_from_file()
        if self.branch_model is None:
            self._load_branch_model()
        self.local_graph = perform_local_feature_extraction(self.local_graph, self.cta_array, self.cta_affine, self.branch_model)
        
        if save:
            print(f"Saving local graph to {self.local_graph_path}")
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
            self._load_segments_graph_pred()
        self.local_graph, self.segments_graph, self.segments_vessel_type_dict = perform_segment_feature_extraction(self.local_graph, self.segments_graph)
        self.segment_features = self.local_graph.graph["segment_features"]

        if save:
            print(f"Saving local graph to {self.local_graph_path}")
            save_pickle(self.local_graph, self.local_graph_path)
            print(f"Saving segments graph to {self.segments_graph_pred_path}")
            save_pickle(self.segments_graph, self.segments_graph_pred_path)
            for vessel_type in self.segments_vessel_type_dict.keys():
                if self.segments_vessel_type_dict[vessel_type] is not None:
                    print(f"Saving segments vessel type to {os.path.join(self.single_segments_dir_path, '{}.pickle'.format(vessel_type))}")
                    save_pickle(self.segments_vessel_type_dict[vessel_type], os.path.join(self.single_segments_dir_path, '{}.pickle'.format(vessel_type)))
            print(f"Saving single segments plot to {self.single_segments_plot_path}")
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
            print(f"Saving local graph to {self.local_graph_path}")
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
                print(f"Saving supersegment to {os.path.join(self.supersegments_dir_path, f'{config[0]} + {config[1]} + {config[2]}.pickle')}")
                save_pickle(supersegment, os.path.join(self.supersegments_dir_path, f"{config[0]} + {config[1]} + {config[2]}.pickle"))
            print(f"Saving supersegments plot to {self.supersegments_plot_path}")
            make_supersegment_plots(self.supersegments, local_graph=self.local_graph, output_path=self.supersegments_plot_path)

    def build_and_featurize_individual_centerline_graph(self, centerline_model, radius_array_name="MaximumInscribedSphereRadius", centerline_id=None, save=True):
        """
        Builds and featurizes an individual centerline graph from a vtkpolydata. 

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Centerline model to build the individual centerline graph from.
        radius_array_name : str, optional
            Name of the radius array in the centerline_model. The default is "MaximumInscribedSphereRadius".
        centerline_id : int, optional
            ID of the centerline. This will be used to name the saved graph and plot. The default is None.
        save : bool, optional
            Whether to save the generated graph. The default is True.

        Returns
        -------

        """
        if save: os.makedirs(self.individual_centerlines_dir_path, exist_ok=True)
        if self.cta_array is None or self.cta_affine is None:
            self._load_cta_nifti_from_file()
        if not centerline_sanity_check(centerline_model, self.cta_array, self.cta_affine):
            raise ValueError("Centerline model is not valid. It is not within the image volume.")
        else:
            self.individual_centerline_graph = build_individual_centerline_graph_from_vtkpolydata(centerline_model, self.cta_affine, self.cta_array.shape, radius_array_name=radius_array_name, centerline_id=centerline_id, sampling_distance_mm=self.sampling_distance_mm)
            self.individual_centerline_graph = perform_local_feature_extraction_individual_centerline(self.individual_centerline_graph, self.cta_array, self.cta_affine)
            self.individual_centerline_graph = extract_segment_features(self.individual_centerline_graph, use_blanking=False)

            if save:
                print(f"Saving individual centerline graph to {os.path.join(self.individual_centerlines_dir_path, f'individual_centerline_{centerline_id}.pickle')}")
                save_pickle(self.individual_centerline_graph, os.path.join(self.individual_centerlines_dir_path, f"individual_centerline_{centerline_id}.pickle"))
                print(f"Saving individual centerline graph plot to {os.path.join(self.individual_centerlines_dir_path, f'individual_centerline_{centerline_id}.png')}")
                make_graph_plot(self.individual_centerline_graph, output_path=os.path.join(self.individual_centerlines_dir_path, f"individual_centerline_{centerline_id}.png"))

    def single_segment_pickle_to_vtk(self, graph, use_branch_model_for_blanking=True, mis_array_name="MaximumInscribedSphereRadius", compute_frenet=True, save_path=None):
        """
        Converts a single-segment networkx graph (e.g. one of the pickles in the
        case's `single_segments/` directory) to a `vtkPolyData` centerline in native
        NIfTI/VTK coordinates. The CTA NIfTI affine and shape are taken from the
        extractor's loaded CTA — loaded on demand from `self.cta_nifti_path` if not
        already available.

        Parameters
        ----------
        graph : networkx.Graph
            Single-segment graph to convert.
        use_branch_model_for_blanking : bool, optional
            If True (default), derives the `Blanking` array by nearest-neighbour
            lookup against the branch model loaded from `self.branch_model_path`,
            rather than trusting the pickle's `features femoral["blanking"]` value
            (which may be all-zero if upstream featurization was run without a
            branch model). The branch model is loaded on demand if not already
            available.
        mis_array_name : str, optional
            Name to assign to the MIS radius point-data array. The default is
            "MaximumInscribedSphereRadius".
        compute_frenet : bool, optional
            Whether to compute Tangents/Normals/Binormals from the polyline. The
            default is True.
        save_path : string or path-like object, optional
            If provided, the converted centerline is written to this path.

        Returns
        -------
        centerline : vtk.vtkPolyData
            Centerline polydata in native space.

        """
        if self.cta_array is None or self.cta_affine is None:
            self._load_cta_nifti_from_file()
        branch_model = None
        if use_branch_model_for_blanking:
            if self.branch_model is None:
                self._load_branch_model()
            branch_model = self.branch_model
        centerline = pickle_to_vtk(graph, self.cta_affine, self.cta_array.shape, branch_model=branch_model, mis_array_name=mis_array_name, compute_frenet=compute_frenet)
        if save_path is not None:
            print(f"Saving converted centerline to {save_path}")
            save_vtkpolydata(centerline, save_path)
        return centerline

    def add_radius_arrays(self, centerline_model, surface_model, mis_array_name="MaximumInscribedSphereRadius", save_path=None):
        """
        Adds cross-section-based radius arrays (`Radius CE`, `Radius CC`, `Ovality`)
        as point data to a centerline polydata. Standalone — does not depend on or
        modify the local-graph pipeline.

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Centerline polydata (single segment) carrying `Tangents` and a MIS
            radius array.
        surface_model : vtk.vtkPolyData
            Vessel surface mesh used to cut cross-sections.
        mis_array_name : str, optional
            Name of the MIS radius array on the centerline. The default is
            "MaximumInscribedSphereRadius".
        save_path : string or path-like object, optional
            If provided, the augmented centerline is written to this path.

        Returns
        -------
        out : vtk.vtkPolyData
            New centerline polydata with the radius arrays added.

        """
        out = perform_radius_extraction(centerline_model, surface_model, mis_array_name=mis_array_name)
        if save_path is not None:
            print(f"Saving centerline with radius arrays to {save_path}")
            save_vtkpolydata(out, save_path)
        return out

    def add_curvature_arrays(self, centerline_model, savgol_window_length=50, savgol_polyorder=3, save_path=None):
        """
        Adds Frenet-Serret-derived arrays (`Curvature`, `Torsion`, `Filtered curvature`,
        `Distance from origin`) as point data to a centerline polydata. Standalone —
        does not depend on or modify the local-graph pipeline.

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Centerline polydata (single segment) carrying `Tangents`, `Normals`, and
            `Binormals` arrays.
        savgol_window_length : int, optional
            Window length for Savitzky-Golay smoothing. The default is 50.
        savgol_polyorder : int, optional
            Polynomial order for Savitzky-Golay smoothing. The default is 3.
        save_path : string or path-like object, optional
            If provided, the augmented centerline is written to this path.

        Returns
        -------
        out : vtk.vtkPolyData
            New centerline polydata with the curvature arrays added.

        """
        out = perform_curvature_extraction(centerline_model, savgol_window_length=savgol_window_length, savgol_polyorder=savgol_polyorder)
        if save_path is not None:
            print(f"Saving centerline with curvature arrays to {save_path}")
            save_vtkpolydata(out, save_path)
        return out

    def add_curve_ids(self, centerline_model, peak_height=0.030, peak_width=10, save_path=None):
        """
        Adds a `CurveIds` int point-data array to a centerline polydata,
        segmenting it into successive turns based on peaks in the `Filtered
        curvature` point-data array. Standalone — does not depend on or modify
        the local-graph pipeline. The centerline must already carry a `Filtered
        curvature` array (from a prior call to `add_curvature_arrays` or
        `add_centerline_geometry`).

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Centerline polydata (single segment) with a `Filtered curvature`
            point-data array.
        peak_height : float, optional
            Minimum peak height for `scipy.signal.find_peaks`. The default is
            0.030.
        peak_width : int, optional
            Minimum peak width in samples for `scipy.signal.find_peaks`. The
            default is 10.
        save_path : string or path-like object, optional
            If provided, the augmented centerline is written to this path.

        Returns
        -------
        out : vtk.vtkPolyData
            New centerline polydata with the `CurveIds` array added.

        """
        out = perform_curve_id_extraction(centerline_model, peak_height=peak_height, peak_width=peak_width)
        if save_path is not None:
            print(f"Saving centerline with curve ids to {save_path}")
            save_vtkpolydata(out, save_path)
        return out

    def add_centerline_geometry(self, centerline_model, surface_model=None, mis_array_name="MaximumInscribedSphereRadius", savgol_window_length=50, savgol_polyorder=3, compute_curves=False, peak_height=0.030, peak_width=10, save_path=None):
        """
        Convenience method to chain `add_radius_arrays` (skipped if `surface_model`
        is None) and `add_curvature_arrays`. Returns a single new centerline
        polydata with all arrays added.

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Centerline polydata (single segment).
        surface_model : vtk.vtkPolyData, optional
            Vessel surface mesh. If None, only curvature arrays are added.
        mis_array_name : str, optional
            Name of the MIS radius array on the centerline. The default is
            "MaximumInscribedSphereRadius".
        savgol_window_length : int, optional
            Window length for Savitzky-Golay smoothing. The default is 50.
        savgol_polyorder : int, optional
            Polynomial order for Savitzky-Golay smoothing. The default is 3.
        compute_curves : bool, optional
            If True, additionally segments the centerline into turns and adds a
            `CurveIds` int point-data array via `perform_curve_id_extraction`.
            The default is False.
        peak_height : float, optional
            Minimum peak height for curve-id detection. Only used when
            `compute_curves` is True. The default is 0.030.
        peak_width : int, optional
            Minimum peak width in samples for curve-id detection. Only used when
            `compute_curves` is True. The default is 10.
        save_path : string or path-like object, optional
            If provided, the augmented centerline is written to this path.

        Returns
        -------
        out : vtk.vtkPolyData
            New centerline polydata with all geometry arrays added.

        """
        out = centerline_model
        if surface_model is not None:
            out = perform_radius_extraction(out, surface_model, mis_array_name=mis_array_name)
        out = perform_curvature_extraction(out, savgol_window_length=savgol_window_length, savgol_polyorder=savgol_polyorder)
        if compute_curves:
            out = perform_curve_id_extraction(out, peak_height=peak_height, peak_width=peak_width)
        if save_path is not None:
            print(f"Saving centerline with geometry arrays to {save_path}")
            save_vtkpolydata(out, save_path)
        return out

    def add_carotid_analysis(
        self,
        centerline_model,
        radius_array_name="Radius CE",
        side=None,
        use_landmark=False,
        bulb_landmark_world_mm=None,
        proximal_vessel_substring=None,
        bif_mask_width_mm=25.0,
        bulb_threshold_mm=0.3,
        bulb_expand_nodes=1,
        detect_intracranial=False,
        intracranial_smoothing_sigma=2,
        intracranial_dt_max_mm=20,
        intracranial_dt_tolerance_mm=5,
        intracranial_dt_clip_mm=50,
        save_path=None,
    ):
        """
        Detects the carotid bulb on a CCA→ICA centerline polydata (LCA / RCA /
        BT-RCA convention) and adds the intermediate signals as point-data
        arrays. Thin wrapper around `perform_carotid_analysis` that adds the
        framework save-on-disk convention and the optional per-side landmark
        resolution from ``{case_dir}/{mode}/landmarks.json``. When
        ``detect_intracranial`` is True, the carotid bulb pass is followed by
        an intracranial-transition detection pass (see
        ``add_intracranial_transition`` /
        ``perform_intracranial_transition_detection``) and both sets of arrays
        land on the same returned polydata.

        Only valid on centerlines that span CCA → ICA. The caller is
        responsible for picking the right centerline — there is no
        auto-detection from ``centerline_id``.

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Per-vessel centerline polydata spanning CCA → ICA. Must carry
            `Blanking`, `Distance from origin`, and the radius array named by
            `radius_array_name`. `VesselTypeName` is required only as a fallback
            when neither ``side`` nor ``bulb_landmark_world_mm`` resolves a
            landmark.
        radius_array_name : str, optional
            Name of the radius array on the centerline. The default is
            ``"Radius CE"``; pass ``"MaximumInscribedSphereRadius"`` to fall
            back to the VMTK MIS radius when no surface mesh is available.
        side : {"LCA", "RCA", "BT-RCA"}, optional
            Used to resolve side-dependent defaults: the proximal-trim
            substring (``"CCA"`` for LCA / RCA, ``"BT"`` for BT-RCA), and —
            only when ``use_landmark`` is True — the bifurcation landmark
            label to look up in ``{case_dir}/{mode}/landmarks.json``
            (``l-eica`` for LCA; ``r-eica`` for RCA / BT-RCA).
        use_landmark : bool, optional
            When True (and ``bulb_landmark_world_mm`` not given explicitly),
            load ``{case_dir}/{mode}/landmarks.json`` and centre the
            bifurcation mask on the centerline node closest to the side's
            landmark. If the file or label is missing, falls back to the
            ``VesselTypeName`` string-match centre with a warning. Off by
            default — the centre is recovered from ``VesselTypeName`` unless
            the caller explicitly opts in. The default is False.
        bulb_landmark_world_mm : sequence of float, optional
            Explicit world-coordinate ``(x, y, z)`` of the side's
            bifurcation landmark. Overrides both ``use_landmark`` and the
            ``VesselTypeName`` fallback.
        proximal_vessel_substring : str, optional
            Substring used to identify the proximal vessel segment for the
            leading-edge trim. When None, resolved from ``side``: ``"CCA"``
            for LCA / RCA, ``"BT"`` for BT-RCA. Defaults to ``"CCA"`` when
            ``side`` is also None. Explicit values override the side-based
            mapping.
        bif_mask_width_mm : float, optional
            Total arclength width (mm) of the bifurcation mask, centred on
            the bifurcation node. The default is 40.0 (4 cm: 2 cm proximal +
            2 cm distal).
        bulb_threshold_mm : float, optional
            A centerline node is threshold-positive when
            ``|raw - interp| > bulb_threshold_mm``. The bulb is the closed
            interval between the first and last threshold-positive node
            (interior dips below threshold are kept as bulb). The default
            is 0.3.
        bulb_expand_nodes : int, optional
            Symmetric expansion of the bulb span, in nodes per side. The
            default is 1.
        detect_intracranial : bool, optional
            If True, additionally runs the intracranial-transition detection
            and adds ``Intracranial`` and ``DistanceTransformValueSmoothed``
            point-data arrays. Off by default because the underlying cranium
            EDT is expensive; results are cached in memory and on disk so the
            cost is paid once per case. The default is False.
        intracranial_smoothing_sigma : float, optional
            Gaussian smoothing sigma (in nodes) for the per-point DT signal
            before differentiation. Only used when ``detect_intracranial`` is
            True. The default is 2.
        intracranial_dt_max_mm : float, optional
            DT-value upper bound for a qualifying minimum (mm). Only used
            when ``detect_intracranial`` is True. The default is 20.
        intracranial_dt_tolerance_mm : float, optional
            DT-value tolerance around the absolute minimum for the
            transition-selection rule (mm). Only used when
            ``detect_intracranial`` is True. The default is 5.
        intracranial_dt_clip_mm : float, optional
            DT clipping value (mm). Only used when ``detect_intracranial`` is
            True. The default is 50.
        save_path : string or path-like object, optional
            If provided, the augmented centerline is written to this path.

        Returns
        -------
        out : vtk.vtkPolyData
            New centerline polydata with `KeepAfterBlankingTrim`,
            `Radius interp`, `Radius diff (raw - interp)`,
            `BifurcationMask`, and `BulbMask` arrays added — plus
            `Intracranial` and `DistanceTransformValueSmoothed` when
            ``detect_intracranial`` is True.

        """
        if bulb_landmark_world_mm is None and use_landmark and side is not None:
            bulb_landmark_world_mm = self._resolve_bulb_landmark(side)
        if proximal_vessel_substring is None:
            proximal_vessel_substring = self._resolve_proximal_vessel_substring(side)

        out = perform_carotid_analysis(
            centerline_model,
            radius_array_name=radius_array_name,
            bulb_landmark_world_mm=bulb_landmark_world_mm,
            proximal_vessel_substring=proximal_vessel_substring,
            bif_mask_width_mm=bif_mask_width_mm,
            bulb_threshold_mm=bulb_threshold_mm,
            bulb_expand_nodes=bulb_expand_nodes,
        )

        if detect_intracranial:
            if self.cta_array is None or self.cta_affine is None:
                self._load_cta_nifti_from_file()
            distance_transform = self._get_or_compute_cranium_distance_transform(intracranial_dt_clip_mm)
            out, _, _, _ = perform_intracranial_transition_detection(
                out,
                self.cta_array,
                self.cta_affine,
                distance_transform=distance_transform,
                smoothing_sigma=intracranial_smoothing_sigma,
                dt_max_mm=intracranial_dt_max_mm,
                dt_tolerance_mm=intracranial_dt_tolerance_mm,
                dt_clip_mm=intracranial_dt_clip_mm,
            )

        if save_path is not None:
            print(f"Saving centerline with carotid analysis arrays to {save_path}")
            save_vtkpolydata(out, save_path)
        return out

    def add_intracranial_transition(
        self,
        centerline_model,
        smoothing_sigma=2,
        dt_max_mm=20,
        dt_tolerance_mm=5,
        dt_clip_mm=50,
        save_path=None,
    ):
        """
        Detects the intracranial transition on a CCA→ICA centerline polydata
        and adds the resulting point-data arrays. Thin wrapper around
        ``perform_intracranial_transition_detection`` that owns the cranium-DT
        caching (in-memory on ``self._cranium_distance_transform`` and on
        disk under ``{case_dir}/{mode}/cranium_distance_transform.npy``) so
        repeated calls in the same session — and across sessions — don't
        recompute the EDT.

        Parameters
        ----------
        centerline_model : vtk.vtkPolyData
            Per-vessel centerline polydata.
        smoothing_sigma : float, optional
            Gaussian smoothing sigma (in nodes). The default is 2.
        dt_max_mm : float, optional
            DT-value upper bound for a qualifying minimum (mm). The default
            is 20.
        dt_tolerance_mm : float, optional
            DT-value tolerance around the absolute minimum (mm). The default
            is 5.
        dt_clip_mm : float, optional
            DT clipping value (mm). The default is 50.
        save_path : string or path-like object, optional
            If provided, the augmented centerline is written to this path.

        Returns
        -------
        out : vtk.vtkPolyData
            New centerline polydata with `Intracranial` and
            `DistanceTransformValueSmoothed` arrays added.

        """
        if self.cta_array is None or self.cta_affine is None:
            self._load_cta_nifti_from_file()
        distance_transform = self._get_or_compute_cranium_distance_transform(dt_clip_mm)

        out, _, _, _ = perform_intracranial_transition_detection(
            centerline_model,
            self.cta_array,
            self.cta_affine,
            distance_transform=distance_transform,
            smoothing_sigma=smoothing_sigma,
            dt_max_mm=dt_max_mm,
            dt_tolerance_mm=dt_tolerance_mm,
            dt_clip_mm=dt_clip_mm,
        )

        if save_path is not None:
            print(f"Saving centerline with intracranial-transition arrays to {save_path}")
            save_vtkpolydata(out, save_path)
        return out

    def _get_or_compute_cranium_distance_transform(self, dt_clip_mm):
        """
        Returns the clipped cranium distance transform with three-tier caching:
        (1) in-memory on ``self._cranium_distance_transform``; (2) on disk at
        ``self._cranium_distance_transform_path``; (3) compute fresh, store
        in memory, and persist to disk. The caller is responsible for ensuring
        ``self.cta_array`` is loaded.
        """
        if self._cranium_distance_transform is not None:
            return self._cranium_distance_transform

        if os.path.isfile(self._cranium_distance_transform_path):
            print(f"Loading cranium distance transform from {self._cranium_distance_transform_path}")
            self._cranium_distance_transform = load_numpy(self._cranium_distance_transform_path)
            return self._cranium_distance_transform

        print("Computing cranium distance transform (this can take a while)...")
        from arterial.feature_extraction.vtk_centerline_geometry.carotid_analysis import (
            _compute_cranium_distance_transform,
        )
        distance_transform = _compute_cranium_distance_transform(self.cta_array, dt_clip_mm)
        self._cranium_distance_transform = distance_transform
        os.makedirs(os.path.dirname(self._cranium_distance_transform_path), exist_ok=True)
        print(f"Saving cranium distance transform to {self._cranium_distance_transform_path}")
        save_numpy(distance_transform, self._cranium_distance_transform_path)
        return distance_transform

    def _resolve_bulb_landmark(self, side):
        """
        Returns the world-coordinate ``(x, y, z)`` of the side's bifurcation
        landmark from ``{case_dir}/{mode}/landmarks.json``. ``side`` must be
        one of ``"LCA"``, ``"RCA"``, ``"BT-RCA"``; LCA maps to ``l-eica`` and
        RCA / BT-RCA map to ``r-eica`` (the carotid-bifurcation landmarks
        emitted by ``LandmarkDetector``). Returns ``None`` (with a warning)
        when the file or label is missing, so the caller can fall back to
        ``VesselTypeName`` string matching.
        """
        side_to_label = {"LCA": "l-eica", "RCA": "r-eica", "BT-RCA": "r-eica"}
        if side not in side_to_label:
            raise ValueError(
                f"Unknown side {side!r}; expected one of {list(side_to_label)}."
            )
        label = side_to_label[side]
        landmarks_path = os.path.join(self.case_dir, self.mode, "landmarks.json")
        if not os.path.isfile(landmarks_path):
            print(
                f"WARNING: landmarks.json not found at {landmarks_path}; "
                f"falling back to VesselTypeName string match for {side}."
            )
            return None
        landmarks = load_json(landmarks_path)
        if label not in landmarks:
            print(
                f"WARNING: label {label!r} not found in {landmarks_path}; "
                f"falling back to VesselTypeName string match for {side}."
            )
            return None
        coord = tuple(landmarks[label])
        # The landmark detector emits (0, 0, 0) as a sentinel when no
        # reliable landmark was found; treat that as "missing" and fall back.
        if all(abs(float(c)) < 1e-6 for c in coord):
            print(
                f"WARNING: {label!r} in {landmarks_path} is the (0, 0, 0) "
                f"sentinel (no reliable landmark detected); falling back to "
                f"VesselTypeName string match for {side}."
            )
            return None
        return coord

    def _resolve_proximal_vessel_substring(self, side):
        """
        Returns the ``VesselTypeName`` substring used to anchor the leading-
        edge trim in carotid analysis. ``"CCA"`` for LCA / RCA (matches
        ``LCCA`` / ``RCCA``); ``"BT"`` for BT-RCA (brachiocephalic-trunk
        proximal end). Defaults to ``"CCA"`` when ``side`` is None.
        """
        if side is None:
            return "CCA"
        side_to_substring = {"LCA": "CCA", "RCA": "CCA", "BT-RCA": "BT"}
        if side not in side_to_substring:
            raise ValueError(
                f"Unknown side {side!r}; expected one of {list(side_to_substring)}."
            )
        return side_to_substring[side]

    def is_local_featurized(self):
        """
        Tells whether local features have been extracted on the local graph.

        Returns
        -------
        is_featurized : bool
            False when the local graph has not been built yet.

        """
        if self.local_graph is None or self.local_graph.number_of_nodes() == 0:
            return False
        return "features femoral" in self.local_graph.nodes[0]

    def is_segment_featurized(self):
        """
        Tells whether segment features have been extracted on the local graph.

        Returns
        -------
        is_featurized : bool
            False when the local graph has not been built yet.

        """
        return self.local_graph is not None and "segment_features" in self.local_graph.graph

    def is_global_featurized(self):
        """
        Tells whether global features have been extracted on the local graph.

        Returns
        -------
        is_featurized : bool
            False when the local graph has not been built yet.

        """
        return self.local_graph is not None and "aortic_arch_type" in self.local_graph.graph

    def _load_cta_nifti_from_file           (self):
        if not os.path.isfile(self.cta_nifti_path):
            raise FileNotFoundError(f"CTA nifti file not found in {self.cta_nifti_path}")
        
        self.cta_nifti = load_nifti(self.cta_nifti_path)
        self.cta_array = self.cta_nifti.get_fdata()
        self.cta_affine = self.cta_nifti.affine

    def _load_centerline_segments_array(self):
        if not os.path.isfile(self.centerline_segments_array_path):
            raise FileNotFoundError(f"Centerline segments array not found in {self.centerline_segments_array_path}")
        
        self.centerline_segments_array = load_numpy(self.centerline_segments_array_path)

    def _load_branch_model(self):
        if not os.path.isfile(self.branch_model_path):
            raise FileNotFoundError(f"Branch model not found in {self.branch_model_path}")

        self.branch_model = load_vtkpolydata(self.branch_model_path)

    def _load_segments_graph_pred(self):
        if not os.path.isfile(self.segments_graph_pred_path):
            raise FileNotFoundError(f"Segments graph not found in {self.segments_graph_pred_path}")
        self.segments_graph = load_pickle(self.segments_graph_pred_path)

    def _set_case_dir(self, case_dir):
        if not isinstance(case_dir, str):
            raise ValueError("case_dir should be a string.")
        self.case_dir = case_dir
    
    def _set_mode(self, mode):
        if mode not in ["extracranial_vessels", "intracranial_vessels"]:
            raise ValueError("mode should be either 'extracranial_vessels' or 'intracranial_vessels'.")
        self.mode = mode

    def _set_sampling_distance_mm(self, sampling_distance_mm):
        if not isinstance(sampling_distance_mm, (int, float)):
            raise ValueError("sampling_distance_mm should be an integer or a float.")
        self.sampling_distance_mm = sampling_distance_mm

    def _set_cta_nifti_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.cta_nifti_path = path

    def _set_centerline_segments_array_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.centerline_segments_array_path = path

    def _set_branch_model_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.branch_model_path = path

    def _set_segments_graph_pred_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.segments_graph_pred_path = path

    def _set_local_graph_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.local_graph_path = path

    def _set_local_graph_plot_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.local_graph_plot_path = path

    def _set_single_segments_dir_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.single_segments_dir_path = path    

    def _set_single_segments_plot_path(self, path):  
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.single_segments_plot_path = path   

    def _set_supersegments_dir_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.supersegments_dir_path = path
        
    def _set_supersegments_plot_path(self, path):
        if not isinstance(path, str):
            raise ValueError("path should be a string.")
        self.supersegments_plot_path = path