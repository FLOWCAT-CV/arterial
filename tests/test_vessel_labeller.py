#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

from helpers import ArterialTestCase, slow
from arterial.vessel_labelling.vessel_labeller import VesselLabeller


class TestVesselLabellerInit(ArterialTestCase):
    """Constructor, paths and loaders. No inference."""

    def setUp(self):
        super().setUp()
        self.centerline_segments_array_path = self.require_fixture("centerline_segments_array.npy")
        self.labeller = VesselLabeller(self.case_dir, self.mode, self.centerline_segments_array_path)

    def test_init_stores_arguments_and_paths(self):
        self.assertEqual(self.labeller.case_dir, self.case_dir)
        self.assertEqual(self.labeller.mode, self.mode)
        self.assertEqual(self.labeller.centerline_segments_array_path, self.centerline_segments_array_path)
        expected = {"segments_graph_path": "segments_graph.pickle", "segments_graph_pred_path": "segments_graph_pred.pickle",
                    "segments_graph_plot_path": "segments_graph.png", "segments_graph_pred_plot_path": "segments_graph_pred.png"}
        for attr, name in expected.items():
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.labeller, attr), os.path.join(self.mode_dir, name))

    def test_init_leaves_results_empty(self):
        for attr in ["centerline_segments_array", "segments_graph", "segments_graph_pred"]:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(self.labeller, attr))

    def test_load_centerline_segments_array(self):
        self.labeller._load_centerline_segments_array()
        self.assertIsNotNone(self.labeller.centerline_segments_array)
        self.assertEqual(self.labeller.centerline_segments_array.ndim, 2)

    def test_load_segments_graph_pred_from_case_dir(self):
        self.stage_fixture("segments_graph_pred.pickle", "segments_graph_pred.pickle")
        self.labeller._load_segments_graph_pred()
        self.assertGreater(self.labeller.segments_graph_pred.number_of_edges(), 0)


class TestVesselLabellerPipeline(ArterialTestCase):
    """Graph construction and GNN vessel-type prediction from the fixture centerline segments."""

    def setUp(self):
        super().setUp()
        self.centerline_segments_array_path = self.require_fixture("centerline_segments_array.npy")
        self.labeller = VesselLabeller(self.case_dir, self.mode, self.centerline_segments_array_path)

    def test_build_segments_graph_without_save(self):
        self.labeller.build_segments_graph(save=False)
        graph = self.labeller.segments_graph
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)
        self.assertFileMissing(self.labeller.segments_graph_path)
        self.assertFileMissing(self.labeller.segments_graph_plot_path)

    def test_build_segments_graph_with_save(self):
        self.labeller.build_segments_graph(save=True)
        self.assertFileExists(self.labeller.segments_graph_path)
        self.assertFileExists(self.labeller.segments_graph_plot_path)

    def _check_prediction(self, graph):
        self.assertIsNotNone(graph)
        self.assertGreater(graph.number_of_edges(), 0)
        names = set()
        for _, _, data in graph.edges(data=True):
            self.assertIn("vessel_type", data)
            self.assertIn("vessel_type_name", data)
            names.add(data["vessel_type_name"])
        self.assertGreater(len(names), 1, "every edge got the same vessel type")

    @slow
    def test_predict_single_model(self):
        self.labeller.build_segments_graph(save=False)
        self.labeller.predict_vessel_types(ensemble=False, save=False)
        self._check_prediction(self.labeller.segments_graph_pred)
        self.assertFileMissing(self.labeller.segments_graph_pred_path)

    @slow
    def test_predict_ensemble_and_save(self):
        self.labeller.build_segments_graph(save=True)
        self.labeller.predict_vessel_types(ensemble=True, save=True)
        self._check_prediction(self.labeller.segments_graph_pred)
        self.assertFileExists(self.labeller.segments_graph_pred_path)
        self.assertFileExists(self.labeller.segments_graph_pred_plot_path)
        fresh = VesselLabeller(self.case_dir, self.mode, self.centerline_segments_array_path)
        fresh._load_segments_graph_pred()
        self.assertEqual(fresh.segments_graph_pred.number_of_edges(), self.labeller.segments_graph_pred.number_of_edges())
