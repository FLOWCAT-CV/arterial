#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

from helpers import ArterialTestCase, slow
from arterial.feature_extraction.feature_extractor import FeatureExtractor


class FeatureExtractorTestCase(ArterialTestCase):
    """Builds a FeatureExtractor over the fixture inputs."""

    def setUp(self):
        super().setUp()
        self.sampling_distance_mm = 2
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")
        self.centerline_segments_array_path = self.require_fixture("centerline_segments_array.npy")
        self.branch_model_path = self.require_fixture("branch_model.vtk")
        self.segments_graph_pred_path = self.require_fixture("segments_graph_pred.pickle")
        self.extractor = FeatureExtractor(self.case_dir, self.mode, self.sampling_distance_mm, self.cta_nifti_path,
                                          self.centerline_segments_array_path, self.branch_model_path, self.segments_graph_pred_path)


class TestFeatureExtractorInit(FeatureExtractorTestCase):
    """Constructor and paths. No feature extraction."""

    def test_init_stores_arguments_and_paths(self):
        self.assertEqual(self.extractor.case_dir, self.case_dir)
        self.assertEqual(self.extractor.mode, self.mode)
        self.assertEqual(self.extractor.sampling_distance_mm, self.sampling_distance_mm)
        self.assertEqual(self.extractor.cta_nifti_path, self.cta_nifti_path)
        self.assertEqual(self.extractor.centerline_segments_array_path, self.centerline_segments_array_path)
        self.assertEqual(self.extractor.branch_model_path, self.branch_model_path)
        self.assertEqual(self.extractor.segments_graph_pred_path, self.segments_graph_pred_path)
        expected = {"local_graph_path": "local_graph.pickle", "local_graph_plot_path": "local_graph.png",
                    "single_segments_dir_path": "single_segments", "single_segments_plot_path": "single_segments.png",
                    "supersegments_dir_path": "supersegments", "supersegments_plot_path": "supersegments.png"}
        for attr, name in expected.items():
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.extractor, attr), os.path.join(self.mode_dir, name))

    def test_init_leaves_results_empty(self):
        for attr in ["cta_nifti", "cta_array", "cta_affine", "centerline_segments_array", "branch_model",
                     "segments_graph", "local_graph", "segment_features", "segments_vessel_type_dict", "supersegments"]:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(self.extractor, attr))
        self.assertFalse(self.extractor.is_local_featurized())
        self.assertFalse(self.extractor.is_segment_featurized())
        self.assertFalse(self.extractor.is_global_featurized())


class TestFeatureExtractorPipeline(FeatureExtractorTestCase):
    """Feature extraction stages in order, one subtest per stage."""

    @slow
    def test_pipeline(self):
        extractor = self.extractor

        with self.subTest(stage="build_local_graph"):
            extractor.build_local_graph()
            self.assertIsNotNone(extractor.centerline_segments_array)
            self.assertIsNotNone(extractor.segments_graph)
            self.assertIsNotNone(extractor.local_graph)
            self.assertGreater(extractor.local_graph.number_of_nodes(), extractor.segments_graph.number_of_nodes(),
                               "the resampled local graph should be denser than the segments graph")
            self.assertFileExists(extractor.local_graph_path)
            self.assertFileExists(extractor.local_graph_plot_path)
            self.assertFalse(extractor.is_local_featurized())

        with self.subTest(stage="local_features"):
            extractor.extract_local_features()
            self.assertIsNotNone(extractor.cta_array)
            self.assertIsNotNone(extractor.cta_affine)
            self.assertIsNotNone(extractor.branch_model)
            self.assertTrue(extractor.is_local_featurized())
            self.assertFalse(extractor.is_segment_featurized())

        with self.subTest(stage="segment_features"):
            extractor.extract_segment_features()
            self.assertIsNotNone(extractor.segment_features)
            self.assertIsNotNone(extractor.segments_vessel_type_dict)
            self.assertTrue(extractor.is_segment_featurized())
            self.assertFalse(extractor.is_global_featurized())
            self.assertDirExists(extractor.single_segments_dir_path)
            self.assertFileExists(extractor.single_segments_plot_path)
            for vessel_type, segment in extractor.segments_vessel_type_dict.items():
                if segment is not None:
                    self.assertFileExists(os.path.join(extractor.single_segments_dir_path, f"{vessel_type}.pickle"))

        with self.subTest(stage="global_features"):
            extractor.extract_global_features()
            self.assertTrue(extractor.is_global_featurized())

        with self.subTest(stage="supersegments"):
            extractor.extract_supersegments()
            self.assertIsNotNone(extractor.supersegments)
            self.assertGreater(len(extractor.supersegments), 0)
            self.assertDirExists(extractor.supersegments_dir_path)
            self.assertFileExists(extractor.supersegments_plot_path)
            for config in extractor.supersegments:
                self.assertEqual(len(config), 3, f"supersegment key should be (access, side, circulation): {config}")
                self.assertFileExists(os.path.join(extractor.supersegments_dir_path, f"{config[0]} + {config[1]} + {config[2]}.pickle"))
