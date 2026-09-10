#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import unittest
import os, shutil
from arterial.feature_extraction.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.mode = "extracranial_vessels"
        self.sampling_distance_mm = 2
        self.cta_nifti_path = os.path.join(self.case_dir, "input_test_data", "cta.nii.gz")
        self.centerline_segments_array_path = os.path.join(self.case_dir, "input_test_data", "centerline_segments_array.npy")
        self.branch_model_path = os.path.join(self.case_dir, "input_test_data", "branch_model.vtk")
        self.segments_graph_pred_path = os.path.join(self.case_dir, "input_test_data", "segments_graph_pred.pickle")
        self.feature_extractor = FeatureExtractor(self.case_dir, self.mode, self.sampling_distance_mm, self.cta_nifti_path, self.centerline_segments_array_path, self.branch_model_path, self.segments_graph_pred_path)

    def test_init(self):
        self.assertEqual(self.feature_extractor.case_dir, self.case_dir)
        self.assertEqual(self.feature_extractor.mode, self.mode)
        self.assertEqual(self.feature_extractor.sampling_distance_mm, self.sampling_distance_mm)
        self.assertEqual(self.feature_extractor.cta_nifti_path, self.cta_nifti_path)
        self.assertEqual(self.feature_extractor.centerline_segments_array_path, self.centerline_segments_array_path)
        self.assertEqual(self.feature_extractor.branch_model_path, self.branch_model_path)
        self.assertEqual(self.feature_extractor.segments_graph_pred_path, self.segments_graph_pred_path)
        self.assertIsNone(self.feature_extractor.cta_nifti)
        self.assertIsNone(self.feature_extractor.cta_array)
        self.assertIsNone(self.feature_extractor.cta_affine)
        self.assertIsNone(self.feature_extractor.centerline_segments_array)
        self.assertIsNone(self.feature_extractor.branch_model)
        self.assertIsNone(self.feature_extractor.segments_graph)
        self.assertEqual(self.feature_extractor.local_graph, None)
        self.assertEqual(self.feature_extractor.local_graph_path, os.path.join(self.case_dir, self.mode, "local_graph.pickle"))
        self.assertEqual(self.feature_extractor.local_graph_plot_path, os.path.join(self.case_dir, self.mode, "local_graph.png"))
        self.assertIsNone(self.feature_extractor.segment_features)
        self.assertIsNone(self.feature_extractor.segments_vessel_type_dict)
        self.assertEqual(self.feature_extractor.single_segments_dir_path, os.path.join(self.case_dir, self.mode, "single_segments"))
        self.assertEqual(self.feature_extractor.single_segments_plot_path, os.path.join(self.case_dir, self.mode, "single_segments.png"))
        self.assertIsNone(self.feature_extractor.supersegments)
        self.assertEqual(self.feature_extractor.supersegments_dir_path, os.path.join(self.case_dir, self.mode, "supersegments"))
        self.assertEqual(self.feature_extractor.supersegments_plot_path, os.path.join(self.case_dir, self.mode, "supersegments.png"))

    def test_full_pipeline(self):
        self.feature_extractor.build_local_graph()
        self.assertIsNotNone(self.feature_extractor.centerline_segments_array)
        self.assertIsNotNone(self.feature_extractor.segments_graph)
        self.assertTrue(os.path.exists(self.feature_extractor.local_graph_path))
        self.assertTrue(os.path.exists(self.feature_extractor.local_graph_plot_path))
        self.assertFalse(self.feature_extractor.is_local_featurized())
        self.assertFalse(self.feature_extractor.is_segment_featurized())
        self.assertFalse(self.feature_extractor.is_global_featurized())

        self.feature_extractor.extract_local_features()
        self.assertIsNotNone(self.feature_extractor.cta_array)
        self.assertIsNotNone(self.feature_extractor.cta_affine)
        self.assertIsNotNone(self.feature_extractor.branch_model)
        self.assertTrue(self.feature_extractor.is_local_featurized())
        self.assertFalse(self.feature_extractor.is_segment_featurized())
        self.assertFalse(self.feature_extractor.is_global_featurized())

        self.feature_extractor.extract_segment_features()
        self.assertIsNotNone(self.feature_extractor.segments_graph)
        self.assertIsNotNone(self.feature_extractor.segment_features)
        self.assertIsNotNone(self.feature_extractor.segments_vessel_type_dict)
        self.assertTrue(os.path.exists(self.feature_extractor.single_segments_dir_path))
        for vessel_type in self.feature_extractor.segments_vessel_type_dict.keys():
            if self.feature_extractor.segments_vessel_type_dict[vessel_type] is not None:
                self.assertTrue(os.path.exists(os.path.join(self.feature_extractor.single_segments_dir_path, "{}.pickle".format(vessel_type))))
        self.assertTrue(os.path.exists(self.feature_extractor.single_segments_plot_path))
        self.assertTrue(self.feature_extractor.is_local_featurized())
        self.assertTrue(self.feature_extractor.is_segment_featurized())
        self.assertFalse(self.feature_extractor.is_global_featurized())

        self.feature_extractor.extract_global_features()
        self.assertTrue(self.feature_extractor.is_local_featurized())
        self.assertTrue(self.feature_extractor.is_segment_featurized())
        self.assertTrue(self.feature_extractor.is_global_featurized())

        self.feature_extractor.extract_supersegments()
        self.assertIsNotNone(self.feature_extractor.supersegments)
        self.assertTrue(os.path.exists(self.feature_extractor.supersegments_dir_path))
        for config in self.feature_extractor.supersegments.keys():
            self.assertTrue(os.path.exists(os.path.join(self.feature_extractor.supersegments_dir_path, f"{config[0]} + {config[1]} + {config[2]}.pickle")))
        self.assertTrue(os.path.exists(self.feature_extractor.supersegments_plot_path))

    @classmethod
    def tearDownClass(cls):
        # Remove all the files generated during the tests
        cls.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        for filename in os.listdir(cls.case_dir):
            if filename not in ["input_test_data", "output"]:
                if os.path.isfile(os.path.join(cls.case_dir, filename)):
                    os.remove(os.path.join(cls.case_dir, filename))
                elif os.path.isdir(os.path.join(cls.case_dir, filename)):
                    shutil.rmtree(os.path.join(cls.case_dir, filename))

if __name__ == '__main__':
    unittest.main()