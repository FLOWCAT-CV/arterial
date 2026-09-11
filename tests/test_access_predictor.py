#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import json

from helpers import ArterialTestCase, slow
from arterial.access_prediction.access_predictor import AccessPredictor

ACCESS = ["femoral"]
SIDE = ["left", "right"]


class AccessPredictorTestCase(ArterialTestCase):
    """Builds an AccessPredictor over the fixture CTA and supersegments."""

    def setUp(self):
        super().setUp()
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")
        self.supersegments_dir_path = self.require_fixture("supersegments")
        for access in ACCESS:
            for side in SIDE:
                self.require_fixture("supersegments", f"{access} + {side} + anterior.pickle")
        self.predictor = AccessPredictor(case_dir=self.case_dir, cta_nifti_path=self.cta_nifti_path,
                                         supersegments_dir_path=self.supersegments_dir_path, access=list(ACCESS), side=list(SIDE))

    def output_dir(self, access, side):
        return os.path.join(self.predictor.access_prediction_dir_path, f"{access}_{side}")


class TestAccessPredictorInit(AccessPredictorTestCase):
    """Constructor and CTA geometry. No inference."""

    def test_init_stores_arguments_and_paths(self):
        self.assertEqual(self.predictor.case_dir, self.case_dir)
        self.assertEqual(self.predictor.cta_nifti_path, self.cta_nifti_path)
        self.assertEqual(self.predictor.access, ACCESS)
        self.assertEqual(self.predictor.side, SIDE)
        self.assertEqual(self.predictor.access_prediction_dir_path, os.path.join(self.case_dir, "extracranial_vessels", "access_prediction"))

    def test_init_leaves_results_empty(self):
        self.assertIsNone(self.predictor.cta_nifti)
        self.assertIsNone(self.predictor.lpi_corner_coordinates)
        for attr in ["raw_supersegment_dict", "preprocessed_supersegment_dict", "predictions_dict",
                     "attention_maps_dict", "attention_maps_vtk_dict", "attention_maps_point_dict"]:
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.predictor, attr), {})

    def test_default_access_and_side_are_fresh_lists(self):
        first = AccessPredictor(case_dir=self.case_dir)
        second = AccessPredictor(case_dir=self.case_dir)
        first.access.append("radial")
        self.assertEqual(second.access, ["femoral"])
        self.assertEqual(second.side, ["left", "right"])

    def test_load_cta_nifti(self):
        self.predictor._load_cta_nifti_from_file()
        self.assertIsNotNone(self.predictor.cta_nifti)

    def test_compute_lpi_corner_coordinates(self):
        self.predictor.compute_lpi_corner_coordinates()
        self.assertIsNotNone(self.predictor.cta_nifti)
        self.assertEqual(len(self.predictor.lpi_corner_coordinates), 3)


class TestAccessPredictorPipeline(AccessPredictorTestCase):
    """Supersegment preprocessing and ArterialGNet prediction."""

    def test_preprocess_supersegments(self):
        self.predictor.preprocess_supersegments(save=True)
        for access in ACCESS:
            for side in SIDE:
                with self.subTest(access=access, side=side):
                    key = (access, side)
                    self.assertIn(key, self.predictor.raw_supersegment_dict)
                    preprocessed = self.predictor.preprocessed_supersegment_dict[key]
                    for part in ["global_features", "segment_graph", "dense_graph"]:
                        self.assertIn(part, preprocessed)
                    for name in ["global_features.json", "segment_supersegment.pickle", "dense_supersegment.pickle",
                                 "preprocessed_supersegment_dict.pickle", "combined_plot.png"]:
                        self.assertFileExists(os.path.join(self.output_dir(access, side), name))

    @slow
    def test_predict_accessibility_with_attention(self):
        self.predictor.preprocess_supersegments(save=True)
        self.predictor.predict_accessibility(return_attention_map=True, save=True)
        for access in ACCESS:
            for side in SIDE:
                with self.subTest(access=access, side=side):
                    key = (access, side)
                    prediction = self.predictor.predictions_dict[key]
                    self.assertGreaterEqual(float(prediction["mean"]), 0.0)
                    self.assertLessEqual(float(prediction["mean"]), 1.0)
                    self.assertGreaterEqual(float(prediction["std"]), 0.0)
                    self.assertIn(key, self.predictor.attention_maps_dict)
                    self.assertIn(key, self.predictor.attention_maps_vtk_dict)
                    self.assertIn(key, self.predictor.attention_maps_point_dict)
                    out = self.output_dir(access, side)
                    for name in ["access_prediction.json", "attention_map.pickle", "attention_map.png",
                                 "attention_map.vtk", "attention_map_points.json"]:
                        self.assertFileExists(os.path.join(out, name))
                    with open(os.path.join(out, "access_prediction.json")) as handle:
                        saved = json.load(handle)
                    self.assertAlmostEqual(float(saved["mean"]), float(prediction["mean"]), places=6)
