#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import io
import contextlib
import unittest

import numpy as np
import torch

from arterial.landmark_detection import utils as lm_utils
from arterial.landmark_detection.model import MonaiUNet3DSeg

# index layout used by the post-processing: 0=l-tica 1=r-tica 2=l-eica 3=r-eica 4=r-mca 5=l-mca


class TestCoordinateTransforms(unittest.TestCase):

    def test_ijk_ras_roundtrip(self):
        affine = np.array([[0.5, 0, 0, -100], [0, -0.5, 0, 50], [0, 0, 0.8, -20], [0, 0, 0, 1]])
        ijk = np.array([10, 20, 30], dtype=float)
        ras = lm_utils.ijk_to_ras(ijk, affine)
        np.testing.assert_allclose(ras, [-95, 40, 4])
        np.testing.assert_allclose(lm_utils.ras_to_ijk(ras, affine), ijk)


class TestMirrorCorrection(unittest.TestCase):

    def healthy(self):
        mean = lm_utils._TICA_MCA_DIST_MEAN
        return np.array([[-30, 0, 0], [30, 0, 0], [-32, 0, -10], [32, 0, -10],
                         [30 + mean[0], 0, 0], [-30 - mean[1], 0, 0]], dtype=float)

    def test_plausible_landmarks_are_untouched(self):
        centroids = self.healthy()
        corrected, log = lm_utils._apply_mca_mirror_correction(centroids.copy())
        np.testing.assert_allclose(corrected, centroids)
        self.assertEqual(log, {})

    def test_implausible_mca_is_mirrored_from_the_other_side(self):
        centroids = self.healthy()
        centroids[5] = [-120, 0, 0]  # l-mca far too distal
        corrected, log = lm_utils._apply_mca_mirror_correction(centroids.copy())
        expected = centroids[4] * [-1, 1, 1]  # midplane is x = 0 here
        np.testing.assert_allclose(corrected[5], expected)
        self.assertIn("l-mca", log)
        self.assertAlmostEqual(log["l-mca"]["displacement_mm"], round(float(np.linalg.norm(expected - centroids[5])), 1))

    def test_both_sides_bad_means_no_correction(self):
        centroids = self.healthy()
        centroids[4] = [200, 0, 0]; centroids[5] = [-200, 0, 0]
        corrected, log = lm_utils._apply_mca_mirror_correction(centroids.copy())
        np.testing.assert_allclose(corrected, centroids)
        self.assertEqual(log, {})


    def test_missing_mca_is_replaced_by_the_mirrored_one(self):
        centroids = self.healthy()
        centroids[5] = np.nan  # l-mca not detected
        corrected, log = lm_utils._apply_mca_mirror_correction(centroids.copy())
        np.testing.assert_allclose(corrected[5], centroids[4] * [-1, 1, 1])
        self.assertEqual(log, {"l-mca": {"displacement_mm": None}})

    def test_missing_tica_is_left_out_of_the_midplane(self):
        centroids = self.healthy()
        centroids[1] = np.nan  # r-tica not detected: the r-mca distance is unknown and the l-mca is healthy
        corrected, log = lm_utils._apply_mca_mirror_correction(centroids.copy())
        self.assertTrue(np.all(np.isnan(corrected[1])), "a missing landmark must stay missing")
        midplane = np.mean(centroids[[0, 2, 3], 0])
        np.testing.assert_allclose(corrected[4], [2 * midplane - centroids[5, 0], 0, 0])
        self.assertIn("r-mca", log)


class TestPostprocessPreds(unittest.TestCase):

    def test_centroids_and_mask_from_synthetic_logits(self):
        depth, height, width = 20, 24, 28
        preds = torch.full((1, 7, depth, height, width), -5.0)
        # blob centres as (d, h, w); chosen so that the TICA-MCA distances are plausible
        centres = {1: (4, 4, 4), 2: (4, 20, 4), 3: (4, 4, 12), 4: (4, 20, 12), 5: (4, 20, 21), 6: (4, 4, 20)}
        for label, (d, h, w) in centres.items():
            preds[0, label, d - 1:d + 2, h - 1:h + 2, w - 1:w + 2] = 5.0
        affine = np.eye(4)
        centroids, mask = lm_utils.postprocess_preds(preds, affine, return_mask=True)
        self.assertEqual(centroids.shape, (6, 3))
        for label, (d, h, w) in centres.items():
            np.testing.assert_allclose(centroids[label - 1], [h, w, d], atol=1e-3)
        self.assertEqual(mask.shape, (height, width, depth))
        self.assertTrue(set(np.unique(mask).tolist()) <= set(range(7)))
        self.assertEqual(int(mask[4, 4, 4]), 1)
        self.assertEqual(int(mask[20, 21, 4]), 5)

    def test_missing_landmark_class_is_reported_and_not_placed_at_the_corner(self):
        depth, height, width = 20, 24, 28
        preds = torch.full((1, 7, depth, height, width), -5.0)
        centres = {1: (4, 4, 4), 2: (4, 20, 4), 3: (4, 4, 12), 4: (4, 20, 12), 5: (4, 20, 21)}  # class 6 (l-mca) is never predicted
        for label, (d, h, w) in centres.items():
            preds[0, label, d - 1:d + 2, h - 1:h + 2, w - 1:w + 2] = 5.0
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            centroids, _ = lm_utils.postprocess_preds(preds, np.eye(4))
        self.assertIn("landmark class 6 was not detected", output.getvalue())
        self.assertFalse(np.allclose(centroids[5], 0), "a missed landmark must not become the volume corner")
        # The healthy r-mca is mirrored across the midplane instead
        midplane = np.mean(centroids[:4, 0])
        np.testing.assert_allclose(centroids[5], [2 * midplane - centroids[4, 0], centroids[4, 1], centroids[4, 2]], atol=1e-4)

    def test_no_mask_when_not_requested(self):
        preds = torch.full((1, 7, 8, 8, 8), -5.0)
        preds[0, 1:, 3:5, 3:5, 3:5] = 5.0
        centroids, mask = lm_utils.postprocess_preds(preds, np.eye(4), return_mask=False)
        self.assertIsNone(mask)
        self.assertEqual(centroids.shape, (6, 3))


class TestModel(unittest.TestCase):

    def test_unet_forward_shape_on_cpu(self):
        model = MonaiUNet3DSeg(in_channels=2).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 2, 16, 16, 16))
        self.assertEqual(tuple(out.shape), (1, 7, 16, 16, 16))


class TestRefinement(unittest.TestCase):

    def test_nearest_centerline_point_is_measured_in_millimetres(self):
        # 0.5 x 0.5 x 2 mm voxels: 3 voxels away in z is 6 mm, 4 voxels away in x is 2 mm
        affine = np.diag([0.5, 0.5, 2.0, 1.0])
        refiner = lm_utils._LandmarkRefiner({}, np.zeros((10, 10, 10), np.uint8), affine)
        refiner._centerline_points = np.array([[5, 5, 8], [9, 5, 5]])
        refined, distance = refiner._snap_to_centerline(np.array([5.0, 5.0, 5.0]), search_radius_mm=5.0)
        np.testing.assert_array_equal(refined, [9, 5, 5])
        self.assertAlmostEqual(distance, 2.0)

    def test_points_beyond_the_search_radius_in_millimetres_are_not_snapped(self):
        affine = np.diag([0.5, 0.5, 2.0, 1.0])
        refiner = lm_utils._LandmarkRefiner({}, np.zeros((10, 10, 10), np.uint8), affine)
        refiner._centerline_points = np.array([[5, 5, 8]])  # 6 mm away
        refined, distance = refiner._snap_to_centerline(np.array([5.0, 5.0, 5.0]), search_radius_mm=5.0)
        np.testing.assert_array_equal(refined, [5.0, 5.0, 5.0])
        self.assertEqual(distance, 0.0)
