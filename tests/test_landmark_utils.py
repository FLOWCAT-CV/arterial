#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

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
