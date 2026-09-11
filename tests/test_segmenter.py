#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

import numpy as np
import nibabel as nib

from helpers import ArterialTestCase, slow
from arterial.segmentation.segmenter import VesselSegmenter


class TestVesselSegmenterInit(ArterialTestCase):
    """Constructor and setter behaviour. No inference."""

    def setUp(self):
        super().setUp()
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")
        self.segmenter = VesselSegmenter(self.case_dir, self.mode, self.cta_nifti_path, fast_segmentation=True)

    def test_init_stores_arguments(self):
        self.assertEqual(self.segmenter.case_dir, self.case_dir)
        self.assertEqual(self.segmenter.mode, self.mode)
        self.assertEqual(self.segmenter.cta_nifti_path, self.cta_nifti_path)
        self.assertTrue(self.segmenter.fast_segmentation)

    def test_init_leaves_volumes_unloaded(self):
        for attr in ["cta_nifti", "cta_array", "cta_affine", "segmentation_nifti", "segmentation_array",
                     "cta_head_array", "cta_neck_array", "cta_head_affine",
                     "segmentation_head_array", "segmentation_neck_array"]:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(self.segmenter, attr))

    def test_setters_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            self.segmenter._set_mode("not_a_mode")
        with self.assertRaises(ValueError):
            self.segmenter._set_fast_segmentation("yes")
        with self.assertRaises(ValueError):
            self.segmenter._set_case_dir(42)

    def test_missing_cta_raises(self):
        segmenter = VesselSegmenter(self.case_dir, self.mode, os.path.join(self.case_dir, "nope.nii.gz"))
        with self.assertRaises(FileNotFoundError):
            segmenter._load_cta_nifti_from_file()


class TestVesselSegmenterInference(ArterialTestCase):
    """nnU-Net inference on the fixture CTA. One test per mode."""

    def setUp(self):
        super().setUp()
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")
        self.cta_shape = nib.load(self.cta_nifti_path).shape

    def _check_segmentation(self, segmenter, expected_shape):
        self.assertIsNotNone(segmenter.segmentation_nifti)
        self.assertIsNotNone(segmenter.segmentation_array)
        self.assertEqual(tuple(segmenter.segmentation_array.shape), tuple(expected_shape))
        labels = np.unique(segmenter.segmentation_array)
        self.assertTrue(set(labels.tolist()) <= {0, 1}, f"unexpected labels {labels}")
        self.assertGreater(int((segmenter.segmentation_array > 0).sum()), 1000, "segmentation is (almost) empty")
        self.assertFileExists(os.path.join(self.mode_dir, "segmentation.nii.gz"))

    @slow
    def test_extracranial_fast_segmentation(self):
        segmenter = VesselSegmenter(self.case_dir, "extracranial_vessels", self.cta_nifti_path, fast_segmentation=True)
        segmenter.segment_vessels_from_cta()
        self._check_segmentation(segmenter, self.cta_shape)

    @slow
    def test_extracranial_full_segmentation(self):
        segmenter = VesselSegmenter(self.case_dir, "extracranial_vessels", self.cta_nifti_path, fast_segmentation=False)
        segmenter.segment_vessels_from_cta()
        self._check_segmentation(segmenter, self.cta_shape)
        for attr in ["cta_head_array", "cta_neck_array", "cta_head_affine",
                     "segmentation_head_array", "segmentation_neck_array"]:
            with self.subTest(attr=attr):
                self.assertIsNotNone(getattr(segmenter, attr))
        self.assertLess(segmenter.cta_head_array.shape[2], self.cta_shape[2], "head slab should be shorter than the CTA")

    @slow
    def test_intracranial_segmentation(self):
        self.mode = "intracranial_vessels"
        segmenter = VesselSegmenter(self.case_dir, "intracranial_vessels", self.cta_nifti_path)
        segmenter.segment_vessels_from_cta()
        # Intracranial mode segments the cropped head slab, so the output lives in head-slab space.
        self.assertIsNotNone(segmenter.cta_head_array)
        self.assertIsNotNone(segmenter.cta_head_affine)
        self._check_segmentation(segmenter, segmenter.cta_head_array.shape)
        self.assertTrue(all(h <= c for h, c in zip(segmenter.cta_head_array.shape, self.cta_shape)), "head slab larger than the CTA")
