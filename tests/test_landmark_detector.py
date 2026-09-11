#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

import numpy as np
import nibabel as nib

from helpers import ArterialTestCase, slow, ras_bounding_box
from arterial.landmark_detection.landmark_detector import LandmarkDetector

EXPECTED_LANDMARKS = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]


class TestLandmarkDetectorInit(ArterialTestCase):
    """Constructor, paths and CTA loading. No inference."""

    def setUp(self):
        super().setUp()
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")
        self.detector = LandmarkDetector(self.case_dir, self.mode, self.cta_nifti_path)

    def test_init_stores_arguments_and_paths(self):
        self.assertEqual(self.detector.case_dir, self.case_dir)
        self.assertEqual(self.detector.mode, self.mode)
        self.assertEqual(self.detector.cta_nifti_path, self.cta_nifti_path)
        self.assertEqual(self.detector.landmarks_ras_json_path, os.path.join(self.mode_dir, "landmarks.json"))
        self.assertEqual(self.detector.landmarks_slicer_json_path, os.path.join(self.mode_dir, "landmarks_slicer.json"))
        self.assertEqual(self.detector.predicted_mask_nifti_path, os.path.join(self.mode_dir, "landmarks_mask.nii.gz"))

    def test_init_leaves_results_empty(self):
        for attr in ["cta_nifti", "cta_array", "cta_affine", "landmarks_ras_mm_dict",
                     "landmarks_slicer_json", "predicted_mask_nifti"]:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(self.detector, attr))

    def test_load_cta_nifti(self):
        self.detector._load_cta_nifti_from_file()
        self.assertIsNotNone(self.detector.cta_nifti)
        self.assertEqual(self.detector.cta_array.ndim, 3)
        self.assertEqual(self.detector.cta_affine.shape, (4, 4))


class TestLandmarkDetectorInference(ArterialTestCase):
    """3D U-Net inference on the fixture CTA."""

    def setUp(self):
        super().setUp()
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")
        self.lower, self.upper = ras_bounding_box(nib.load(self.cta_nifti_path))

    def _check_landmarks(self, landmarks):
        self.assertIsNotNone(landmarks)
        self.assertEqual(set(landmarks.keys()), set(EXPECTED_LANDMARKS))
        for name, coords in landmarks.items():
            with self.subTest(landmark=name):
                coords = np.asarray(coords, dtype=float)
                self.assertEqual(coords.shape, (3,))
                self.assertTrue(np.all(np.isfinite(coords)), f"non-finite coordinate {coords}")
                self.assertTrue(np.all(coords >= self.lower - 1) and np.all(coords <= self.upper + 1),
                                f"{name} at {coords} lies outside the CTA bounding box {self.lower}..{self.upper}")

    @slow
    def test_detect_cta_only_saves_json(self):
        detector = LandmarkDetector(self.case_dir, self.mode, self.cta_nifti_path)
        landmarks = detector.detect_landmarks_on_cta(return_mask=False, save=True, use_segmentation_model=False, refine_with_segmentation=False)
        self._check_landmarks(landmarks)
        self.assertIs(landmarks, detector.landmarks_ras_mm_dict)
        self.assertIsNotNone(detector.landmarks_slicer_json)
        self.assertFileExists(detector.landmarks_ras_json_path)
        self.assertFileExists(detector.landmarks_slicer_json_path)
        self.assertFileMissing(detector.predicted_mask_nifti_path)

    @slow
    def test_detect_with_mask(self):
        detector = LandmarkDetector(self.case_dir, self.mode, self.cta_nifti_path)
        detector.detect_landmarks_on_cta(return_mask=True, save=True, use_segmentation_model=False, refine_with_segmentation=False)
        self._check_landmarks(detector.landmarks_ras_mm_dict)
        self.assertFileExists(detector.predicted_mask_nifti_path)
        mask = np.asarray(nib.load(detector.predicted_mask_nifti_path).dataobj)
        self.assertEqual(mask.ndim, 3)
        self.assertTrue(set(np.unique(mask).tolist()) <= set(range(7)), "mask labels should be 0..6")
        self.assertGreater(int((mask > 0).sum()), 0, "mask is empty")

    @slow
    def test_detect_without_save_writes_nothing(self):
        detector = LandmarkDetector(self.case_dir, self.mode, self.cta_nifti_path)
        detector.detect_landmarks_on_cta(return_mask=False, save=False, use_segmentation_model=False, refine_with_segmentation=False)
        self._check_landmarks(detector.landmarks_ras_mm_dict)
        self.assertFalse(os.path.exists(self.mode_dir) and os.listdir(self.mode_dir), "no files should be written when save=False")

    @slow
    def test_detect_with_segmentation_model(self):
        # The two-channel model and segmentation-based refinement, using the fixture segmentation.
        segmentation_path = self.require_fixture("segmentation.nii.gz")
        detector = LandmarkDetector(self.case_dir, self.mode, self.cta_nifti_path, segmentation_nifti_path=segmentation_path)
        detector.detect_landmarks_on_cta(return_mask=False, save=True, use_segmentation_model=True, refine_with_segmentation=True)
        self._check_landmarks(detector.landmarks_ras_mm_dict)
        self.assertFileExists(detector.landmarks_ras_json_path)
