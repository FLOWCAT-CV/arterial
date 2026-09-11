#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

import numpy as np

from helpers import ArterialTestCase, slow
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor


class TestCenterlineExtractorInit(ArterialTestCase):
    """Constructor, paths and loaders. No VMTK."""

    def setUp(self):
        super().setUp()
        self.segmentation_nifti_path = self.require_fixture("segmentation.nii.gz")
        self.extractor = CenterlineExtractor(self.case_dir, self.mode, self.segmentation_nifti_path, fast_segmentation=False)

    def test_init_stores_arguments_and_paths(self):
        self.assertEqual(self.extractor.case_dir, self.case_dir)
        self.assertEqual(self.extractor.mode, self.mode)
        self.assertEqual(self.extractor.segmentation_nifti_path, self.segmentation_nifti_path)
        self.assertFalse(self.extractor.fast_segmentation)
        expected_dirs = {"centerlines_dir_path": "centerlines", "segmentations_dir_path": "segmentations",
                         "branch_models_dir_path": "branch_models", "clipped_models_dir_path": "clipped_models"}
        for attr, name in expected_dirs.items():
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.extractor, attr), os.path.join(self.mode_dir, name))
        expected_files = {"segmentation_path": "segmentation.vtk", "branch_model_path": "branch_model.vtk",
                          "clipped_model_path": "clipped_model.vtk", "centerline_segments_array_path": "centerline_segments_array.npy"}
        for attr, name in expected_files.items():
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.extractor, attr), os.path.join(self.mode_dir, name))

    def test_init_leaves_results_empty(self):
        for attr in ["segmentation_nifti", "segmentation_array", "segmentation_affine", "image_shape",
                     "segmentation_model", "branch_model", "clipped_model", "centerline_segments_array"]:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(self.extractor, attr))
        for attr in ["centerline_model_list", "segmentation_model_list", "branch_model_list", "clipped_model_list"]:
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.extractor, attr), [])

    def test_setters_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            self.extractor._set_fast_segmentation("fast")

    def test_load_segmentation_nifti(self):
        self.extractor._load_segmentation_nifti_from_file()
        self.assertIsNotNone(self.extractor.segmentation_nifti)
        self.assertEqual(self.extractor.segmentation_affine.shape, (4, 4))
        self.assertEqual(tuple(self.extractor.image_shape), tuple(self.extractor.segmentation_array.shape))

    def test_load_branch_model_from_case_dir(self):
        self.stage_fixture("branch_model.vtk", "branch_model.vtk")
        self.extractor._load_branch_model()
        self.assertIsNotNone(self.extractor.branch_model)
        self.assertGreater(self.extractor.branch_model.GetNumberOfPoints(), 0)


class TestCenterlineExtractorPipeline(ArterialTestCase):
    """VMTK centerline extraction from the fixture segmentation, one stage per subtest."""

    def setUp(self):
        super().setUp()
        self.segmentation_nifti_path = self.require_fixture("segmentation.nii.gz")

    def _run_pipeline(self, fast_segmentation):
        extractor = CenterlineExtractor(self.case_dir, self.mode, self.segmentation_nifti_path, fast_segmentation=fast_segmentation)

        with self.subTest(stage="preprocessing"):
            extractor.perform_preprocessing()
            self.assertIsNotNone(extractor.segmentation_model)
            self.assertGreater(len(extractor.segmentation_model_list), 0)
            self.assertFileExists(extractor.segmentation_path)

        with self.subTest(stage="centerline_extraction"):
            extractor.perform_centerline_extraction()
            self.assertGreater(len(extractor.centerline_model_list), 0)
            for idx in range(len(extractor.centerline_model_list)):
                self.assertFileExists(os.path.join(extractor.centerlines_dir_path, f"centerlines_{idx}.vtk"))
                self.assertFileExists(os.path.join(extractor.segmentations_dir_path, f"segmentation_{idx}.vtk"))

        with self.subTest(stage="branch_model_extraction"):
            extractor.perform_branch_model_extraction()
            self.assertIsNotNone(extractor.branch_model)
            self.assertGreater(extractor.branch_model.GetNumberOfPoints(), 0)
            self.assertFileExists(extractor.branch_model_path)
            for idx, branch_model in enumerate(extractor.branch_model_list):
                if branch_model is not None:
                    self.assertFileExists(os.path.join(extractor.branch_models_dir_path, f"branch_model_{idx}.vtk"))

        with self.subTest(stage="postprocessing"):
            extractor.perform_centerline_postprocessing()
            array = extractor.centerline_segments_array
            self.assertIsNotNone(array)
            self.assertEqual(array.ndim, 2)
            self.assertGreater(array.shape[0], 0)
            self.assertFileExists(extractor.centerline_segments_array_path)
            reloaded = np.load(extractor.centerline_segments_array_path, allow_pickle=True)
            self.assertEqual(reloaded.shape, array.shape)

        with self.subTest(stage="reload_from_case_dir"):
            fresh = CenterlineExtractor(self.case_dir, self.mode, self.segmentation_nifti_path, fast_segmentation=fast_segmentation)
            fresh._load_centerline_model_list()
            fresh._load_segmentation_model_list()
            fresh._load_segmentation()
            fresh._load_branch_model_list()
            fresh._load_branch_model()
            self.assertEqual(len(fresh.centerline_model_list), len(extractor.centerline_model_list))
            self.assertIsNotNone(fresh.segmentation_model)
            self.assertIsNotNone(fresh.branch_model)

    @slow
    def test_fast_pipeline(self):
        self._run_pipeline(fast_segmentation=True)

    @slow
    def test_full_pipeline(self):
        # Full-resolution mode keeps the intracranial vessels; expect tens of minutes of VMTK work.
        self._run_pipeline(fast_segmentation=False)
