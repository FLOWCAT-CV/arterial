#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import unittest
import os, shutil
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

class TestCenterlineExtractor(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.mode = "extracranial_vessels"
        self.segmentation_nifti_path = os.path.join(self.case_dir, "input_test_data", "segmentation.nii.gz")
        self.fast_segmentation = False
        self.centerline_extractor = CenterlineExtractor(self.case_dir, self.mode, self.segmentation_nifti_path, self.fast_segmentation)

    def test_init(self):
        self.assertEqual(self.centerline_extractor.case_dir, self.case_dir)
        self.assertEqual(self.centerline_extractor.mode, self.mode)
        self.assertEqual(self.centerline_extractor.segmentation_nifti_path, self.segmentation_nifti_path)
        self.assertEqual(self.centerline_extractor.fast_segmentation, self.fast_segmentation)
        self.assertIsNone(self.centerline_extractor.segmentation_nifti)
        self.assertIsNone(self.centerline_extractor.segmentation_array)
        self.assertIsNone(self.centerline_extractor.segmentation_affine)
        self.assertIsNone(self.centerline_extractor.image_shape)
        self.assertEqual(self.centerline_extractor.centerlines_dir_path, os.path.join(self.case_dir, self.mode, "centerlines"))
        self.assertEqual(self.centerline_extractor.segmentations_dir_path, os.path.join(self.case_dir, self.mode, "segmentations"))
        self.assertEqual(self.centerline_extractor.branch_models_dir_path, os.path.join(self.case_dir, self.mode, "branch_models"))
        self.assertEqual(self.centerline_extractor.clipped_models_dir_path, os.path.join(self.case_dir, self.mode, "clipped_models"))
        self.assertEqual(self.centerline_extractor.centerline_model_list, [])
        self.assertEqual(self.centerline_extractor.segmentation_model_list, [])
        self.assertEqual(self.centerline_extractor.branch_model_list, [])
        self.assertEqual(self.centerline_extractor.clipped_model_list, [])
        self.assertEqual(self.centerline_extractor.segmentation_path, os.path.join(self.case_dir, self.mode, "segmentation.vtk"))
        self.assertEqual(self.centerline_extractor.branch_model_path, os.path.join(self.case_dir, self.mode, "branch_model.vtk"))
        self.assertEqual(self.centerline_extractor.clipped_model_path, os.path.join(self.case_dir, self.mode, "clipped_model.vtk"))
        self.assertIsNone(self.centerline_extractor.segmentation_model)
        self.assertIsNone(self.centerline_extractor.branch_model)
        self.assertIsNone(self.centerline_extractor.clipped_model)
        self.assertEqual(self.centerline_extractor.centerline_segments_array_path, os.path.join(self.case_dir, self.mode, "centerline_segments_array.npy"))
        self.assertIsNone(self.centerline_extractor.centerline_segments_array)

    def test_full_pipeline(self):
        self.centerline_extractor.perform_centerline_extraction()
        self.assertNotEqual(self.centerline_extractor.centerline_model_list, [])
        self.assertNotEqual(self.centerline_extractor.segmentation_model_list, [])
        self.assertIsNotNone(self.centerline_extractor.segmentation_model)
        for idx in range(len(self.centerline_extractor.centerline_model_list)):
            self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.centerlines_dir_path, f"centerlines_{idx}.vtk")))
            self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.segmentations_dir_path, f"segmentation_{idx}.vtk")))
        self.assertTrue(os.path.exists(self.centerline_extractor.segmentation_path))

        self.centerline_extractor.perform_branch_model_extraction()
        self.assertNotEqual(self.centerline_extractor.branch_model_list, [])
        self.assertIsNotNone(self.centerline_extractor.branch_model)
        for idx, branch_model in enumerate(self.centerline_extractor.branch_model_list):
            if branch_model is not None:
                self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.branch_models_dir_path, f"branch_model_{idx}.vtk")))
        self.assertTrue(os.path.exists(self.centerline_extractor.branch_model_path))
    
        # self.centerline_extractor.perform_clipped_model_extraction()
        # self.assertNotEqual(self.centerline_extractor.clipped_model_list, [])
        # self.assertIsNotNone(self.centerline_extractor.clipped_model)
        # for idx, clipped_model in enumerate(self.centerline_extractor.clipped_model_list):
        #     if clipped_model is not None:
        #         self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.clipped_models_dir_path, f"clipped_model_{idx}.vtk")))
        # self.assertTrue(os.path.exists(self.centerline_extractor.clipped_model_path))

        self.centerline_extractor.perform_centerline_postprocessing()
        self.assertIsNotNone(self.centerline_extractor.centerline_segments_array)

        self.centerline_extractor.segmentation_nifti = None
        self.centerline_extractor.segmentation_affine = None
        self.centerline_extractor.image_shape = None
        self.centerline_extractor.centerline_model_list = []
        self.centerline_extractor.segmentation_model_list = []
        self.centerline_extractor.segmentation_model = None
        self.centerline_extractor.branch_model_list = []
        self.centerline_extractor.branch_model = None
        self.centerline_extractor.clipped_model_list = []
        self.centerline_extractor.clipped_model = None
        self.centerline_extractor.centerline_segments_array = None

        self.centerline_extractor._load_segmentation_nifti_from_file()
        self.assertIsNotNone(self.centerline_extractor.segmentation_nifti)
        self.assertIsNotNone(self.centerline_extractor.segmentation_affine)
        self.assertIsNotNone(self.centerline_extractor.image_shape)

        self.centerline_extractor._load_centerline_model_list()
        self.assertNotEqual(self.centerline_extractor.centerline_model_list, [])

        self.centerline_extractor._load_segmentation_model_list()
        self.assertNotEqual(self.centerline_extractor.segmentation_model_list, [])

        self.centerline_extractor._load_segmentation()
        self.assertIsNotNone(self.centerline_extractor.segmentation_model)

        self.centerline_extractor._load_branch_model_list()
        self.assertNotEqual(self.centerline_extractor.branch_model_list, [])

        self.centerline_extractor._load_branch_model()
        self.assertIsNotNone(self.centerline_extractor.branch_model)

        # self.centerline_extractor._load_clipped_model_list()
        # self.assertNotEqual(self.centerline_extractor.clipped_model_list, [])

        # self.centerline_extractor._load_clipped_model()
        # self.assertIsNotNone(self.centerline_extractor.clipped_model)

    def test_fast_pipeline(self):
        self.centerline_extractor.centerline_model_list = []
        self.centerline_extractor.segmentation_model_list = []
        self.centerline_extractor.segmentation_model = None
        self.centerline_extractor.branch_model_list = []
        self.centerline_extractor.branch_model = None
        self.centerline_extractor.clipped_model_list = []
        self.centerline_extractor.clipped_model = None
        self.centerline_extractor.centerline_segments_array = None
        self.centerline_extractor._set_fast_segmentation(True)

        self.centerline_extractor.perform_centerline_extraction()
        self.assertNotEqual(self.centerline_extractor.centerline_model_list, [])
        self.assertNotEqual(self.centerline_extractor.segmentation_model_list, [])
        self.assertIsNotNone(self.centerline_extractor.segmentation_model)
        for idx in range(len(self.centerline_extractor.centerline_model_list)):
            self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.centerlines_dir_path, f"centerlines_{idx}.vtk")))
            self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.segmentations_dir_path, f"segmentation_{idx}.vtk")))
        self.assertTrue(os.path.exists(self.centerline_extractor.segmentation_path))

        self.centerline_extractor.perform_branch_model_extraction()
        self.assertNotEqual(self.centerline_extractor.branch_model_list, [])
        self.assertIsNotNone(self.centerline_extractor.branch_model)
        for idx, branch_model in enumerate(self.centerline_extractor.branch_model_list):
            if branch_model is not None:
                self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.branch_models_dir_path, f"branch_model_{idx}.vtk")))
        self.assertTrue(os.path.exists(self.centerline_extractor.branch_model_path))

        self.centerline_extractor.perform_centerline_postprocessing()
        self.assertIsNotNone(self.centerline_extractor.centerline_segments_array)

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