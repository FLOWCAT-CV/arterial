import unittest
import os, shutil
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

class TestCenterlineExtractor(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.mode = "extracranial_vessels"
        self.segmentation_nifti_path = os.path.join(self.case_dir, f"input_test_data/{self.mode}_segmentation.nii.gz")
        self.fast_segmentation = False
        self.centerline_extractor = CenterlineExtractor(self.case_dir, self.mode, self.segmentation_nifti_path, self.fast_segmentation)

    def test_init(self):
        self.assertEqual(self.centerline_extractor.case_dir, self.case_dir)
        self.assertEqual(self.centerline_extractor.mode, self.mode)
        self.assertEqual(self.centerline_extractor.segmentation_nifti_path, self.segmentation_nifti_path)
        self.assertEqual(self.centerline_extractor.fast_segmentation, self.fast_segmentation)
        self.assertIsNone(self.centerline_extractor.segmentation_nifti)
        self.assertIsNone(self.centerline_extractor.affine)
        self.assertIsNone(self.centerline_extractor.image_shape)
        self.assertEqual(self.centerline_extractor.centerlines_dir_path, os.path.join(self.case_dir, "centerlines"))
        self.assertEqual(self.centerline_extractor.segmentations_dir_path, os.path.join(self.case_dir, "segmentations"))
        self.assertEqual(self.centerline_extractor.branch_models_dir_path, os.path.join(self.case_dir, "branch_models"))
        self.assertEqual(self.centerline_extractor.clipped_models_dir_path, os.path.join(self.case_dir, "clipped_models"))
        self.assertEqual(self.centerline_extractor.centerline_model_list, [])
        self.assertEqual(self.centerline_extractor.segmentation_model_list, [])
        self.assertEqual(self.centerline_extractor.branch_model_list, [])
        self.assertEqual(self.centerline_extractor.clipped_model_list, [])
        self.assertEqual(self.centerline_extractor.segmentation_path, os.path.join(self.case_dir, f"{self.mode}_segmentation.vtk"))
        self.assertEqual(self.centerline_extractor.branch_model_path, os.path.join(self.case_dir, f"{self.mode}_branch_model.vtk"))
        self.assertEqual(self.centerline_extractor.clipped_model_path, os.path.join(self.case_dir, f"{self.mode}_clipped_model.vtk"))
        self.assertIsNone(self.centerline_extractor.segmentation)
        self.assertIsNone(self.centerline_extractor.branch_model)
        self.assertIsNone(self.centerline_extractor.clipped_model)
        self.assertEqual(self.centerline_extractor.centerline_segments_array_path, os.path.join(self.case_dir, f"{self.mode}_centerline_segments_array.npy"))
        self.assertIsNone(self.centerline_extractor.centerline_segments_array)

    def test_full_pipeline(self):
        self.centerline_extractor.perform_centerline_extraction()
        self.assertNotEqual(self.centerline_extractor.centerline_model_list, [])
        self.assertNotEqual(self.centerline_extractor.segmentation_model_list, [])
        self.assertIsNotNone(self.centerline_extractor.segmentation)
        for idx in range(len(self.centerline_extractor.centerline_model_list)):
            self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.centerlines_dir_path, f"{self.mode}_centerlines_{idx}.vtk")))
            self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.segmentations_dir_path, f"{self.mode}_segmentation_{idx}.vtk")))
        self.assertTrue(os.path.exists(self.centerline_extractor.segmentation_path))

        self.centerline_extractor.perform_branch_model_extraction()
        self.assertNotEqual(self.centerline_extractor.branch_model_list, [])
        self.assertIsNotNone(self.centerline_extractor.branch_model)
        for idx, branch_model in enumerate(self.centerline_extractor.branch_model_list):
            if branch_model is not None:
                self.assertTrue(os.path.exists(os.path.join(self.centerline_extractor.branch_models_dir_path, f"{self.mode}_branch_model_{idx}.vtk")))
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
        self.centerline_extractor.affine = None
        self.centerline_extractor.image_shape = None
        self.centerline_extractor.centerline_model_list = []
        self.centerline_extractor.segmentation_model_list = []
        self.centerline_extractor.segmentation = None
        self.centerline_extractor.branch_model_list = []
        self.centerline_extractor.branch_model = None
        self.centerline_extractor.clipped_model_list = []
        self.centerline_extractor.clipped_model = None
        self.centerline_extractor.centerline_segments_array = None

        self.centerline_extractor.load_segmentation_nifti()
        self.assertIsNotNone(self.centerline_extractor.segmentation_nifti)
        self.assertIsNotNone(self.centerline_extractor.affine)
        self.assertIsNotNone(self.centerline_extractor.image_shape)

        self.centerline_extractor.load_centerline_model_list()
        self.assertNotEqual(self.centerline_extractor.centerline_model_list, [])

        self.centerline_extractor.load_segmentation_model_list()
        self.assertNotEqual(self.centerline_extractor.segmentation_model_list, [])

        self.centerline_extractor.load_segmentation()
        self.assertIsNotNone(self.centerline_extractor.segmentation)

        self.centerline_extractor.load_branch_model_list()
        self.assertNotEqual(self.centerline_extractor.branch_model_list, [])

        self.centerline_extractor.load_branch_model()
        self.assertIsNotNone(self.centerline_extractor.branch_model)

        # self.centerline_extractor.load_clipped_model_list()
        # self.assertNotEqual(self.centerline_extractor.clipped_model_list, [])

        # self.centerline_extractor.load_clipped_model()
        # self.assertIsNotNone(self.centerline_extractor.clipped_model)

    @classmethod
    def tearDownClass(cls):
        # Remove all the files generated during the tests
        cls.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        for filename in os.listdir(cls.case_dir):
            if filename != "input_test_data":
                if os.path.isfile(os.path.join(cls.case_dir, filename)):
                    os.remove(os.path.join(cls.case_dir, filename))
                elif os.path.isdir(os.path.join(cls.case_dir, filename)):
                    shutil.rmtree(os.path.join(cls.case_dir, filename))

if __name__ == '__main__':
    unittest.main()