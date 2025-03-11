import unittest
import os, shutil
from arterial.segmentation.segmenter import VesselSegmenter

class TestVesselSegmenter(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.mode = "extracranial_vessels"
        self.cta_nifti_path = os.path.join(self.case_dir, "input_test_data/cta.nii.gz")
        self.fast_segmentation = True
        self.segmenter = VesselSegmenter(self.case_dir, self.mode, self.cta_nifti_path, self.fast_segmentation)

    def test_init(self):
        self.assertEqual(self.segmenter.case_dir, self.case_dir)
        self.assertEqual(self.segmenter.mode, self.mode)
        self.assertEqual(self.segmenter.cta_nifti_path, self.cta_nifti_path)
        self.assertEqual(self.segmenter.fast_segmentation, self.fast_segmentation)
        self.assertIsNone(self.segmenter.cta_nifti)
        self.assertIsNone(self.segmenter.cta_array)
        self.assertIsNone(self.segmenter.cta_affine)
        self.assertIsNone(self.segmenter.segmentation_nifti)
        self.assertIsNone(self.segmenter.segmentation_array)
        self.assertIsNone(self.segmenter.cta_head_array)
        self.assertIsNone(self.segmenter.cta_neck_array)
        self.assertIsNone(self.segmenter.cta_head_affine)
        self.assertIsNone(self.segmenter.segmentation_head_array)
        self.assertIsNone(self.segmenter.segmentation_neck_array)

    def test_extracranial_vessels_fast_segmentation(self):
        self.segmenter.mode = "extracranial_vessels"
        self.segmenter.fast_segmentation = True
        if os.path.exists(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz")):
            os.remove(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz"))
        self.segmenter.segmentation_nifti = None
        self.segmenter.segmentation_array = None

        self.segmenter.segment_vessels_from_cta()

        self.assertIsNotNone(self.segmenter.segmentation_nifti)
        self.assertIsNotNone(self.segmenter.segmentation_array)
        self.assertTrue(os.path.exists(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz")))

    def test_extracranial_vessels_segmentation(self):
        self.segmenter.set_mode("extracranial_vessels")
        self.segmenter.set_segmentation_nifti_path(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz"))
        self.segmenter.fast_segmentation = False
        if os.path.exists(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz")):
            os.remove(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz"))
        self.segmenter.segmentation_nifti = None
        self.segmenter.segmentation_array = None
        self.segmenter.cta_head_array = None
        self.segmenter.cta_neck_array = None
        self.segmenter.cta_head_affine = None
        self.segmenter.segmentation_head_array = None
        self.segmenter.segmentation_neck_array = None

        self.segmenter.segment_vessels_from_cta()

        self.assertIsNotNone(self.segmenter.segmentation_nifti)
        self.assertIsNotNone(self.segmenter.segmentation_array)
        self.assertTrue(os.path.exists(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz")))
        self.assertIsNotNone(self.segmenter.cta_head_array)
        self.assertIsNotNone(self.segmenter.cta_neck_array)
        self.assertIsNotNone(self.segmenter.cta_head_affine)
        self.assertIsNotNone(self.segmenter.segmentation_head_array)
        self.assertIsNotNone(self.segmenter.segmentation_neck_array)

    def test_intracranial_vessels_segmentation(self):
        self.segmenter.set_mode("intracranial_vessels")
        self.segmenter.set_segmentation_nifti_path(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz"))
        if os.path.exists(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz")):
            os.remove(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz"))
        self.segmenter.segmentation_nifti = None
        self.segmenter.segmentation_array = None
        self.segmenter.cta_head_array = None
        self.segmenter.cta_neck_array = None
        self.segmenter.cta_head_affine = None
        self.segmenter.segmentation_head_array = None
        self.segmenter.segmentation_neck_array = None

        self.segmenter.segment_vessels_from_cta()

        self.assertIsNotNone(self.segmenter.segmentation_nifti)
        self.assertIsNotNone(self.segmenter.segmentation_array)
        self.assertTrue(os.path.exists(os.path.join(self.segmenter.case_dir, self.segmenter.mode, "segmentation.nii.gz")))
        self.assertIsNotNone(self.segmenter.cta_head_array)
        self.assertIsNotNone(self.segmenter.cta_neck_array)
        self.assertIsNotNone(self.segmenter.cta_head_affine)

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