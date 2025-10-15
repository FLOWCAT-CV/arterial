import unittest
import os, shutil
import numpy as np
from arterial.landmark_detection.landmark_detector import LandmarkDetector

class TestLandmarkDetector(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.mode = "extracranial_vessels"
        self.cta_nifti_path = os.path.join(self.case_dir, "input_test_data", "cta.nii.gz")
        self.landmark_detector = LandmarkDetector(self.case_dir, self.mode, self.cta_nifti_path)

    def test_init(self):
        self.assertEqual(self.landmark_detector.case_dir, self.case_dir)
        self.assertEqual(self.landmark_detector.mode, self.mode)
        self.assertEqual(self.landmark_detector.cta_nifti_path, self.cta_nifti_path)
        self.assertIsNone(self.landmark_detector.cta_nifti)
        self.assertIsNone(self.landmark_detector.cta_array)
        self.assertIsNone(self.landmark_detector.cta_affine)
        self.assertIsNone(self.landmark_detector.landmarks_ras_mm_dict)
        self.assertIsNone(self.landmark_detector.landmarks_slicer_json)
        self.assertIsNone(self.landmark_detector.predicted_mask_nifti)
        self.assertEqual(self.landmark_detector.landmarks_ras_json_path, 
                        os.path.join(self.case_dir, self.mode, "landmarks.json"))
        self.assertEqual(self.landmark_detector.landmarks_slicer_json_path, 
                        os.path.join(self.case_dir, self.mode, "landmarks_slicer.json"))
        self.assertEqual(self.landmark_detector.predicted_mask_nifti_path, 
                        os.path.join(self.case_dir, self.mode, "predicted_mask.nii.gz"))

    def test_load_cta_nifti(self):
        self.landmark_detector._load_cta_nifti_from_file()
        self.assertIsNotNone(self.landmark_detector.cta_nifti)
        self.assertIsNotNone(self.landmark_detector.cta_array)
        self.assertIsNotNone(self.landmark_detector.cta_affine)
        self.assertEqual(self.landmark_detector.cta_affine.shape, (4, 4))
        self.assertGreater(self.landmark_detector.cta_array.ndim, 0)

    def test_detect_landmarks_without_mask(self):
        # Remove existing files if any
        if os.path.exists(os.path.join(self.case_dir, self.mode)):
            shutil.rmtree(os.path.join(self.case_dir, self.mode))
        
        self.landmark_detector.detect_landmarks_on_cta(return_mask=False, save=True)
        
        # Check that landmarks were detected
        self.assertIsNotNone(self.landmark_detector.landmarks_ras_mm_dict)
        self.assertIsNotNone(self.landmark_detector.landmarks_slicer_json)
        self.assertIsNone(self.landmark_detector.predicted_mask_nib)
        
        # Check that landmark dict has expected structure
        expected_landmarks = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
        for landmark in expected_landmarks:
            self.assertIn(landmark, self.landmark_detector.landmarks_ras_mm_dict)
            self.assertEqual(len(self.landmark_detector.landmarks_ras_mm_dict[landmark]), 3)
        
        # Check that files were saved
        self.assertTrue(os.path.exists(self.landmark_detector.landmarks_ras_json_path))
        self.assertTrue(os.path.exists(self.landmark_detector.landmarks_slicer_json_path))
        self.assertFalse(os.path.exists(self.landmark_detector.predicted_mask_nifti_path))

    def test_detect_landmarks_with_mask(self):
        # Reset landmarks
        self.landmark_detector.landmarks_ras_mm_dict = None
        self.landmark_detector.landmarks_slicer_json = None
        self.landmark_detector.predicted_mask_nifti = None
        
        # Remove existing files if any
        if os.path.exists(os.path.join(self.case_dir, self.mode)):
            shutil.rmtree(os.path.join(self.case_dir, self.mode))
        
        self.landmark_detector.detect_landmarks_on_cta(return_mask=True, save=True)
        
        # Check that landmarks and mask were detected
        self.assertIsNotNone(self.landmark_detector.landmarks_ras_mm_dict)
        self.assertIsNotNone(self.landmark_detector.landmarks_slicer_json)
        self.assertIsNotNone(self.landmark_detector.predicted_mask_nib)
        
        # Check that all files were saved including mask
        self.assertTrue(os.path.exists(self.landmark_detector.landmarks_ras_json_path))
        self.assertTrue(os.path.exists(self.landmark_detector.landmarks_slicer_json_path))
        self.assertTrue(os.path.exists(self.landmark_detector.predicted_mask_nifti_path))

    def test_detect_landmarks_without_save(self):
        # Reset landmarks
        self.landmark_detector.landmarks_ras_mm_dict = None
        self.landmark_detector.landmarks_slicer_json = None
        
        # Remove existing directory if any
        if os.path.exists(os.path.join(self.case_dir, self.mode)):
            shutil.rmtree(os.path.join(self.case_dir, self.mode))
        
        self.landmark_detector.detect_landmarks_on_cta(return_mask=False, save=False)
        
        # Check that landmarks were detected
        self.assertIsNotNone(self.landmark_detector.landmarks_ras_mm_dict)
        self.assertIsNotNone(self.landmark_detector.landmarks_slicer_json)
        
        # Check that no files were created (directory shouldn't exist or should be empty)
        if os.path.exists(os.path.join(self.case_dir, self.mode)):
            files = os.listdir(os.path.join(self.case_dir, self.mode))
            self.assertEqual(len(files), 0, "No files should be created when save=False")

    def test_landmark_coordinates_validity(self):
        # Run detection
        if self.landmark_detector.landmarks_ras_mm_dict is None:
            self.landmark_detector.detect_landmarks_on_cta(return_mask=False, save=False)
        
        # Check that all coordinates are finite numbers
        for landmark, coords in self.landmark_detector.landmarks_ras_mm_dict.items():
            self.assertEqual(len(coords), 3, f"Landmark {landmark} should have 3 coordinates")
            for coord in coords:
                self.assertTrue(np.isfinite(coord), 
                              f"Landmark {landmark} has non-finite coordinate: {coord}")
            
            # Check that no landmark is at origin (0, 0, 0) - would indicate failure
            distance_from_origin = np.linalg.norm(coords)
            self.assertGreater(distance_from_origin, 1.0,
                             f"Landmark {landmark} too close to origin: {coords}")

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
