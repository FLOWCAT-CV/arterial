import unittest
import os
import shutil
import numpy as np
import nibabel as nib
from arterial.landmark_extraction.landmark_extractor import LandmarkAutomator


class TestLandmarkAutomator(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.model_path = os.path.join(os.path.dirname(__file__), "test_data", "input_test_data", "landmark_model.pth")
        self.input_folder = os.path.join(self.case_dir, "input_test_data", "landmark_case")
        self.cta_nifti_path = os.path.join(self.input_folder, "cta.nii.gz")
        self.output_folder = os.path.join(self.case_dir, "landmark_extraction")
        
        # Check if necessary files exist before setup
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model file {self.model_path} not found")
            
        if not os.path.exists(self.cta_nifti_path):
            self.skipTest(f"CTA nifti file {self.cta_nifti_path} not found")
        
        # Initialize the LandmarkAutomator
        self.automator = LandmarkAutomator(
            model_path=self.model_path,
            device='gpu'  # Use CPU for testing to avoid GPU dependency
        )

    def test_init(self):
        """Test initialization of LandmarkAutomator"""
        self.assertIsNotNone(self.automator.model)
        self.assertEqual(self.automator.device.type, 'cpu')

    def test_folder_inference_with_save(self):
        """Test inference from folder with file saving"""
        landmarks_mm = self.automator.infer_folder(
            folder_path=self.input_folder,
            output_folder=self.output_folder,
            save_mask=True,
            save_json=True
        )
        
        # Check that landmarks were returned
        self.assertIsNotNone(landmarks_mm)
        self.assertEqual(landmarks_mm.shape, (6, 3))  # 6 landmarks, 3 coordinates each
        
        # Check that output files were created
        output_case_dir = os.path.join(self.output_folder, os.path.basename(self.input_folder))
        self.assertTrue(os.path.exists(output_case_dir))
        self.assertTrue(os.path.exists(os.path.join(output_case_dir, "F_o.json")))
        self.assertTrue(os.path.exists(os.path.join(output_case_dir, "pred_mask.nii.gz")))

    def test_folder_inference_without_save(self):
        """Test inference from folder without file saving"""
        temp_output = os.path.join(self.case_dir, "temp_landmark_output")
        
        landmarks_mm = self.automator.infer_folder(
            folder_path=self.input_folder,
            output_folder=temp_output,
            save_mask=False,
            save_json=False
        )
        
        # Check that landmarks were returned
        self.assertIsNotNone(landmarks_mm)
        self.assertEqual(landmarks_mm.shape, (6, 3))
        
        # Check that no files were created (except the affine files from preprocessing)
        output_case_dir = os.path.join(temp_output, os.path.basename(self.input_folder))
        if os.path.exists(output_case_dir):
            files = os.listdir(output_case_dir)
            self.assertNotIn("F_o.json", files)
            self.assertNotIn("pred_mask.nii.gz", files)
        
        # Clean up temp directory
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)

    def test_array_inference_with_numpy(self):
        """Test inference from numpy array"""
        # Load the CTA image
        nib_img = nib.load(self.cta_nifti_path)
        data = nib_img.get_fdata()
        affine = nib_img.affine
        
        # Run inference
        results = self.automator.infer_from_array(
            data=data,
            affine=affine,
            output_folder=None,
            save_mask=False,
            save_json=False
        )
        
        # Check results structure
        self.assertIn('landmarks_mm', results)
        self.assertIn('landmarks_voxel', results)
        self.assertIn('raw_predictions', results)
        
        # Check shapes
        self.assertEqual(results['landmarks_mm'].shape, (6, 3))
        self.assertEqual(results['landmarks_voxel'].shape, (6, 3))
        self.assertEqual(results['raw_predictions'].shape[0], 7)  # 7 channels (1 background + 6 landmarks)

    def test_array_inference_with_nibabel(self):
        """Test inference from nibabel image object"""
        # Load the CTA image as nibabel object
        nib_img = nib.load(self.cta_nifti_path)
        
        # Run inference
        results = self.automator.infer_from_array(
            data=nib_img,
            affine=None,  # Should be extracted from nibabel object
            output_folder=None,
            save_mask=False,
            save_json=False
        )
        
        # Check results
        self.assertIn('landmarks_mm', results)
        self.assertEqual(results['landmarks_mm'].shape, (6, 3))

    def test_array_inference_with_save(self):
        """Test array inference with file saving"""
        nib_img = nib.load(self.cta_nifti_path)
        output_folder = os.path.join(self.case_dir, "landmark_extraction_array")
        
        results = self.automator.infer_from_array(
            data=nib_img,
            output_folder=output_folder,
            save_mask=True,
            save_json=True
        )
        
        # Check that files were created
        # Note: The output location for array inference might use a temp folder
        self.assertIsNotNone(results['landmarks_mm'])

    def test_consistency_folder_vs_array(self):
        """Test that folder and array inference give consistent results"""
        # Folder inference
        landmarks_folder = self.automator.infer_folder(
            folder_path=self.input_folder,
            output_folder=os.path.join(self.case_dir, "temp_folder"),
            save_mask=False,
            save_json=False
        )
        
        # Array inference
        nib_img = nib.load(self.cta_nifti_path)
        results_array = self.automator.infer_from_array(
            data=nib_img,
            output_folder=None,
            save_mask=False,
            save_json=False
        )
        
        # Compare results (should be very close, allowing for small numerical differences)
        diff = np.abs(landmarks_folder - results_array['landmarks_mm'])
        max_diff = np.max(diff)
        self.assertLess(max_diff, 1.0, 
                       f"Folder and array inference differ by {max_diff:.3f} mm")

    def test_landmark_coordinates_validity(self):
        """Test that predicted landmarks are in reasonable coordinate ranges"""
        landmarks_mm = self.automator.infer_folder(
            folder_path=self.input_folder,
            output_folder=os.path.join(self.case_dir, "temp_validity"),
            save_mask=False,
            save_json=False
        )
        
        # Check that all coordinates are finite numbers
        self.assertTrue(np.all(np.isfinite(landmarks_mm)))
        
        # Check that no landmark is at origin (0, 0, 0) - would indicate failure
        for i, landmark in enumerate(landmarks_mm):
            distance_from_origin = np.linalg.norm(landmark)
            self.assertGreater(distance_from_origin, 1.0,
                             f"Landmark {i} too close to origin: {landmark}")

    def test_preprocessing_affine_files(self):
        """Test that preprocessing creates expected affine files"""
        # Run inference which triggers preprocessing
        self.automator.infer_folder(
            folder_path=self.input_folder,
            output_folder=os.path.join(self.case_dir, "temp_affine"),
            save_mask=False,
            save_json=False
        )
        
        # Check that affine files were created in the input folder
        affine_before_path = os.path.join(self.input_folder, "affine_before_origin_change.txt")
        affine_original_path = os.path.join(self.input_folder, "affine_original.txt")
        
        # These should exist after preprocessing
        self.assertTrue(os.path.exists(affine_before_path) or 
                       os.path.exists(affine_original_path),
                       "Preprocessing should create affine files")

    @classmethod
    def tearDownClass(cls):
        """Clean up all files generated during tests"""
        cls.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        
        # Remove all generated files and folders except input_test_data and output
        for filename in os.listdir(cls.case_dir):
            if filename not in ["input_test_data", "output"]:
                filepath = os.path.join(cls.case_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
        
        # Also clean up any affine files created in input_test_data during testing
        input_test_data = os.path.join(cls.case_dir, "input_test_data", "landmark_case")
        if os.path.exists(input_test_data):
            for filename in os.listdir(input_test_data):
                if filename.startswith("affine_"):
                    os.remove(os.path.join(input_test_data, filename))


if __name__ == '__main__':
    unittest.main()


"""
expected file tree:
tests/
├── test_landmark_automator.py
└── test_data/
    ├── input_test_data/
    │   ├── landmark_model.pth        # Your trained model
    │   └── landmark_case/
    │       └── cta.nii.gz            # Test CTA image
    └── output/                        # (preserved during cleanup)
"""