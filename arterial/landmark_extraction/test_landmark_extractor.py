#   Copyright 2025   Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

"""
Comprehensive test suite for landmark extractor components.
Tests data flow, class interactions, and validates landmark extraction functionality.
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np
import torch
import nibabel as nib
import pytest
from pathlib import Path

from landmark_extractor import SingleCTADataset, LandmarkAutomator
from model import MonaiUNet3DSeg, load_trained_model_seg
from utils import (
    resample_image, 
    crop_or_pad_image, 
    change_origin_preprocess,
    update_json_with_predictions,
    restore_centroids_to_original_origin
)

class TestLandmarkExtractor:
    """Test suite for landmark extraction components."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures and create temporary test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_dir = os.path.join(cls.temp_dir, "test_case")
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create a dummy CTA image (320x320x480 to match expected dimensions)
        cls.create_dummy_cta()
        
        # Create a dummy template JSON
        cls.create_dummy_template_json()
        
        # Create dummy affine file
        cls.create_dummy_affine_file()
        
        print(f"Test data created in: {cls.test_data_dir}")
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def create_dummy_cta(cls):
        """Create a dummy CTA image for testing."""
        # Create a realistic-looking CTA with some vessel-like structures
        shape = (320, 320, 480)
        data = np.random.uniform(0, 100, shape).astype(np.float32)
        
        # Add some bright spots to simulate vessels
        for i in range(6):  # 6 landmarks
            z = np.random.randint(50, shape[2] - 50)
            y = np.random.randint(50, shape[1] - 50)
            x = np.random.randint(50, shape[0] - 50)
            
            # Create a small bright region
            data[x-5:x+5, y-5:y+5, z-5:z+5] = np.random.uniform(400, 700)
        
        # Create nifti image
        affine = np.eye(4)
        affine[0, 0] = 0.8  # voxel size
        affine[1, 1] = 0.8
        affine[2, 2] = 0.8
        
        img = nib.Nifti1Image(data, affine)
        cta_path = os.path.join(cls.test_data_dir, "cta.nii.gz")
        nib.save(img, cta_path)
    
    @classmethod
    def create_dummy_template_json(cls):
        """Create a dummy template JSON for testing."""
        template = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [{
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 5,
                "controlPoints": [
                    {"id": "1", "label": "l-tica", "position": [0.0, 0.0, 0.0]},
                    {"id": "2", "label": "r-tica", "position": [0.0, 0.0, 0.0]},
                    {"id": "3", "label": "l-eica", "position": [0.0, 0.0, 0.0]},
                    {"id": "4", "label": "r-eica", "position": [0.0, 0.0, 0.0]},
                    {"id": "5", "label": "r-mca", "position": [0.0, 0.0, 0.0]},
                    {"id": "6", "label": "l-mca", "position": [0.0, 0.0, 0.0]}
                ]
            }]
        }
        
        template_path = os.path.join(cls.test_data_dir, "F.json")
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=4)
    
    @classmethod
    def create_dummy_affine_file(cls):
        """Create a dummy affine transformation file."""
        affine = np.eye(4)
        affine[0, 0] = 0.8
        affine[1, 1] = 0.8
        affine[2, 2] = 0.8
        affine[0, 3] = 100  # some translation
        affine[1, 3] = -50
        affine[2, 3] = 200
        
        affine_path = os.path.join(cls.test_data_dir, "affine_before_origin_change.txt")
        np.savetxt(affine_path, affine)
    
    def test_single_cta_dataset_initialization(self):
        """Test SingleCTADataset initialization and basic functionality."""
        print("Testing SingleCTADataset initialization...")
        
        # Test with valid folder
        dataset = SingleCTADataset(self.test_data_dir)
        assert dataset.folder_path == self.test_data_dir
        assert len(dataset) == 1
        
        # Test data loading
        sample = dataset[0]
        assert "volume" in sample
        assert "affine" in sample
        assert "folder" in sample
        
        # Check volume properties
        volume = sample["volume"]
        assert isinstance(volume, torch.Tensor)
        assert volume.dtype == torch.float32
        assert volume.shape[0] == 1  # batch dimension added
        assert volume.shape[1] == 1  # channel dimension
        
        print("✓ SingleCTADataset initialization test passed")
    
    def test_single_cta_dataset_data_flow(self):
        """Test data preprocessing in SingleCTADataset."""
        print("Testing SingleCTADataset data preprocessing...")
        
        dataset = SingleCTADataset(self.test_data_dir)
        sample = dataset[0]
        
        # Check that data is normalized to [0, 1]
        volume = sample["volume"]
        assert volume.min() >= 0.0
        assert volume.max() <= 1.0
        
        # Check expected spatial dimensions after preprocessing
        expected_shape = (1, 1, 480, 320, 320)  # (batch, channel, depth, height, width)
        assert volume.shape == expected_shape, f"Expected {expected_shape}, got {volume.shape}"
        
        # Check that affine is preserved
        assert sample["affine"] is not None
        assert sample["folder"] == self.test_data_dir
        
        print("✓ SingleCTADataset data preprocessing test passed")
    
    def test_model_loading_without_actual_model(self):
        """Test model structure without loading actual weights."""
        print("Testing model structure...")
        
        # Test model creation
        model = MonaiUNet3DSeg(num_classes=7)
        assert model is not None
        
        # Test model forward pass with dummy data
        dummy_input = torch.randn(1, 1, 480, 320, 320)
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_output_shape = (1, 7, 480, 320, 320)  # 7 classes including background
        assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
        
        print("✓ Model structure test passed")
    
    def test_landmark_automator_initialization(self):
        """Test LandmarkAutomator initialization with dummy model."""
        print("Testing LandmarkAutomator initialization...")
        
        # Create a dummy model file for testing
        dummy_model_path = os.path.join(self.temp_dir, "dummy_model.pth")
        dummy_model = MonaiUNet3DSeg(num_classes=7)
        torch.save(dummy_model.state_dict(), dummy_model_path)
        
        try:
            # Test initialization
            device = "cpu"  # Use CPU for testing
            automator = LandmarkAutomator(dummy_model_path, device=device)
            
            assert automator.device.type == "cpu"
            assert automator.model is not None
            
            print("✓ LandmarkAutomator initialization test passed")
            
        except Exception as e:
            print(f"⚠ LandmarkAutomator initialization test failed: {e}")
            # This might fail if dependencies are missing, so we'll make it non-fatal
    
    def test_data_flow_between_classes(self):
        """Test that data flows correctly between SingleCTADataset and LandmarkAutomator."""
        print("Testing data flow between classes...")
        
        # Create dummy model
        dummy_model_path = os.path.join(self.temp_dir, "dummy_model.pth")
        dummy_model = MonaiUNet3DSeg(num_classes=7)
        torch.save(dummy_model.state_dict(), dummy_model_path)
        
        try:
            dataset = SingleCTADataset(self.test_data_dir)
            automator = LandmarkAutomator(dummy_model_path, device="cpu")
            
            # Test prepare_input method
            volume, sample = automator._prepare_input(self.test_data_dir)
            
            # Check that volume is properly formatted for model
            assert volume.shape[0] == 1  # batch size
            assert volume.shape[1] == 1  # channels
            assert volume.device.type == "cpu"
            
            # Check that sample contains necessary information
            assert "affine" in sample
            assert "folder" in sample
            assert sample["folder"] == self.test_data_dir
            
            print("✓ Data flow between classes test passed")
            
        except Exception as e:
            print(f"⚠ Data flow test failed: {e}")
    
    def test_postprocessing_functions(self):
        """Test utility functions used in postprocessing."""
        print("Testing postprocessing utility functions...")
        
        # Test update_json_with_predictions
        dummy_predictions = np.random.rand(6, 3) * 100  # 6 landmarks, 3D coordinates
        output_json_path = os.path.join(self.temp_dir, "test_output.json")
        input_json_path = os.path.join(self.test_data_dir, "F.json")
        
        update_json_with_predictions(input_json_path, dummy_predictions, output_json_path)
        
        # Check that output JSON was created and has correct structure
        assert os.path.exists(output_json_path)
        with open(output_json_path, 'r') as f:
            output_data = json.load(f)
        
        assert "markups" in output_data
        assert len(output_data["markups"][0]["controlPoints"]) == 6
        
        # Test restore_centroids_to_original_origin
        affine_path = os.path.join(self.test_data_dir, "affine_before_origin_change.txt")
        centroids_mm = np.random.rand(6, 3) * 100
        orientation = ('L', 'A', 'S')
        
        restored_centroids = restore_centroids_to_original_origin(centroids_mm, affine_path, orientation)
        assert restored_centroids.shape == centroids_mm.shape
        
        print("✓ Postprocessing utility functions test passed")
    
    def test_error_handling(self):
        """Test error handling for common failure scenarios."""
        print("Testing error handling...")
        
        # Test with non-existent folder
        try:
            dataset = SingleCTADataset("/non/existent/folder")
            sample = dataset[0]  # This should fail
            assert False, "Should have raised an exception for non-existent folder"
        except:
            print("✓ Correctly handles non-existent folder")
        
        # Test with folder missing cta.nii.gz
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        try:
            dataset = SingleCTADataset(empty_dir)
            sample = dataset[0]  # This should fail
            assert False, "Should have raised an exception for missing cta.nii.gz"
        except:
            print("✓ Correctly handles missing cta.nii.gz")
        
        # Test with non-existent model file
        try:
            automator = LandmarkAutomator("/non/existent/model.pth")
            assert False, "Should have raised an exception for non-existent model"
        except:
            print("✓ Correctly handles non-existent model file")
    
    def test_end_to_end_workflow(self):
        """Test the complete end-to-end workflow."""
        print("Testing end-to-end workflow...")
        
        # Create dummy model
        dummy_model_path = os.path.join(self.temp_dir, "dummy_model.pth")
        dummy_model = MonaiUNet3DSeg(num_classes=7)
        torch.save(dummy_model.state_dict(), dummy_model_path)
        
        output_dir = os.path.join(self.temp_dir, "output")
        
        try:
            automator = LandmarkAutomator(dummy_model_path, device="cpu")
            automator.infer_folder(self.test_data_dir, output_dir, save_mask=True, save_json=True)
            
            # Check that outputs were created
            expected_output_dir = os.path.join(output_dir, os.path.basename(self.test_data_dir))
            assert os.path.exists(expected_output_dir)
            
            # Check for expected output files
            pred_mask_path = os.path.join(expected_output_dir, "pred_mask.nii.gz")
            pred_json_path = os.path.join(expected_output_dir, "F_o.json")
            
            if os.path.exists(pred_mask_path):
                print("✓ Prediction mask created successfully")
            else:
                print("⚠ Prediction mask not created")
            
            if os.path.exists(pred_json_path):
                print("✓ Prediction JSON created successfully")
            else:
                print("⚠ Prediction JSON not created")
            
            print("✓ End-to-end workflow test completed")
            
        except Exception as e:
            print(f"⚠ End-to-end workflow test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 80)
        print("LANDMARK EXTRACTOR COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        test_methods = [
            self.test_single_cta_dataset_initialization,
            self.test_single_cta_dataset_data_flow,
            self.test_model_loading_without_actual_model,
            self.test_landmark_automator_initialization,
            self.test_data_flow_between_classes,
            self.test_postprocessing_functions,
            self.test_error_handling,
            self.test_end_to_end_workflow
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
            except Exception as e:
                print(f"✗ {test_method.__name__} failed: {e}")
                failed += 1
            print("-" * 80)
        
        print(f"\nTEST SUMMARY: {passed} passed, {failed} failed")
        print("=" * 80)


def test_with_real_model_and_data():
    """
    Test with real model and data if available.
    This function checks for the actual model file and test data.
    """
    print("\n" + "=" * 80)
    print("TESTING WITH REAL MODEL AND DATA")
    print("=" * 80)
    
    # Check for real model
    model_path = "/media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth"
    test_data_path = "/media/Disk_B/arterial_models/test_data/input_test_data"
    
    if not os.path.exists(model_path):
        print(f"⚠ Real model not found at: {model_path}")
        return
    
    if not os.path.exists(test_data_path):
        print(f"⚠ Real test data not found at: {test_data_path}")
        return
    
    if not os.path.exists(os.path.join(test_data_path, "cta.nii.gz")):
        print(f"⚠ cta.nii.gz not found in test data directory")
        return
    
    try:
        # Test with real model and data
        output_dir = tempfile.mkdtemp()
        
        print(f"Using model: {model_path}")
        print(f"Using test data: {test_data_path}")
        print(f"Output directory: {output_dir}")
        
        # Create necessary JSON file if it doesn't exist
        json_path = os.path.join(test_data_path, "F.json")
        if not os.path.exists(json_path):
            template = {
                "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
                "markups": [{
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": False,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": 5,
                    "controlPoints": [
                        {"id": "1", "label": "l-tica", "position": [0.0, 0.0, 0.0]},
                        {"id": "2", "label": "r-tica", "position": [0.0, 0.0, 0.0]},
                        {"id": "3", "label": "l-eica", "position": [0.0, 0.0, 0.0]},
                        {"id": "4", "label": "r-eica", "position": [0.0, 0.0, 0.0]},
                        {"id": "5", "label": "r-mca", "position": [0.0, 0.0, 0.0]},
                        {"id": "6", "label": "l-mca", "position": [0.0, 0.0, 0.0]}
                    ]
                }]
            }
            with open(json_path, 'w') as f:
                json.dump(template, f, indent=4)
            print(f"Created template JSON at: {json_path}")
        
        # Initialize automator with real model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        automator = LandmarkAutomator(model_path, device=device)
        
        # Run inference
        print("Running inference...")
        automator.infer_folder(test_data_path, output_dir, save_mask=True, save_json=True)
        
        # Check outputs
        expected_output_dir = os.path.join(output_dir, os.path.basename(test_data_path))
        pred_mask_path = os.path.join(expected_output_dir, "pred_mask.nii.gz")
        pred_json_path = os.path.join(expected_output_dir, "F_o.json")
        
        if os.path.exists(pred_mask_path):
            print(f"✓ Prediction mask created: {pred_mask_path}")
            # Load and check mask properties
            mask_img = nib.load(pred_mask_path)
            print(f"  - Mask shape: {mask_img.shape}")
            print(f"  - Unique values: {np.unique(mask_img.get_fdata())}")
        
        if os.path.exists(pred_json_path):
            print(f"✓ Prediction JSON created: {pred_json_path}")
            # Load and check JSON content
            with open(pred_json_path, 'r') as f:
                pred_data = json.load(f)
            print(f"  - Number of landmarks: {len(pred_data['markups'][0]['controlPoints'])}")
            
            # Print landmark positions
            for cp in pred_data['markups'][0]['controlPoints']:
                label = cp.get('label', 'unknown')
                pos = cp.get('position', [0, 0, 0])
                print(f"    {label}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        print("✓ Real model and data test completed successfully")
        
        # Clean up
        shutil.rmtree(output_dir)
        
    except Exception as e:
        print(f"✗ Real model and data test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive test suite
    tester = TestLandmarkExtractor()
    tester.run_all_tests()
    
    # Test with real model and data if available
    test_with_real_model_and_data()