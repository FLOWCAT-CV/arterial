#!/usr/bin/env python3

import os
import sys
import logging
import torch
import argparse
from pathlib import Path

# Add the parent directory to Python path to import modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from landmark_extractor import LandmarkAutomator, SingleCTADataset
from utils import postprocess_heatmaps
from template_embedded import load_template_from_embedded, save_template_to_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_landmark_extraction(cta_folder_path: str, model_path: str, output_folder: str):
    """
    Test landmark extraction with flexible paths.
    
    Args:
        cta_folder_path (str): Path to folder containing CTA file (cta.nii.gz)
        model_path (str): Path to the trained model (.pth file)
        output_folder (str): Path to save output results
    
    Returns:
        bool: True if test successful, False otherwise
    """
    logger.info("Testing landmark extraction with custom paths...")
    logger.info(f"CTA folder: {cta_folder_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output folder: {output_folder}")
    
    # Validate input paths
    if not os.path.exists(cta_folder_path):
        logger.error(f"CTA folder does not exist: {cta_folder_path}")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        return False
    
    # Check for CTA file in the folder
    cta_file = os.path.join(cta_folder_path, "cta.nii.gz")
    if not os.path.exists(cta_file):
        logger.error(f"CTA file not found: {cta_file}")
        return False
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_folder}")
    
    try:
        # Initialize components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Test LandmarkAutomator initialization
        logger.info("Initializing LandmarkAutomator...")
        automator = LandmarkAutomator(model_path, device)
        logger.info("✓ LandmarkAutomator initialized successfully")
        
        # Test SingleCTADataset initialization
        logger.info("Initializing SingleCTADataset...")
        dataset = SingleCTADataset(cta_folder_path)
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            logger.error("No data found in the CTA folder")
            return False
        
        # Process the CTA data
        logger.info("Processing CTA data...")
        sample = dataset[0]
        logger.info(f"✓ Loaded sample with shape: {sample['image'].shape}")
        
        # Prepare input for model
        image_batch = sample['image'].unsqueeze(0).to(device)
        logger.info(f"✓ Prepared model input with shape: {image_batch.shape}")
        
        # Run inference
        logger.info("Running model inference...")
        with torch.no_grad():
            output = automator.model(image_batch)
        
        logger.info(f"✓ Model inference completed. Output shape: {output.shape}")
        logger.info(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Post-process results
        logger.info("Post-processing results...")
        heatmaps = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
        landmarks = postprocess_heatmaps(heatmaps)
        logger.info(f"✓ Extracted {len(landmarks)} landmarks")
        
        # Display landmarks
        landmark_labels = ['r-tica', 'l-tica', 'r-eica', 'l-eica', 'l-mca', 'r-mca']
        for i, landmark in enumerate(landmarks):
            label = landmark_labels[i] if i < len(landmark_labels) else f"landmark_{i}"
            logger.info(f"  {label}: {landmark}")
        
        # Save results
        logger.info("Saving results...")
        
        # Save landmarks as text file
        landmarks_file = os.path.join(output_folder, "landmarks.txt")
        with open(landmarks_file, 'w') as f:
            f.write("# Extracted Landmarks\n")
            f.write("# Format: label: [x, y, z]\n\n")
            for i, landmark in enumerate(landmarks):
                label = landmark_labels[i] if i < len(landmark_labels) else f"landmark_{i}"
                f.write(f"{label}: {landmark}\n")
        logger.info(f"✓ Landmarks saved to: {landmarks_file}")
        
        # Save heatmaps (optional)
        heatmaps_file = os.path.join(output_folder, "heatmaps.pt")
        torch.save(heatmaps, heatmaps_file)
        logger.info(f"✓ Heatmaps saved to: {heatmaps_file}")
        
        # Create template JSON file using embedded template
        logger.info("Creating template JSON from embedded data...")
        template_file = os.path.join(output_folder, "template.json")
        save_template_to_file(template_file)
        logger.info(f"✓ Template saved to: {template_file}")
        
        logger.info("🎉 Landmark extraction test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during landmark extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_cta_dataset():
    """Test the SingleCTADataset class."""
    logger.info("Testing SingleCTADataset...")
    
    # Test data path (adjust as needed)
    test_folder = "/media/Disk_B/arterial_models/test_data/input_test_data"
    
    if not os.path.exists(test_folder):
        logger.warning(f"Test folder {test_folder} does not exist. Creating a mock test...")
        return False
    
    try:
        # Create dataset
        dataset = SingleCTADataset(test_folder)
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test data loading
            sample = dataset[0]
            logger.info(f"Sample keys: {sample.keys()}")
            logger.info(f"Image shape: {sample['image'].shape}")
            logger.info(f"Image dtype: {sample['image'].dtype}")
            logger.info(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            
            return True
        else:
            logger.warning("Dataset is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error testing SingleCTADataset: {e}")
        return False


def test_landmark_automator():
    """Test the LandmarkAutomator class."""
    logger.info("Testing LandmarkAutomator...")
    
    # Model path (adjust as needed)
    model_path = "/media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth"
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file {model_path} does not exist. Cannot test LandmarkAutomator.")
        return False
    
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create automator
        automator = LandmarkAutomator(model_path, device)
        logger.info("LandmarkAutomator created successfully")
        
        # Check model properties
        logger.info(f"Model device: {next(automator.model.parameters()).device}")
        logger.info(f"Model in eval mode: {not automator.model.training}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing LandmarkAutomator: {e}")
        return False


def test_end_to_end_pipeline():
    """Test the complete landmark extraction pipeline."""
    logger.info("Testing end-to-end pipeline...")
    
    # Paths (adjust as needed)
    test_folder = "/media/Disk_B/arterial_models/test_data/input_test_data"
    model_path = "/media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth"
    output_folder = "/media/Disk_B/arterial_models/test_data/output"
    
    # Check if required files exist
    if not os.path.exists(test_folder):
        logger.warning(f"Test folder {test_folder} does not exist.")
        return False
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file {model_path} does not exist.")
        return False
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        automator = LandmarkAutomator(model_path, device)
        dataset = SingleCTADataset(test_folder)
        
        if len(dataset) == 0:
            logger.warning("No data found in test folder")
            return False
        
        logger.info(f"Processing {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
            # Add batch dimension
            image_batch = sample['image'].unsqueeze(0).to(device)
            logger.info(f"Input batch shape: {image_batch.shape}")
            
            # Run inference
            with torch.no_grad():
                output = automator.model(image_batch)
            
            logger.info(f"Model output shape: {output.shape}")
            logger.info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Post-process results
            heatmaps = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
            landmarks = postprocess_heatmaps(heatmaps)
            
            logger.info(f"Extracted {len(landmarks)} landmarks")
            for j, landmark in enumerate(landmarks):
                logger.info(f"  Landmark {j}: {landmark}")
            
            # Save results (you can customize this part)
            output_file = os.path.join(output_folder, f"landmarks_sample_{i}.txt")
            with open(output_file, 'w') as f:
                for j, landmark in enumerate(landmarks):
                    f.write(f"Landmark {j}: {landmark}\n")
            
            logger.info(f"Results saved to {output_file}")
        
        logger.info("End-to-end pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in end-to-end pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_flow():
    """Test data flow between classes to ensure proper information passing."""
    logger.info("Testing data flow between classes...")
    
    test_folder = "/media/Disk_B/arterial_models/test_data/input_test_data"
    model_path = "/media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth"
    
    if not os.path.exists(test_folder) or not os.path.exists(model_path):
        logger.warning("Required files not found for data flow test")
        return False
    
    try:
        # Test 1: Dataset creation and data access
        logger.info("Test 1: Dataset creation and access")
        dataset = SingleCTADataset(test_folder)
        
        if len(dataset) == 0:
            logger.warning("No data in dataset")
            return False
        
        sample = dataset[0]
        original_shape = sample['image'].shape
        original_dtype = sample['image'].dtype
        logger.info(f"Original image - Shape: {original_shape}, Dtype: {original_dtype}")
        
        # Test 2: Model initialization and device handling
        logger.info("Test 2: Model initialization")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        automator = LandmarkAutomator(model_path, device)
        
        model_device = next(automator.model.parameters()).device
        logger.info(f"Model loaded on device: {model_device}")
        
        # Test 3: Data preprocessing and model input
        logger.info("Test 3: Data preprocessing and model input")
        image_batch = sample['image'].unsqueeze(0).to(device)
        input_shape = image_batch.shape
        input_device = image_batch.device
        logger.info(f"Model input - Shape: {input_shape}, Device: {input_device}")
        
        # Test 4: Model inference
        logger.info("Test 4: Model inference")
        with torch.no_grad():
            output = automator.model(image_batch)
        
        output_shape = output.shape
        output_device = output.device
        logger.info(f"Model output - Shape: {output_shape}, Device: {output_device}")
        
        # Test 5: Post-processing
        logger.info("Test 5: Post-processing")
        heatmaps = output.squeeze(0).cpu()
        landmarks = postprocess_heatmaps(heatmaps)
        logger.info(f"Post-processed landmarks: {len(landmarks)} points")
        
        # Verify data consistency
        logger.info("Data flow verification:")
        logger.info(f"  ✓ Dataset loads image with shape {original_shape}")
        logger.info(f"  ✓ Model processes batch with shape {input_shape}")
        logger.info(f"  ✓ Model outputs heatmaps with shape {output_shape}")
        logger.info(f"  ✓ Post-processing extracts {len(landmarks)} landmarks")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data flow test: {e}")
        return False


def test_embedded_template():
    """Test the embedded template functionality."""
    logger.info("Testing embedded template functionality...")
    
    try:
        # Test loading template from embedded data
        template_dict = load_template_from_embedded()
        logger.info("✓ Successfully loaded template from embedded data")
        
        # Verify template structure
        assert "@schema" in template_dict, "Template missing @schema"
        assert "markups" in template_dict, "Template missing markups"
        assert len(template_dict["markups"]) > 0, "Template markups is empty"
        
        markup = template_dict["markups"][0]
        assert "controlPoints" in markup, "Template missing controlPoints"
        assert len(markup["controlPoints"]) == 6, f"Expected 6 landmarks, got {len(markup['controlPoints'])}"
        
        logger.info(f"✓ Template contains {len(markup['controlPoints'])} landmarks")
        
        # Test saving template to file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            save_template_to_file(tmp_file.name)
            logger.info(f"✓ Template saved to temporary file: {tmp_file.name}")
            
            # Clean up
            os.unlink(tmp_file.name)
        
        logger.info("✓ Embedded template test passed")
        return True
        
    except Exception as e:
        logger.error(f"Error testing embedded template: {e}")
        return False


def main():
    """Run tests with command line argument support."""
    parser = argparse.ArgumentParser(description='Test landmark extractor with flexible paths')
    parser.add_argument('--cta-folder', type=str, help='Path to folder containing cta.nii.gz')
    parser.add_argument('--model-path', type=str, help='Path to model .pth file')
    parser.add_argument('--output-folder', type=str, help='Path to output folder')
    parser.add_argument('--run-all', action='store_true', help='Run all default tests')
    
    args = parser.parse_args()
    
    logger.info("Starting landmark extractor tests...")
    
    # If custom paths provided, run custom test
    if args.cta_folder and args.model_path and args.output_folder:
        logger.info("Running custom test with provided paths...")
        success = test_landmark_extraction(args.cta_folder, args.model_path, args.output_folder)
        return 0 if success else 1
    
    # Otherwise run standard tests
    tests = [
        ("Embedded Template", test_embedded_template),
        ("SingleCTADataset", test_single_cta_dataset),
        ("LandmarkAutomator", test_landmark_automator),
        ("Data Flow", test_data_flow),
    ]
    
    if args.run_all:
        tests.append(("End-to-End Pipeline", test_end_to_end_pipeline))
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("All tests passed! ✓")
        return 0
    else:
        logger.warning(f"{total_tests - passed_tests} test(s) failed! ✗")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)