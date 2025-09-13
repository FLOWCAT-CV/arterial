#!/usr/bin/env python3
"""
Comprehensive Arterial Module Test Script

This script initializes and tests all main classes from every module in the arterial package.
It demonstrates how to set up each module with appropriate input and output folders.

The script tests the following modules:
1. Landmark Extraction (LandmarkAutomator)
2. Vessel Segmentation (VesselSegmenter) 
3. Centerline Extraction (CenterlineExtractor)
4. Vessel Labelling (VesselLabeller)
5. Feature Extraction (FeatureExtractor)
6. Access Prediction (AccessPredictor)
7. Full Pipeline (ArterialProcessor)

Usage:
    python test_arterial_modules.py --input_folder /path/to/input --output_folder /path/to/output [--model_path /path/to/model.pth]
"""

import os
import sys
import argparse
import shutil
import json
import numpy as np
from datetime import datetime

# Add the parent directory to the path to import arterial modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all main classes
from landmark_extraction.landmark_extractor import LandmarkAutomator
from segmentation.segmenter import VesselSegmenter
from centerline_extraction.centerline_extractor import CenterlineExtractor
from vessel_labelling.vessel_labeller import VesselLabeller
from feature_extraction.feature_extractor import FeatureExtractor
from access_prediction.access_predictor import AccessPredictor
from run.processor import ArterialProcessor


class ArterialModuleTester:
    """
    Class to test all arterial modules with initialization and basic functionality checks.
    """
    
    def __init__(self, input_folder, output_folder, model_path=None):
        """
        Initialize the tester with input and output folders.
        
        Parameters:
        -----------
        input_folder : str
            Path to input folder containing test data
        output_folder : str
            Path to output folder for results
        model_path : str, optional
            Path to model file for landmark extraction
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path
        self.test_results = {}
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Setup test case directory
        self.test_case_dir = os.path.join(self.output_folder, "test_case")
        os.makedirs(self.test_case_dir, exist_ok=True)
        
        print("=" * 80)
        print("ARTERIAL MODULES COMPREHENSIVE TEST")
        print("=" * 80)
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Test case directory: {self.test_case_dir}")
        if self.model_path:
            print(f"Model path: {self.model_path}")
        print("=" * 80)
    
    def setup_test_data(self):
        """
        Setup test data by copying/creating necessary files.
        """
        print("\n📁 Setting up test data...")
        
        # Copy CTA file if it exists
        cta_input = os.path.join(self.input_folder, "cta.nii.gz")
        cta_output = os.path.join(self.test_case_dir, "cta.nii.gz")
        
        if os.path.exists(cta_input):
            shutil.copy(cta_input, cta_output)
            print(f"✓ Copied CTA file to {cta_output}")
        else:
            print(f"⚠️  No CTA file found at {cta_input}")
        
        # Create template JSON for landmarks
        template_path = os.path.join(os.path.dirname(__file__), "..", "landmark_extraction", "template.json")
        f_json_path = os.path.join(self.test_case_dir, "F.json")
        
        if os.path.exists(template_path):
            shutil.copy(template_path, f_json_path)
            print(f"✓ Created F.json template")
        
        # Create dummy affine file
        affine_path = os.path.join(self.test_case_dir, "affine_before_origin_change.txt")
        identity_affine = np.eye(4)
        np.savetxt(affine_path, identity_affine)
        print(f"✓ Created affine transformation file")
        
        return os.path.exists(cta_output)
    
    def test_landmark_extraction(self):
        """Test the LandmarkAutomator class."""
        print("\n🎯 Testing Landmark Extraction Module...")
        
        try:
            if not self.model_path:
                print("⚠️  No model path provided, skipping actual inference")
                # Just test initialization without model
                print("  - Testing class import and basic structure")
                print("  - LandmarkAutomator class available ✓")
                self.test_results['landmark_extraction'] = {'status': 'partial', 'message': 'No model provided'}
                return True
            
            # Initialize LandmarkAutomator
            landmark_automator = LandmarkAutomator(
                model_path=self.model_path,
                device='cpu'  # Use CPU for testing
            )
            print("  ✓ LandmarkAutomator initialized successfully")
            
            # Test with prepared data
            output_landmarks = os.path.join(self.output_folder, "landmarks_output")
            os.makedirs(output_landmarks, exist_ok=True)
            
            # Note: Actual inference would require a valid model and properly formatted input
            print("  ✓ Ready for landmark extraction inference")
            self.test_results['landmark_extraction'] = {'status': 'success', 'message': 'Initialized successfully'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in landmark extraction: {e}")
            self.test_results['landmark_extraction'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_vessel_segmentation(self):
        """Test the VesselSegmenter class."""
        print("\n🩸 Testing Vessel Segmentation Module...")
        
        try:
            # Initialize VesselSegmenter for extracranial vessels
            segmenter = VesselSegmenter(
                case_dir=self.test_case_dir,
                mode="extracranial_vessels",
                fast_segmentation=True,  # Use fast mode for testing
                cta_nifti_path=os.path.join(self.test_case_dir, "cta.nii.gz")
            )
            print("  ✓ VesselSegmenter (extracranial) initialized successfully")
            
            # Initialize VesselSegmenter for intracranial vessels
            segmenter_intra = VesselSegmenter(
                case_dir=self.test_case_dir,
                mode="intracranial_vessels",
                cta_nifti_path=os.path.join(self.test_case_dir, "cta.nii.gz")
            )
            print("  ✓ VesselSegmenter (intracranial) initialized successfully")
            
            print("  ✓ Both segmentation modes available")
            self.test_results['vessel_segmentation'] = {'status': 'success', 'message': 'Both modes initialized'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in vessel segmentation: {e}")
            self.test_results['vessel_segmentation'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_centerline_extraction(self):
        """Test the CenterlineExtractor class."""
        print("\n🧵 Testing Centerline Extraction Module...")
        
        try:
            # Initialize CenterlineExtractor for extracranial vessels
            centerline_extractor = CenterlineExtractor(
                case_dir=self.test_case_dir,
                mode="extracranial_vessels",
                fast_segmentation=True
            )
            print("  ✓ CenterlineExtractor (extracranial) initialized successfully")
            
            # Initialize CenterlineExtractor for intracranial vessels  
            centerline_extractor_intra = CenterlineExtractor(
                case_dir=self.test_case_dir,
                mode="intracranial_vessels",
                fast_segmentation=False
            )
            print("  ✓ CenterlineExtractor (intracranial) initialized successfully")
            
            print("  ✓ Both centerline extraction modes available")
            self.test_results['centerline_extraction'] = {'status': 'success', 'message': 'Both modes initialized'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in centerline extraction: {e}")
            self.test_results['centerline_extraction'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_vessel_labelling(self):
        """Test the VesselLabeller class."""
        print("\n🏷️  Testing Vessel Labelling Module...")
        
        try:
            # Initialize VesselLabeller for extracranial vessels
            vessel_labeller = VesselLabeller(
                case_dir=self.test_case_dir,
                mode="extracranial_vessels"
            )
            print("  ✓ VesselLabeller (extracranial) initialized successfully")
            
            # Initialize VesselLabeller for intracranial vessels
            vessel_labeller_intra = VesselLabeller(
                case_dir=self.test_case_dir,
                mode="intracranial_vessels"
            )
            print("  ✓ VesselLabeller (intracranial) initialized successfully")
            
            print("  ✓ Both vessel labelling modes available")
            self.test_results['vessel_labelling'] = {'status': 'success', 'message': 'Both modes initialized'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in vessel labelling: {e}")
            self.test_results['vessel_labelling'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_feature_extraction(self):
        """Test the FeatureExtractor class."""
        print("\n🔍 Testing Feature Extraction Module...")
        
        try:
            # Initialize FeatureExtractor
            feature_extractor = FeatureExtractor(
                case_dir=self.test_case_dir,
                mode="extracranial_vessels",
                sampling_distance_mm=2,
                cta_nifti_path=os.path.join(self.test_case_dir, "cta.nii.gz")
            )
            print("  ✓ FeatureExtractor (extracranial) initialized successfully")
            
            # Initialize for intracranial vessels
            feature_extractor_intra = FeatureExtractor(
                case_dir=self.test_case_dir,
                mode="intracranial_vessels",
                sampling_distance_mm=1,
                cta_nifti_path=os.path.join(self.test_case_dir, "cta.nii.gz")
            )
            print("  ✓ FeatureExtractor (intracranial) initialized successfully")
            
            print("  ✓ Feature extraction modules available")
            self.test_results['feature_extraction'] = {'status': 'success', 'message': 'Both modes initialized'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in feature extraction: {e}")
            self.test_results['feature_extraction'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_access_prediction(self):
        """Test the AccessPredictor class."""
        print("\n🎯 Testing Access Prediction Module...")
        
        try:
            # Initialize AccessPredictor
            access_predictor = AccessPredictor(
                case_dir=self.test_case_dir,
                cta_nifti_path=os.path.join(self.test_case_dir, "cta.nii.gz"),
                access=["femoral"],
                side=["left", "right"]
            )
            print("  ✓ AccessPredictor initialized successfully")
            
            # Test with different access types
            access_predictor_radial = AccessPredictor(
                case_dir=self.test_case_dir,
                cta_nifti_path=os.path.join(self.test_case_dir, "cta.nii.gz"),
                access=["radial"],
                side=["left", "right"] 
            )
            print("  ✓ AccessPredictor (radial access) initialized successfully")
            
            print("  ✓ Access prediction module available")
            self.test_results['access_prediction'] = {'status': 'success', 'message': 'Multiple access types supported'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in access prediction: {e}")
            self.test_results['access_prediction'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_arterial_processor(self):
        """Test the ArterialProcessor class (full pipeline)."""
        print("\n⚙️  Testing Arterial Processor (Full Pipeline)...")
        
        try:
            # Create a mock args object
            class MockArgs:
                def __init__(self, case_dir):
                    self.case_dir = case_dir
                    self.cta_nifti_path = os.path.join(case_dir, "cta.nii.gz")
                    self.skip_segmentation = True  # Skip time-consuming operations for testing
                    self.skip_centerline_extraction = True
                    self.skip_branching = True
                    self.skip_clipping = True
                    self.skip_vessel_labelling = True
                    self.skip_feature_extraction = True
                    self.skip_access_prediction = True
            
            mock_args = MockArgs(self.test_case_dir)
            
            # Initialize ArterialProcessor
            processor = ArterialProcessor(mock_args)
            print("  ✓ ArterialProcessor initialized successfully")
            print("  ✓ Full pipeline wrapper available")
            
            self.test_results['arterial_processor'] = {'status': 'success', 'message': 'Full pipeline initialized'}
            return True
            
        except Exception as e:
            print(f"  ✗ Error in arterial processor: {e}")
            self.test_results['arterial_processor'] = {'status': 'error', 'message': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all module tests."""
        print("\n🚀 Starting comprehensive module testing...")
        
        # Setup test data
        has_cta = self.setup_test_data()
        
        if not has_cta:
            print("⚠️  No CTA file available - tests will be limited to class initialization")
        
        # Run all tests
        tests = [
            ('Landmark Extraction', self.test_landmark_extraction),
            ('Vessel Segmentation', self.test_vessel_segmentation),
            ('Centerline Extraction', self.test_centerline_extraction),
            ('Vessel Labelling', self.test_vessel_labelling),
            ('Feature Extraction', self.test_feature_extraction),
            ('Access Prediction', self.test_access_prediction),
            ('Arterial Processor', self.test_arterial_processor)
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"  ✗ Unexpected error in {test_name}: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = {'status': 'error', 'message': str(e)}
        
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        success_count = 0
        total_count = len(self.test_results)
        
        for module, result in self.test_results.items():
            status = result['status']
            message = result['message']
            
            if status == 'success':
                icon = "✅"
                success_count += 1
            elif status == 'partial':
                icon = "⚠️ "
            else:
                icon = "❌"
            
            print(f"{icon} {module.replace('_', ' ').title()}: {status.upper()} - {message}")
        
        print("-" * 80)
        print(f"Overall: {success_count}/{total_count} modules passed successfully")
        
        # Save results to file
        results_file = os.path.join(self.output_folder, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': f"{success_count}/{total_count} passed",
                'results': self.test_results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
        print("=" * 80)


def main():
    """Main function to run the comprehensive test script."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test script for all arterial modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_arterial_modules.py --input_folder /data/test_case --output_folder /results/tests
  python test_arterial_modules.py --input_folder /data/test_case --output_folder /results/tests --model_path /models/landmarks.pth
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to input folder (should contain cta.nii.gz if available)'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Path to output folder for test results'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to landmark extraction model (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run tester
    tester = ArterialModuleTester(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_path=args.model_path
    )
    
    tester.run_all_tests()


if __name__ == "__main__":
    main()