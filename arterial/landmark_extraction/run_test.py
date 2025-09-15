#!/usr/bin/env python3
"""
Simple wrapper for landmark extraction testing.
Provides an easy interface to test landmark extraction with custom paths.
"""

import os
import sys
from pathlib import Path

# Add the landmark_extraction directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from test_landmark_extractor_flexible import test_landmark_extraction


def run_landmark_test(cta_folder: str, model_path: str, output_folder: str):
    """
    Simple wrapper to run landmark extraction test.
    
    Args:
        cta_folder (str): Path to folder containing cta.nii.gz
        model_path (str): Path to model .pth file  
        output_folder (str): Path to save results
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("🚀 Starting landmark extraction test...")
    print(f"   CTA folder: {cta_folder}")
    print(f"   Model path: {model_path}")
    print(f"   Output folder: {output_folder}")
    print()
    
    success = test_landmark_extraction(cta_folder, model_path, output_folder)
    
    if success:
        print("\n✅ Test completed successfully!")
        print(f"📁 Results saved to: {output_folder}")
        print("📄 Files created:")
        print("   - landmarks.txt (landmark coordinates)")
        print("   - heatmaps.pt (model output heatmaps)")
        print("   - template.json (3D Slicer template)")
    else:
        print("\n❌ Test failed! Check the logs above for details.")
    
    return success


def main():
    """Command line interface for easy testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run landmark extraction test with custom paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test.py /path/to/cta/folder /path/to/model.pth /path/to/output
  
  # Using arterial_models structure:
  python run_test.py \\
    /media/Disk_B/arterial_models/test_data/input_test_data \\
    /media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth \\
    /media/Disk_B/arterial_models/test_data/output
        """
    )
    
    parser.add_argument('cta_folder', help='Path to folder containing cta.nii.gz')
    parser.add_argument('model_path', help='Path to model .pth file')
    parser.add_argument('output_folder', help='Path to save output results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.cta_folder):
        print(f"❌ Error: CTA folder does not exist: {args.cta_folder}")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file does not exist: {args.model_path}")
        return 1
    
    cta_file = os.path.join(args.cta_folder, "cta.nii.gz")
    if not os.path.exists(cta_file):
        print(f"❌ Error: cta.nii.gz not found in folder: {args.cta_folder}")
        return 1
    
    # Run the test
    success = run_landmark_test(args.cta_folder, args.model_path, args.output_folder)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())