#!/usr/bin/env python3
"""
Minimal CTA to Landmarks Pipeline

This script demonstrates how to extract landmarks from a CTA image using the arterial package.
The pipeline requires:
1. A 'cta.nii.gz' file as input
2. A trained landmark extraction model
3. The output will be saved in the specified folder

Example usage:
    python cta_to_landmarks_pipeline.py --input_file /path/to/cta.nii.gz --output_folder /path/to/output --model_path /path/to/model.pth
"""

import os
import sys
import argparse
import shutil
import numpy as np

# Add the parent directory to the path to import arterial modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arterial.landmark_extraction.landmark_extractor import LandmarkAutomator


def create_required_files(input_file, temp_folder):
    """
    Create required files in a temporary folder for processing.
    
    Parameters:
    -----------
    input_file : str
        Path to the input NIfTI file
    temp_folder : str
        Path to the temporary folder for intermediate files
    """
    # Copy the input file to the temporary folder
    cta_path = os.path.join(temp_folder, "cta.nii.gz")
    shutil.copy(input_file, cta_path)
    
    # Check if F.json exists, if not create from template
    f_json_path = os.path.join(temp_folder, "F.json")
    template_path = '/media/Disk_B/databases/david_projects/VHIR/landMARKS/arterial_update/arterial/landmark_extraction/template.json'
    if os.path.exists(template_path):
        shutil.copy(template_path, f_json_path)
        print(f"Created F.json from template at {f_json_path}")
    else:
        print(f"Warning: Template file not found at {template_path}")
    
    # Create a dummy affine file if it doesn't exist
    affine_path = os.path.join(temp_folder, "affine_before_origin_change.txt")
    identity_affine = np.eye(4)
    np.savetxt(affine_path, identity_affine)
    print(f"Created dummy affine file at {affine_path}")


def validate_inputs(input_file, model_path):
    """
    Validate that all required inputs exist.
    
    Parameters:
    -----------
    input_file : str
        Path to the input NIfTI file
    model_path : str
        Path to the trained model file
        
    Returns:
    --------
    bool
        True if all inputs are valid, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return False
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    return True


def run_cta_to_landmarks_pipeline(input_file, output_folder, model_path, device='auto'):
    """
    Run the complete CTA to landmarks pipeline.
    
    Parameters:
    -----------
    input_file : str
        Path to the input NIfTI file
    output_folder : str
        Path to save the output landmarks and masks
    model_path : str
        Path to the trained landmark extraction model
    device : str
        Device to run inference on ('cpu', 'cuda', or 'auto')
    """
    
    print("=" * 60)
    print("CTA to Landmarks Pipeline")
    print("=" * 60)
    
    # Validate inputs
    if not validate_inputs(input_file, model_path):
        return False
    
    # Create temporary folder for processing
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Create required files in the temporary folder
    create_required_files(input_file, temp_folder)
    
    # Initialize the landmark automator
    print(f"Initializing LandmarkAutomator with model: {model_path}")
    print(f"Using device: {device}")
    
    try:
        landmark_automator = LandmarkAutomator(
            model_path=model_path,
            device=device if device != 'auto' else None
        )
        print("✓ LandmarkAutomator initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing LandmarkAutomator: {e}")
        return False
    
    # Run inference
    print(f"Processing CTA from: {input_file}")
    print(f"Saving results to: {output_folder}")
    
    try:
        landmark_automator.infer_folder(
            folder_path=temp_folder,
            output_folder=output_folder,
            save_mask=True,
        )
        print("✓ Landmark extraction completed successfully")
        
        # Print results summary
        pred_mask_path = os.path.join(output_folder, "pred_mask.nii.gz")
        pred_json_path = os.path.join(output_folder, "F_o.json")
        
        print("\nResults saved:")
        if os.path.exists(pred_mask_path):
            print(f"  - Landmark mask: {pred_mask_path}")
        if os.path.exists(pred_json_path):
            print(f"  - Landmark coordinates: {pred_json_path}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error during landmark extraction: {e}")
        return False
    finally:
        # Clean up temporary folder
        shutil.rmtree(temp_folder, ignore_errors=True)


def main():
    """Main function to run the CTA to landmarks pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Minimal CTA to Landmarks Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cta_to_landmarks_pipeline.py --input_file /data/cta.nii.gz --output_folder /results --model_path /models/landmark_model.pth
  python cta_to_landmarks_pipeline.py --input_file /data/cta.nii.gz --output_folder /results --model_path /models/landmark_model.pth --device cuda
        """
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the input NIfTI file (cta.nii.gz)'
    )
    
    parser.add_argument(
        '--output_folder', 
        type=str,
        required=True,
        help='Path to save output landmarks and masks'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained landmark extraction model (.pth file)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run inference on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    success = run_cta_to_landmarks_pipeline(
        input_file=args.input_file,
        output_folder=args.output_folder,
        model_path=args.model_path,
        device=args.device
    )
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
