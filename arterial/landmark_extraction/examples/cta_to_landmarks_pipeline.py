import os
import sys
import argparse
import shutil
import numpy as np
import torch
import nibabel as nib
import torchio as tio   

from arterial.landmark_extraction.landmark_extractor import LandmarkAutomator

def validate_inputs(input_folder, model_path):
    """
    Validate that all required inputs exist.
    
    Parameters:
    -----------
    input_folder : str
        Path to the folder containing the input NIfTI file (cta.nii.gz)
    model_path : str
        Path to the trained model file
        
    Returns:
    --------
    bool
        True if all inputs are valid, False otherwise
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return False
    
    # Check if cta.nii.gz exists in the folder
    input_file = os.path.join(input_folder, "cta.nii.gz")
    if not os.path.exists(input_file):
        print(f"Error: Input file cta.nii.gz not found in folder {input_folder}")
        return False
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    return True


def run_cta_to_landmarks_pipeline(input_folder, output_folder, model_path, device='auto'):
    """
    Run the complete CTA to landmarks pipeline.
    
    Parameters:
    -----------
    input_folder : str
        Path to the folder containing the input NIfTI file (cta.nii.gz)
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
    if not validate_inputs(input_folder, model_path):
        return False
    
    # Get the input file path
    input_file = os.path.join(input_folder, "cta.nii.gz")
    
    # Create temporary folder for processing
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    
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
            folder_path=output_folder,
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
        print("Cleaning up temporary folder")


def main():
    """Main function to run the CTA to landmarks pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Minimal CTA to Landmarks Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cta_to_landmarks_pipeline.py --input_folder /data --output_folder /results --model_path /models/landmark_model.pth
  python cta_to_landmarks_pipeline.py --input_folder /data --output_folder /results --model_path /models/landmark_model.pth --device cuda
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to the folder containing the input NIfTI file (cta.nii.gz)'
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
        input_folder=args.input_folder,
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
