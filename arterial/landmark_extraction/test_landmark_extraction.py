#!/usr/bin/env python3
"""
Test script for CarotiCAT landmark extraction.

This script demonstrates how to use the LandmarkAutomator class for both
folder-based and array-based inference.

Usage:
    python test_landmark_extraction.py --model_path /path/to/model.pth --input_folder /path/to/input --output_folder /path/to/output
    python test_landmark_extraction.py --model_path /path/to/model.pth --input_nifti /path/to/image.nii.gz --output_folder /path/to/output
"""

import os
import argparse
import numpy as np
import nibabel as nib
import sys

from landmark_extractor import LandmarkAutomator


def test_folder_inference(model_path, input_folder, output_folder, device=None):
    """
    Test inference from a folder containing cta.nii.gz
    
    Parameters
    ----------
        model_path (str): Path to the trained model (.pth file)
        input_folder (str): Path to folder containing 'cta.nii.gz'
        output_folder (str): Path to save results
        device (str, optional): Device to use ('cpu' or 'cuda')
    """
    print("=" * 50)
    print("Testing folder-based inference...")
    print(f"Model: {model_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 50)
    
    # Check if input folder and file exist
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    cta_path = os.path.join(input_folder, "cta.nii.gz")
    if not os.path.exists(cta_path):
        raise ValueError(f"cta.nii.gz not found in: {input_folder}")
    
    # Initialize the landmark automator
    automator = LandmarkAutomator(model_path, device=device)
    
    # Perform inference
    try:
        landmarks_mm = automator.infer_folder(
            folder_path=input_folder,
            output_folder=output_folder,
            save_mask=True,
            save_json=True
        )
        
        print("\nInference completed successfully!")
        print(f"Landmarks (mm coordinates):")
        landmark_names = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
        for i, (name, coords) in enumerate(zip(landmark_names, landmarks_mm)):
            print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
            
        print(f"\nResults saved to: {output_folder}")
        return landmarks_mm
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


def test_array_inference(model_path, input_nifti, output_folder=None, device=None):
    """
    Test inference from a numpy array (loaded from NIfTI)
    
    Parameters
    ----------
        model_path (str): Path to the trained model (.pth file)
        input_nifti (str): Path to NIfTI file
        output_folder (str, optional): Path to save results (if None, no files saved)
        device (str, optional): Device to use ('cpu' or 'cuda')
    """
    print("=" * 50)
    print("Testing array-based inference...")
    print(f"Model: {model_path}")
    print(f"Input NIfTI: {input_nifti}")
    print(f"Output folder: {output_folder}")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_nifti):
        raise ValueError(f"Input NIfTI file does not exist: {input_nifti}")
    
    # Load NIfTI file
    print("Loading NIfTI file...")
    nib_img = nib.load(input_nifti)
    data = nib_img.get_fdata()
    affine = nib_img.affine
    
    print(f"Image shape: {data.shape}")
    print(f"Affine shape: {affine.shape}")
    
    # Initialize the landmark automator
    automator = LandmarkAutomator(model_path, device=device)
    
    # Perform inference
    try:
        # Test with numpy array
        print("\nTesting with numpy array...")
        results_array = automator.infer_from_array(
            data=data,
            affine=affine,
            output_folder=output_folder,
            save_mask=(output_folder is not None),
            save_json=(output_folder is not None)
        )
        
        # Test with nibabel image
        print("Testing with nibabel image...")
        results_nib = automator.infer_from_array(
            data=nib_img,
            output_folder=None,  # Don't save twice
            save_mask=False,
            save_json=False
        )
        
        print("\nInference completed successfully!")
        
        # Display results from numpy array version
        landmarks_mm = results_array['landmarks_mm']
        landmarks_voxel = results_array['landmarks_voxel']
        heatmaps = results_array['heatmaps']
        
        print(f"\nLandmarks (mm coordinates):")
        landmark_names = ["l-tica", "r-tica", "l-eica", "r-eica", "r-mca", "l-mca"]
        for i, (name, coords) in enumerate(zip(landmark_names, landmarks_mm)):
            print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
            
        print(f"\nLandmarks (voxel coordinates):")
        for i, (name, coords) in enumerate(zip(landmark_names, landmarks_voxel)):
            print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
            
        print(f"\nHeatmaps shape: {heatmaps.shape}")
        
        # Compare results (should be identical or very close)
        diff_mm = np.abs(results_array['landmarks_mm'] - results_nib['landmarks_mm'])
        max_diff = np.max(diff_mm)
        print(f"Max difference between numpy and nibabel results: {max_diff:.6f} mm")
        
        if output_folder:
            print(f"\nResults saved to: {output_folder}")
            
        return results_array
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Test CarotiCAT landmark extraction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--input_folder', type=str,
                        help='Path to folder containing cta.nii.gz')
    parser.add_argument('--input_nifti', type=str,
                        help='Path to NIfTI file for array-based testing')
    parser.add_argument('--output_folder', type=str,
                        help='Path to save results')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='Device to use (auto-detected if not specified)')
    parser.add_argument('--test_mode', type=str, choices=['folder', 'array', 'both'], 
                        default='both',
                        help='Which test mode to run')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        sys.exit(1)
    
    if args.test_mode in ['folder', 'both'] and not args.input_folder:
        print("Error: --input_folder is required for folder-based testing")
        sys.exit(1)
        
    if args.test_mode in ['array', 'both'] and not args.input_nifti:
        print("Error: --input_nifti is required for array-based testing")
        sys.exit(1)
    
    # Create output folder if specified
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
    
    # Run tests
    try:
        if args.test_mode in ['folder', 'both']:
            landmarks_folder = test_folder_inference(
                model_path=args.model_path,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                device=args.device
            )
        
        if args.test_mode in ['array', 'both']:
            results_array = test_array_inference(
                model_path=args.model_path,
                input_nifti=args.input_nifti,
                output_folder=args.output_folder,
                device=args.device
            )
        
        # Compare results if both tests were run
        if args.test_mode == 'both':
            print("\n" + "=" * 50)
            print("Comparing folder vs array results...")
            diff = np.abs(landmarks_folder - results_array['landmarks_mm'])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"Max difference: {max_diff:.6f} mm")
            print(f"Mean difference: {mean_diff:.6f} mm")
            
            if max_diff < 1e-3:
                print("✓ Results are consistent between methods!")
            else:
                print("⚠ Results differ between methods - check implementation")
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()