#!/usr/bin/env python3
"""
Debug version to test the actual preprocessing pipeline
"""

import os
import sys
import numpy as np
import torch
import nibabel as nib
import traceback

from arterial.landmark_extraction.model import load_trained_model_seg
from arterial.landmark_extraction.landmark_extractor import SingleCTADataset

def debug_with_actual_preprocessing():
    """Test with the actual preprocessing pipeline"""
    print("=== TESTING WITH ACTUAL PREPROCESSING ===")
    
    input_folder = "/media/Disk_B/arterial_models/tets_landmark/input"
    model_path = "/media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print("1. Loading model...")
        model = load_trained_model_seg(model_path, device)
        model.eval()
        print("✓ Model loaded successfully")
        
        print("\n2. Creating dataset...")
        dataset = SingleCTADataset(input_folder)
        sample = dataset[0]
        print("✓ Dataset created successfully")
        
        print(f"\n3. Checking preprocessed volume...")
        volume = sample["volume"]
        print(f"   - Volume shape: {volume.shape}")
        print(f"   - Volume dtype: {volume.dtype}")
        print(f"   - Volume min/max: {volume.min():.3f}/{volume.max():.3f}")
        
        # Add batch dimension and move to device
        volume_batch = volume.unsqueeze(0).to(device)
        print(f"   - Batch shape: {volume_batch.shape}")
        
        print("\n4. Running model inference...")
        with torch.no_grad():
            preds = model(volume_batch)
        
        print(f"✓ Model inference successful!")
        print(f"   - Predictions shape: {preds.shape}")
        print(f"   - Predictions dtype: {preds.dtype}")
        
        print("\n=== ALL TESTS PASSED ===")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False

def main():
    success = debug_with_actual_preprocessing()
    if success:
        print("\nPreprocessing is working correctly!")
        print("You can now run the full test script.")
    else:
        print("\nStill have issues to resolve...")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Debug version to identify the exact source of the numpy.ndarray error
"""

import os
import sys
import numpy as np
import torch
import nibabel as nib
import traceback

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_trained_model_seg

def debug_model_loading():
    """Test just the model loading"""
    print("Testing model loading...")
    model_path = "/media/Disk_B/arterial_models/landmark_extraction/six_landmarks_11_7.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = load_trained_model_seg(model_path, device)
        model.eval()
        print("✓ Model loaded successfully")
        return model, device
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return None, device

def debug_data_loading():
    """Test the data loading and preprocessing"""
    print("\nTesting data loading...")
    input_folder = "/media/Disk_B/arterial_models/tets_landmark/input"
    cta_path = os.path.join(input_folder, "cta.nii.gz")
    
    try:
        print(f"Loading: {cta_path}")
        img = nib.load(cta_path)
        print(f"✓ NIfTI loaded, shape: {img.shape}")
        
        volume = img.get_fdata().astype(np.float32)
        print(f"✓ Data extracted, shape: {volume.shape}, dtype: {volume.dtype}")
        
        volume = np.clip(volume, 0, 700) / 700
        print(f"✓ Normalized, min: {volume.min():.3f}, max: {volume.max():.3f}")
        
        volume = np.transpose(volume, (2, 0, 1))
        print(f"✓ Transposed, shape: {volume.shape}")
        
        volume = np.expand_dims(volume, axis=0)
        print(f"✓ Channel dimension added, shape: {volume.shape}")
        
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        print(f"✓ Converted to tensor, shape: {volume_tensor.shape}, dtype: {volume_tensor.dtype}")
        
        # Add batch dimension
        volume_batch = volume_tensor.unsqueeze(0)
        print(f"✓ Batch dimension added, shape: {volume_batch.shape}")
        
        return volume_batch, img.affine
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        traceback.print_exc()
        return None, None

def debug_model_inference(model, volume_batch, device):
    """Test model inference"""
    print("\nTesting model inference...")
    
    try:
        volume_batch = volume_batch.to(device)
        print(f"✓ Data moved to device: {device}")
        
        with torch.no_grad():
            preds = model(volume_batch)
            
        print(f"✓ Model inference completed")
        print(f"  - Predictions shape: {preds.shape}")
        print(f"  - Predictions dtype: {preds.dtype}")
        print(f"  - Predictions device: {preds.device}")
        print(f"  - Predictions type: {type(preds)}")
        
        # Test conversion to numpy
        preds_cpu = preds.cpu()
        print(f"✓ Moved to CPU: {type(preds_cpu)}")
        
        preds_numpy = preds_cpu.numpy()
        print(f"✓ Converted to numpy: {type(preds_numpy)}, shape: {preds_numpy.shape}")
        
        preds_batch_removed = preds_numpy[0]
        print(f"✓ Batch dimension removed: {type(preds_batch_removed)}, shape: {preds_batch_removed.shape}")
        
        return preds_batch_removed
        
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        traceback.print_exc()
        return None

def main():
    print("=== DEBUG MODE ===")
    
    # Test each component separately
    model, device = debug_model_loading()
    if model is None:
        return
    
    volume_batch, affine = debug_data_loading()
    if volume_batch is None:
        return
    
    preds = debug_model_inference(model, volume_batch, device)
    if preds is None:
        return
    
    print("\n✓ All components working correctly!")
    print("The error must be in the postprocessing code...")
    
    # Now test the basic postprocessing to find where it fails
    print("\nTesting postprocessing...")
    try:
        # Create combined mask
        combined = np.zeros(preds.shape[1:], dtype=np.uint8)
        confidence_map = np.zeros(preds.shape[1:], dtype=np.float32)
        print("✓ Created masks")
        
        for c in range(1, preds.shape[0]):
            mask = (preds[c] > -1) & ((preds[c] > confidence_map) | (combined == 0))
            confidence_map[mask] = preds[c][mask]
            combined[mask] = c
        print("✓ Confidence mapping completed")
        
        print("=== ALL TESTS PASSED ===")
        
    except Exception as e:
        print(f"✗ Postprocessing failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()