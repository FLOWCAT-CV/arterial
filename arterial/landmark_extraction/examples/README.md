# Arterial Package Examples

This directory contains example scripts demonstrating how to use the arterial package for medical image analysis.

## Scripts Overview

### 1. CTA to Landmarks Pipeline (`cta_to_landmarks_pipeline.py`)

A minimal routine that processes CTA images to extract anatomical landmarks.

**Purpose**: Demonstrates the complete workflow from CTA input to landmark extraction.

**Required inputs**:
- Input folder containing `cta.nii.gz` file
- Trained landmark extraction model (`.pth` file)
- Output folder for results

**Usage**:
```bash
python cta_to_landmarks_pipeline.py --input_folder /path/to/cta/folder --output_folder /path/to/output --model_path /path/to/model.pth
```

**Optional parameters**:
- `--device`: Specify device (`cpu`, `cuda`, or `auto`)

**Outputs**:
- `pred_mask.nii.gz`: Segmentation mask with landmark locations
- `F_o.json`: Landmark coordinates in JSON format

### 2. Arterial Modules Test Script (`test_arterial_modules.py`)

A comprehensive test script that initializes all main classes from every module in the arterial package.

**Purpose**: 
- Test and validate all arterial modules
- Demonstrate proper initialization of each class
- Verify module dependencies and requirements

**Modules tested**:
1. **Landmark Extraction** (`LandmarkAutomator`)
2. **Vessel Segmentation** (`VesselSegmenter`)
3. **Centerline Extraction** (`CenterlineExtractor`) 
4. **Vessel Labelling** (`VesselLabeller`)
5. **Feature Extraction** (`FeatureExtractor`)
6. **Access Prediction** (`AccessPredictor`)
7. **Full Pipeline** (`ArterialProcessor`)

**Usage**:
```bash
python test_arterial_modules.py --input_folder /path/to/input --output_folder /path/to/output
```

**With landmark model**:
```bash
python test_arterial_modules.py --input_folder /path/to/input --output_folder /path/to/output --model_path /path/to/model.pth
```

**Outputs**:
- Console output showing test results for each module
- `test_results.json`: Detailed test results in JSON format
- Test case directory with copied/created input files

## Input Requirements

### For CTA to Landmarks Pipeline:
- `cta.nii.gz`: CTA image in NIfTI format
- `F.json`: Template file (auto-created if missing)
- `affine_before_origin_change.txt`: Affine transformation (auto-created if missing)

### For Module Testing:
- `cta.nii.gz`: CTA image (optional but recommended)
- Other required files are auto-generated for testing

## Module Descriptions

### Landmark Extraction
Extracts anatomical landmarks from CTA images using deep learning models.
- **Main class**: `LandmarkAutomator`
- **Key method**: `infer_folder()`

### Vessel Segmentation  
Segments blood vessels from CTA images using nnU-Net architecture.
- **Main class**: `VesselSegmenter`
- **Modes**: `extracranial_vessels`, `intracranial_vessels`

### Centerline Extraction
Extracts vessel centerlines from segmented vessels.
- **Main class**: `CenterlineExtractor`
- **Output**: Centerline models and segment arrays

### Vessel Labelling
Labels vessel segments with anatomical names.
- **Main class**: `VesselLabeller`  
- **Output**: Labeled vessel graphs

### Feature Extraction
Extracts features at multiple scales from vessel centerlines.
- **Main class**: `FeatureExtractor`
- **Features**: Local, segment, and global features

### Access Prediction
Predicts optimal vascular access routes for endovascular procedures.
- **Main class**: `AccessPredictor`
- **Access types**: Femoral, radial

### Full Pipeline
Orchestrates the complete analysis pipeline.
- **Main class**: `ArterialProcessor`
- **Workflow**: Segmentation → Centerlines → Labelling → Features → Access

## Example Workflows

### Basic Landmark Extraction:
```bash
# 1. Prepare input folder with cta.nii.gz
mkdir /data/case01
cp your_cta.nii.gz /data/case01/cta.nii.gz

# 2. Run landmark extraction
python cta_to_landmarks_pipeline.py \
    --input_folder /data/case01 \
    --output_folder /results/landmarks \
    --model_path /models/landmark_model.pth

# 3. Check results
ls /results/landmarks/case01/
# Should contain: pred_mask.nii.gz, F_o.json
```

### Module Testing:
```bash
# 1. Test all modules
python test_arterial_modules.py \
    --input_folder /data/test_data \
    --output_folder /results/module_tests

# 2. Check test results
cat /results/module_tests/test_results.json
```

## Notes

- **GPU Support**: Both scripts automatically detect CUDA availability
- **Error Handling**: Comprehensive error checking and user feedback
- **Flexible Input**: Scripts handle missing auxiliary files by auto-generation
- **Modular Design**: Each module can be tested independently
- **Comprehensive Logging**: Detailed console output and result files

## Dependencies

Ensure you have the arterial package and its dependencies installed:
- torch
- nibabel
- numpy
- torchio
- cv2 (opencv-python)
- cc3d
- monai

## Troubleshooting

**Common Issues**:
1. **Missing CTA file**: Ensure `cta.nii.gz` exists in input folder
2. **Model loading errors**: Verify model path and compatibility
3. **Import errors**: Check that arterial package is in Python path
4. **Memory issues**: Use CPU device for large images or limited RAM

**Debug Mode**: Add `--device cpu` to reduce memory usage during testing.