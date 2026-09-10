# Landmark Detection Module

The landmark detection module performs automatic anatomical landmark detection on CT Angiography (CTA) images using deep learning. It identifies key vascular bifurcation points critical for targeted centerline extraction and vascular analysis.

---

## Table of Contents

- [Overview](#overview)
- [Detected Landmarks](#detected-landmarks)
- [Detection Methods](#detection-methods)
  - [CTA-Only Detection](#cta-only-detection)
  - [CTA + Segmentation Detection](#cta--segmentation-detection)
- [Usage](#usage)
  - [Minimal Example](#minimal-example)
  - [With Segmentation Enhancement](#with-segmentation-enhancement)
  - [Integration with Centerline Extraction](#integration-with-centerline-extraction)
- [Landmark Refinement](#landmark-refinement)
- [Expected Outputs](#expected-outputs)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)
- [References](#references)

---

## Overview

The landmark detection module uses a **3D U-Net** architecture (MONAI) to segment anatomical landmark regions, then extracts centroid coordinates for each landmark. It supports two operational modes:

| Mode | Landmarks Detected | Use Case |
|------|-------------------|----------|
| `extracranial_vessels` | 6 landmarks | Full carotid/cerebral analysis |
| `intracranial_vessels` | 4 landmarks | Cerebral vessel analysis only |

The module is designed to provide anatomically meaningful endpoints for the centerline extraction module's endpoint-to-endpoint functionality.

---

## Detected Landmarks

The model detects the following anatomical landmarks:

### Extracranial Mode (6 landmarks)

| Label | Anatomical Structure | Description |
|-------|---------------------|-------------|
| `l-tica` | Left Terminal ICA | Left internal carotid artery bifurcation (into MCA/ACA) |
| `r-tica` | Right Terminal ICA | Right internal carotid artery bifurcation (into MCA/ACA) |
| `l-eica` | Left External ICA | Left carotid bifurcation (CCA → ICA/ECA) |
| `r-eica` | Right External ICA | Right carotid bifurcation (CCA → ICA/ECA) |
| `l-mca` | Left MCA | Left middle cerebral artery M1 bifurcation |
| `r-mca` | Right MCA | Right middle cerebral artery M1 bifurcation |

### Intracranial Mode (4 landmarks)

| Label | Anatomical Structure |
|-------|---------------------|
| `l-tica` | Left Terminal ICA |
| `r-tica` | Right Terminal ICA |
| `l-mca` | Left MCA M1 bifurcation |
| `r-mca` | Right MCA M1 bifurcation |

---

## Detection Methods

### CTA-Only Detection

Uses a single-channel 3D U-Net trained on CTA images alone:

```
Input: CTA volume (1 channel)
  ↓
3D U-Net (channels: 16→32→64→128→256)
  ↓
7-class segmentation (background + 6 landmarks)
  ↓
Centroid extraction per class
  ↓
Output: 6 landmark coordinates (RAS mm)
```

**Model file:** `six_landmarks_11_7.pth`

### CTA + Segmentation Detection

Uses a two-channel model that leverages vessel segmentation for improved accuracy:

```
Input: CTA volume + Vessel segmentation (2 channels)
  ↓
3D U-Net (channels: 16→32→64→128→256)
  ↓
7-class segmentation (background + 6 landmarks)
  ↓
Centroid extraction per class
  ↓
Output: 6 landmark coordinates (RAS mm)
```

**Model file:** `six_landmarks_2ch.pth`

The two-channel model typically provides more accurate landmark localization by incorporating explicit vessel structure information.

---

## Usage

### Minimal Example

```python
from arterial.landmark_detection.landmark_detector import LandmarkDetector

# Initialize the detector
detector = LandmarkDetector(
    case_dir="/path/to/case",
    mode="extracranial_vessels",
    cta_nifti_path="/path/to/cta.nii.gz"
)

# Detect landmarks
landmarks = detector.detect_landmarks_on_cta(
    save=True,
    use_segmentation_model=False  # CTA-only model
)

# Access results
print(landmarks)
# {'l-tica': (-25.3, -12.5, 45.2), 'r-tica': (28.1, -11.8, 44.9), ...}
```

### With Segmentation Enhancement

For improved accuracy using vessel segmentation:

```python
from arterial.landmark_detection.landmark_detector import LandmarkDetector

detector = LandmarkDetector(
    case_dir="/path/to/case",
    mode="extracranial_vessels",
    cta_nifti_path="/path/to/cta.nii.gz",
    segmentation_nifti_path="/path/to/segmentation.nii.gz"
)

# Detect with 2-channel model and refinement
landmarks = detector.detect_landmarks_on_cta(
    save=True,
    return_mask=True,                    # Save predicted mask for visualization
    use_segmentation_model=True,         # Use 2-channel model
    refine_with_segmentation=True,       # Snap to vessel structures
    refinement_method='adaptive',        # 'adaptive', 'centerline', or 'bifurcation'
    refinement_radius_mm=5.0             # Maximum snap distance
)
```

### Integration with Centerline Extraction

Using detected landmarks for targeted centerline extraction:

```python
from arterial.landmark_detection.landmark_detector import LandmarkDetector
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

# Step 1: Detect landmarks
detector = LandmarkDetector(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)
landmarks = detector.detect_landmarks_on_cta(save=True)

# Step 2: Extract centerline between landmarks
extractor = CenterlineExtractor(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)

# Extract centerline from right carotid bifurcation to right MCA
extractor.extract_centerline_between_endpoints(
    startpoint=landmarks['r-eica'],
    endpoint=landmarks['r-mca'],
    centerline_id='r-ica-to-mca',
    save=True
)
```

---

## Landmark Refinement

The module includes post-processing refinement that snaps detected landmarks to vessel structures using the segmentation mask.

### Refinement Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `centerline` | Snaps to nearest vessel centerline point | General landmarks |
| `bifurcation` | Snaps to nearest bifurcation point | Bifurcation landmarks |
| `adaptive` | Uses `bifurcation` for eICA, `centerline` for others | Default recommendation |

### Refinement Pipeline

```
┌────────────────────┐
│  Raw Predictions   │
│  (from U-Net)      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Skeletonization   │
│  (vessel skeleton) │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Bifurcation       │
│  Detection         │
│  (neighbor count)  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐     ┌─────────────────────┐
│  Snap to Nearest   │────▶│  Refined Landmarks  │
│  Structure         │     │  (within radius)    │
└────────────────────┘     └─────────────────────┘
```

The refinement reports displacement statistics showing how far each landmark moved.

---

## Expected Outputs

### Directory Structure

```
{case_dir}/{mode}/
├── landmarks.json              # Landmark coordinates as dictionary
├── landmarks_slicer.json       # 3D Slicer-compatible markup JSON
└── landmarks_mask.nii.gz       # Predicted mask (if return_mask=True)
```

### landmarks.json

Simple dictionary mapping landmark labels to RAS coordinates:

```json
{
    "l-tica": [-25.3, -12.5, 45.2],
    "r-tica": [28.1, -11.8, 44.9],
    "l-eica": [-32.1, -85.2, -12.3],
    "r-eica": [35.8, -84.1, -11.9],
    "l-mca": [-42.5, -5.3, 52.1],
    "r-mca": [45.2, -4.8, 51.8]
}
```

### landmarks_slicer.json

Markup JSON compatible with 3D Slicer's Markups module for visualization:

```json
{
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/...",
    "markups": [{
        "type": "Fiducial",
        "coordinateSystem": "LPS",
        "controlPoints": [
            {"id": "1", "label": "l-tica", "position": [-25.3, -12.5, 45.2]},
            ...
        ]
    }]
}
```

### landmarks_mask.nii.gz

Multi-label NIfTI mask with the following encoding:

| Value | Structure |
|-------|-----------|
| 0 | Background |
| 1 | Left Terminal ICA |
| 2 | Right Terminal ICA |
| 3 | Left External ICA |
| 4 | Right External ICA |
| 5 | Right MCA |
| 6 | Left MCA |

---

## Configuration Options

### LandmarkDetector Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **Required** | Output directory for results |
| `mode` | str | `"extracranial_vessels"` | Detection mode: `"extracranial_vessels"` or `"intracranial_vessels"` |
| `cta_nifti_path` | str | `None` | Path to CTA. If `None`, expects `{case_dir}/cta.nii.gz` |
| `segmentation_nifti_path` | str | `None` | Path to segmentation. If `None`, looks in `{case_dir}/{mode}/segmentation.nii.gz` |

### detect_landmarks_on_cta() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_mask` | bool | `False` | Whether to save the predicted landmark mask |
| `save` | bool | `True` | Whether to save results to disk |
| `use_segmentation_model` | bool | `True` | Use 2-channel model (CTA + segmentation) |
| `refine_with_segmentation` | bool | `True` | Refine landmarks by snapping to vessels |
| `refinement_method` | str | `'adaptive'` | Refinement method: `'centerline'`, `'bifurcation'`, or `'adaptive'` |
| `refinement_radius_mm` | float | `5.0` | Maximum search radius for refinement (mm) |

---

## Module Components

### `landmark_detector.py`

Main orchestrator class `LandmarkDetector` for landmark detection.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `detect_landmarks_on_cta()` | Main entry point for landmark detection |
| `_load_cta_nifti_from_file()` | Load CTA from disk |
| `_load_segmentation_if_available()` | Load segmentation if present |
| `_load_cta_nifti_from_nib()` | Load CTA from nibabel object (for pipeline integration) |

### `landmark_detection.py`

Core detection function:

| Function | Description |
|----------|-------------|
| `infer_landmarks_from_array()` | Performs inference and postprocessing |

### `model.py`

Model architecture and loading:

| Class/Function | Description |
|----------------|-------------|
| `MonaiUNet3DSeg` | 3D U-Net model class (MONAI-based) |
| `load_trained_model_seg()` | Load trained model weights |
| `combined_loss()` | Training loss function (Dice + CE) |

**Model Architecture:**
- Spatial dimensions: 3D
- Encoder channels: (16, 32, 64, 128, 256)
- Strides: (2, 2, 2, 2)
- Residual units: 2 per level
- Normalization: Batch Normalization

### `utils.py`

Preprocessing, postprocessing, and refinement utilities:

| Class/Function | Description |
|----------------|-------------|
| `SingleCTADataset` | Dataset class for preprocessing |
| `preprocess_for_landmark_detection()` | Prepare input for model |
| `postprocess_preds()` | Extract centroids from predictions |
| `resample_mask_to_original_cta()` | Resample mask to native resolution |
| `refine_landmarks_with_segmentation()` | Snap landmarks to vessel structures |
| `_LandmarkRefiner` | Internal refinement logic class |

**Preprocessing Pipeline:**

| Step | Operation |
|------|-----------|
| 1 | Resample to 0.8mm isotropic |
| 2 | Crop/Pad to 320×320×480 |
| 3 | Clip intensities [0, 700] HU |
| 4 | Normalize to [0, 1] |
| 5 | Transpose to model format |

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | CUDA-compatible GPU recommended |
| VRAM | ~4GB minimum |
| CPU | Fallback available (slower) |

---

## Model Files

Pre-trained models are stored in the `models/` directory:

| File | Input Channels | Description |
|------|----------------|-------------|
| `six_landmarks_11_7.pth` | 1 | CTA-only model |
| `six_landmarks_2ch.pth` | 2 | CTA + segmentation model |

---

## Environment Requirements

The module requires the `arterial_dir` environment variable to be set:

```bash
export arterial_dir="/path/to/arterial"
```

---

## References

1. **MONAI**: Medical Open Network for AI - [https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

---

## See Also

- [Segmentation Module](../segmentation/README.md) - Vascular segmentation (provides input for 2-channel model)
- [Centerline Extraction Module](../centerline_extraction/README.md) - Uses landmarks for endpoint-to-endpoint extraction


