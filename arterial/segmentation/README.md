# Segmentation Module

The segmentation module provides deep learning-based vessel segmentation for CT Angiography (CTA) images. It leverages **nnU-Net v2** as the underlying framework to perform accurate segmentation of both intracranial and extracranial vasculature.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Minimal Example](#minimal-example)
  - [Extracranial Vessel Segmentation](#extracranial-vessel-segmentation)
  - [Intracranial Vessel Segmentation](#intracranial-vessel-segmentation)
  - [Fast Segmentation Mode](#fast-segmentation-mode)
- [Processing Pipeline Variations](#processing-pipeline-variations)
- [Expected Outputs](#expected-outputs)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)
- [References](#references)

---

## Overview

The segmentation module performs binary vascular segmentation on CTA images, producing masks that identify blood vessels within the head and neck region. It supports two primary modes of operation:

| Mode | Description | Resolution | Use Case |
|------|-------------|------------|----------|
| `extracranial_vessels` | Segments vessels from the neck to the Circle of Willis | Hybrid (3d_fullres + 3d_lowres) | Full carotid tree analysis |
| `intracranial_vessels` | Segments vessels within the cranial cavity | 3d_fullres | Cerebral vascular analysis |

---

## Architecture

The module uses pre-trained **nnU-Net v2** models organized as follows:

```
models/
├── extracranial_vessels/
│   └── nnUNetTrainer__nnUNetPlans__3d_lowres/
├── intracranial_vessels/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
└── totalsegmentator_mandible/
    └── nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres/
```

The module intelligently combines different resolution models to balance accuracy and computational efficiency:

- **3d_fullres**: High-resolution model for detailed intracranial segmentation
- **3d_lowres**: Lower-resolution model for faster processing of larger neck regions
- **TotalSegmentator**: Used for cranium detection to enable head/neck slicing

---

## Usage

### Minimal Example

```python
from arterial.segmentation.segmenter import VesselSegmenter

# Initialize the segmenter
segmenter = VesselSegmenter(
    case_dir="/path/to/case",
    mode="extracranial_vessels",
    cta_nifti_path="/path/to/cta.nii.gz"
)

# Run segmentation
segmenter.segment_vessels_from_cta(save=True)

# Access results
segmentation_array = segmenter.segmentation_array
segmentation_nifti = segmenter.segmentation_nifti
```

### Extracranial Vessel Segmentation

For full head-and-neck CTA analysis:

```python
from arterial.segmentation.segmenter import VesselSegmenter

segmenter = VesselSegmenter(
    case_dir="/path/to/output",
    mode="extracranial_vessels",
    cta_nifti_path="/path/to/cta.nii.gz",
    fast_segmentation=False  # Use high-resolution for head region
)

segmenter.segment_vessels_from_cta(save=True)
```

This mode automatically:
1. Detects the cranium using TotalSegmentator
2. Slices the CTA into head and neck regions
3. Applies 3d_fullres to the head and 3d_lowres to the neck
4. Merges segmentations at the optimal slice (highest Dice similarity)

### Intracranial Vessel Segmentation

For cerebral vasculature only:

```python
segmenter = VesselSegmenter(
    case_dir="/path/to/output",
    mode="intracranial_vessels",
    cta_nifti_path="/path/to/cta.nii.gz"
)

segmenter.segment_vessels_from_cta(save=True)

# Optionally save the cropped head CTA
segmenter.save_head_cta_nifti("/path/to/head_cta.nii.gz")
```

### Fast Segmentation Mode

For rapid processing when high accuracy is not critical:

```python
segmenter = VesselSegmenter(
    case_dir="/path/to/output",
    mode="extracranial_vessels",
    fast_segmentation=True  # Uses 3d_lowres for entire volume
)

segmenter.segment_vessels_from_cta(save=True)
```

> ⚠️ **Note**: Fast segmentation mode produces less reliable cerebral artery segmentation. Use only when processing speed is prioritized over accuracy.

---

## Processing Pipeline Variations

The module adapts its processing strategy based on configuration:

### Standard Pipeline (Default)

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Load CTA   │────▶│  Slice Head/    │────▶│  Segment Head   │
│             │     │  Neck Regions   │     │  (3d_fullres)   │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                    ┌─────────────────┐              │
                    │  Segment Neck   │◀─────────────┘
                    │  (3d_lowres)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐     ┌─────────────────┐
                    │  Merge at Best  │────▶│  Save Output    │
                    │  Dice Slice     │     │                 │
                    └─────────────────┘     └─────────────────┘
```

### Fast Pipeline (`fast_segmentation=True`)

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Load CTA   │────▶│  Segment Whole  │────▶│  Save Output    │
│             │     │  (3d_lowres)    │     │                 │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

### No-Slicing Pipeline (`no_slicing=True`)

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Load CTA   │────▶│  Segment Whole  │────▶│  Save Output    │
│             │     │  (3d_fullres)   │     │                 │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Expected Outputs

After running segmentation, the following files are generated:

| Output File | Description |
|-------------|-------------|
| `{case_dir}/{mode}/segmentation.nii.gz` | Binary segmentation mask (NIfTI format) |
| `{case_dir}/{mode}/segmentation_probabilities.nii.gz` | Probability map (if `return_probabilities=True`) |
| `{case_dir}/head_cta.nii.gz` | Cropped head CTA (if saved) |

### Segmentation Mask Values

| Value | Meaning |
|-------|---------|
| 0 | Background |
| 1 | Vessel |

### Probability Map

When `return_probabilities=True`, the output contains continuous values [0, 1] representing the model's confidence that each voxel belongs to the vessel class.

---

## Configuration Options

### VesselSegmenter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **Required** | Output directory for results |
| `mode` | str | `"extracranial_vessels"` | Segmentation mode: `"extracranial_vessels"` or `"intracranial_vessels"` |
| `cta_nifti_path` | str | `None` | Path to input CTA. If `None`, expects `{case_dir}/cta.nii.gz` |
| `fast_segmentation` | bool | `False` | Use low-resolution model for entire volume |
| `use_vanilla_nnunet` | bool | `True` | Use standard nnU-Net trainer (`all` folds) vs clDice trainer (`fold_0`) |
| `no_slicing` | bool | `False` | Skip head/neck slicing, use full volume |
| `set_threshold_099` | bool | `False` | Apply 0.99 probability threshold for high-confidence segmentation |

### segment_vessels_from_cta() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_probabilities` | bool | `False` | Generate and save probability maps |
| `save` | bool | `True` | Save segmentation to disk |

---

## Module Components

### `segmenter.py`

Contains the main `VesselSegmenter` class that orchestrates the segmentation pipeline.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `segment_vessels_from_cta()` | Main entry point for vessel segmentation |
| `slice_cta()` | Separates CTA into head and neck volumes |
| `save_segmentation_nifti()` | Saves segmentation to disk |
| `save_head_cta_nifti()` | Saves cropped head CTA |

### `inference.py`

Handles the nnU-Net inference logic.

**Key Function:**

```python
perform_single_inference_nnunet(
    img_array,           # 3D numpy array of CTA
    img_affine,          # 4x4 affine transformation matrix
    mode,                # "extracranial_vessels" or "intracranial_vessels"
    nnunet_mode,         # "3d_lowres" or "3d_fullres"
    use_vanilla_nnunet,  # Use standard or clDice trainer
    return_probabilities # Return probability maps
)
```

**Returns:**
- `segmentation_nifti`: NIfTI image object
- `segmentation_array`: 3D numpy array
- `probabilities_nifti`: NIfTI probability map (or `None`)

### `utils.py`

Utility functions for image processing:

| Function | Description |
|----------|-------------|
| `slice_cta_head_and_neck()` | Splits CTA into head and neck regions using cranium detection |
| `join_head_and_neck_segmentations()` | Merges separate segmentations at optimal boundary |
| `get_largest_connected_component()` | Filters small components, retaining largest structure |
| `run_cranium_segmentation_totalsegmentator()` | Detects cranium using TotalSegmentator model |

---

## Environment Requirements

The module requires the `arterial_dir` environment variable to be set, pointing to the Arterial installation directory containing the pre-trained models:

```bash
export arterial_dir="/path/to/arterial"
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU strongly recommended
- **Memory**: Minimum 8GB VRAM for 3d_fullres models
- **CPU**: Fallback option available but significantly slower

---

## References

1. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** *Nature methods*, 18(2), 203-211.

---

## See Also

- [Landmark Detection Module](../landmark_detection/README.md) - Anatomical landmark detection
- [Centerline Extraction Module](../centerline_extraction/README.md) - Vascular centerline extraction
- [Feature Extraction Module](../feature_extraction/README.md) - Geometric feature analysis

