# Run Module (ArterialProcessor)

The run module provides the `ArterialProcessor` class—the central orchestrator that coordinates all Arterial modules into a unified analysis pipeline. It enables end-to-end vascular analysis from a single CTA image through command-line or programmatic invocation.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Command-Line Usage](#command-line-usage)
  - [Basic Usage](#basic-usage)
  - [Full Options](#full-options)
  - [Common Workflows](#common-workflows)
- [Programmatic Usage](#programmatic-usage)
- [Pipeline Steps](#pipeline-steps)
  - [1. Segmentation](#1-segmentation)
  - [2. Centerline Extraction](#2-centerline-extraction)
  - [3. Landmark Detection](#3-landmark-detection)
  - [4. Vessel Labelling](#4-vessel-labelling)
  - [5. Feature Extraction](#5-feature-extraction)
  - [6. Access Prediction](#6-access-prediction)
- [Skip Flags](#skip-flags)
- [Expected Outputs](#expected-outputs)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)

---

## Overview

The `ArterialProcessor` wraps all Arterial modules into a single orchestration class:

```
┌────────────────────────────────────────────────────--─────────────┐
│                      ArterialProcessor                            │
│                                                                   │
│   ┌──────────────-┐  ┌──────────────┐  ┌──────────────-┐          │
│   │VesselSegmenter│  │CenterlineExt.│  │LandmarkDetect │          │
│   └──────────────-┘  └──────────────┘  └──────────────-┘          │
│                                                                   │
│   ┌──────────────-┐  ┌──────────────┐  ┌──────────────-┐          │
│   │VesselLabeller │  │FeatureExtract│  │AccessPredictor│          │
│   └──────────────-┘  └──────────────┘  └──────────────-┘          │
└───────────────────────────────────────────────────────────────────┘
```

| Module | Purpose | Output |
|--------|---------|--------|
| **Segmentation** | Deep learning vessel segmentation | Binary mask (`.nii.gz`) |
| **Centerline Extraction** | VMTK-based centerline computation | VTK models, numpy arrays |
| **Landmark Detection** | Anatomical landmark identification | JSON coordinates |
| **Vessel Labelling** | GNN-based vessel classification | Labeled graphs |
| **Feature Extraction** | Multi-scale feature computation | Featurized graphs |
| **Access Prediction** | Catheter accessibility scoring | Predictions, attention maps |

---

## Pipeline Architecture

```
                           ┌─────────────────┐
                           │   CTA Image     │
                           │   (.nii.gz)     │
                           └────────┬────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 1: SEGMENTATION                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  VesselSegmenter.segment_vessels_from_cta()                         │  │
│  │  • nnU-Net inference (hybrid/fast modes)                            │  │
│  │  • Head/neck slicing for extracranial vessels                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                           Output: {mode}/segmentation.nii.gz              │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 2: CENTERLINE EXTRACTION                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  CenterlineExtractor.perform_centerline_extraction()                │  │
│  │  CenterlineExtractor.perform_branch_model_extraction()              │  │
│  │  CenterlineExtractor.perform_centerline_postprocessing()            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                           Output: centerlines/, branch_model.vtk          │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 3: LANDMARK DETECTION                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  LandmarkDetector.detect_landmarks_on_cta()                         │  │
│  │  CenterlineExtractor.extract_centerline_between_endpoints()         │  │
│  │  FeatureExtractor.build_and_featurize_individual_centerline_graph() │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                           Output: landmarks.json, individual_centerlines/ │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 4: VESSEL LABELLING                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  VesselLabeller.build_segments_graph()                              │  │
│  │  VesselLabeller.predict_vessel_types()                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                           Output: segments_graph_pred.pickle              │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 5: FEATURE EXTRACTION (extracranial only)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  FeatureExtractor.build_local_graph()                               │  │
│  │  FeatureExtractor.extract_local_features()                          │  │
│  │  FeatureExtractor.extract_segment_features()                        │  │
│  │  FeatureExtractor.extract_global_features()                         │  │
│  │  FeatureExtractor.extract_supersegments()                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                           Output: local_graph.pickle, supersegments/      │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 6: ACCESS PREDICTION (extracranial only)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  AccessPredictor.predict_accessibility()                            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                           Output: access_prediction/                      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Command-Line Usage

### Basic Usage

```bash
# Minimal command - full pipeline for extracranial vessels
python perform_analysis.py -cd /path/to/case_dir

# Specify CTA path explicitly
python perform_analysis.py -cd /path/to/case_dir -cnp /path/to/cta.nii.gz

# Intracranial vessel analysis
python perform_analysis.py -cd /path/to/case_dir -m intracranial_vessels
```

### Full Options

```bash
python perform_analysis.py \
    -cd /path/to/case_dir \           # Required: case directory
    -cnp /path/to/cta.nii.gz \        # Optional: CTA path (default: case_dir/cta.nii.gz)
    -m extracranial_vessels \         # Optional: mode (extracranial_vessels|intracranial_vessels)
    -sd 2 \                           # Optional: sampling distance in mm (default: 2)
    -fast \                           # Optional: use fast segmentation
    -ss \                             # Optional: skip segmentation
    -sce \                            # Optional: skip centerline extraction
    -sb \                             # Optional: skip branching
    -sc \                             # Optional: skip clipping
    -svl \                            # Optional: skip vessel labelling
    -sfe \                            # Optional: skip feature extraction
    -sap \                            # Optional: skip access prediction
    -sld \                            # Optional: skip landmark detection
    -clnn \                           # Optional: use clDice-trained nnU-Net
    -ns \                             # Optional: no slicing (for head-only CTA)
    -s99                              # Optional: set segmentation threshold to 0.99
```

### Common Workflows

#### Complete Analysis (Default)

```bash
# Full pipeline from CTA to access prediction
python perform_analysis.py -cd /path/to/case
```

#### Re-run After Segmentation

```bash
# Skip segmentation if already computed
python perform_analysis.py -cd /path/to/case -ss
```

#### Fast Processing

```bash
# Use fast segmentation mode (single-resolution)
python perform_analysis.py -cd /path/to/case -fast
```

#### Intracranial Analysis

```bash
# Process intracranial vessels only
python perform_analysis.py -cd /path/to/case -m intracranial_vessels -ns
```

#### Centerlines and Labelling Only

```bash
# Skip segmentation, feature extraction, and access prediction
python perform_analysis.py -cd /path/to/case -ss -sfe -sap
```

---

## Programmatic Usage

### Basic Example

```python
from argparse import Namespace
from arterial.run.processor import ArterialProcessor

# Create arguments namespace
args = Namespace(
    case_dir="/path/to/case",
    cta_nifti_path=None,  # Will default to case_dir/cta.nii.gz
    mode="extracranial_vessels",
    sampling_distance_mm=2,
    fast_segmentation=False,
    skip_segmentation=False,
    skip_centerline_extraction=False,
    skip_branching=False,
    skip_clipping=False,
    skip_vessel_labelling=False,
    skip_feature_extraction=False,
    skip_access_prediction=False,
    skip_landmark_detection=False,
    cl_dice_nnunet=False,
    no_slicing=False,
    set_threshold_099=False
)

# Initialize processor
processor = ArterialProcessor(args)

# Run full analysis
timing_results = processor.perform_analysis()

# Access timing information
print(f"Segmentation: {timing_results['segmentation_time']:.2f}s")
print(f"Centerline extraction: {timing_results['centerline_extraction_time']:.2f}s")
print(f"Total: {timing_results['total_time']:.2f}s")
```

### Running Individual Steps

```python
from argparse import Namespace
from arterial.run.processor import ArterialProcessor

args = Namespace(
    case_dir="/path/to/case",
    cta_nifti_path=None,
    mode="extracranial_vessels",
    sampling_distance_mm=2,
    fast_segmentation=False,
    skip_segmentation=True,  # Skip for this example
    skip_centerline_extraction=True,
    skip_branching=True,
    skip_clipping=True,
    skip_vessel_labelling=True,
    skip_feature_extraction=True,
    skip_access_prediction=True,
    skip_landmark_detection=True,
    cl_dice_nnunet=False,
    no_slicing=False,
    set_threshold_099=False
)

processor = ArterialProcessor(args)

# Run individual steps as needed
processor.perform_segmentation()
processor.perform_centerline_extraction()
processor.perform_landmark_detection()
processor.perform_vessel_labelling()
processor.perform_feature_extraction()
processor.perform_access_prediction()
```

### Accessing Module Objects

```python
# After initialization, access individual module objects
processor = ArterialProcessor(args)

# Access the vessel segmenter
segmenter = processor.vessel_segmenter

# Access the centerline extractor
centerline_ext = processor.centerline_extractor

# Access the feature extractor (after running)
processor.perform_feature_extraction()
local_graph = processor.feature_extractor.local_graph
```

---

## Pipeline Steps

### 1. Segmentation

**Method:** `perform_segmentation()`

Performs deep learning-based vessel segmentation using nnU-Net.

**Outputs:**
- `{mode}/segmentation.nii.gz` - Binary vessel mask

**Skip flag:** `-ss` / `--skip_segmentation`

---

### 2. Centerline Extraction

**Method:** `perform_centerline_extraction()`

Extracts vessel centerlines using VMTK, performs branching and clipping.

**Outputs:**
- `{mode}/centerlines/centerlines_{idx}.vtk` - Individual centerline models
- `{mode}/branch_models/branch_model_{idx}.vtk` - Branch models per centerline
- `{mode}/branch_model.vtk` - Merged branch model
- `{mode}/centerline_segments_array.npy` - Numpy array for downstream processing

**Skip flags:**
- `-sce` / `--skip_centerline_extraction` - Skip entire step
- `-sb` / `--skip_branching` - Skip branch model extraction
- `-sc` / `--skip_clipping` - Skip surface clipping

---

### 3. Landmark Detection

**Method:** `perform_landmark_detection()`

Detects anatomical landmarks and extracts targeted centerlines between them.

**Detected Landmarks (extracranial mode):**
- Terminal ICA bifurcations (L/R)
- External ICA bifurcations (L/R)
- MCA bifurcations (L/R)

**Extracted Centerlines:**
| ID | Start | End |
|----|-------|-----|
| `l-ica` | L-EICA | L-TICA |
| `r-ica` | R-EICA | R-TICA |
| `l-mca` | L-TICA | L-MCA |
| `r-mca` | R-TICA | R-MCA |
| `l-ica_mca` | L-EICA | L-MCA |
| `r-ica_mca` | R-EICA | R-MCA |

**Outputs:**
- `{mode}/landmarks.json` - Detected landmark coordinates
- `{mode}/landmarks_slicer.json` - 3D Slicer format
- `{mode}/individual_centerlines/individual_centerline_{id}.vtk` - Centerline models
- `{mode}/individual_centerlines/individual_centerline_{id}.pickle` - Featurized graphs

**Skip flag:** `-sld` / `--skip_landmark_detection`

---

### 4. Vessel Labelling

**Method:** `perform_vessel_labelling()`

Classifies vessel segments using Graph Neural Networks.

**Outputs:**
- `{mode}/segments_graph.pickle` - Unlabeled segments graph
- `{mode}/segments_graph.png` - Graph visualization
- `{mode}/segments_graph_pred.pickle` - Labeled segments graph
- `{mode}/segments_graph_pred.png` - Labeled visualization

**Skip flag:** `-svl` / `--skip_vessel_labelling`

---

### 5. Feature Extraction

**Method:** `perform_feature_extraction()`

> **Note:** Only runs for `extracranial_vessels` mode.

Computes multi-scale features and maps catheter pathways.

**Outputs:**
- `{mode}/local_graph.pickle` - Featurized centerline graph
- `{mode}/local_graph.png` - Graph visualization
- `{mode}/single_segments/` - Per-vessel-type segments
- `{mode}/single_segments.png` - Segment visualization
- `{mode}/supersegments/` - Catheter pathway subgraphs (8 configurations)
- `{mode}/supersegments.png` - Pathway visualization

**Skip flag:** `-sfe` / `--skip_feature_extraction`

---

### 6. Access Prediction

**Method:** `perform_access_prediction()`

> **Note:** Only runs for `extracranial_vessels` mode.

Predicts catheter accessibility and generates attention maps.

**Outputs (per access/side):**
- `access_prediction/{access}_{side}/access_prediction.json` - Predictions
- `access_prediction/{access}_{side}/attention_map.pickle` - Attention weights
- `access_prediction/{access}_{side}/attention_map.png` - Attention visualization
- `access_prediction/{access}_{side}/attention_map.vtk` - 3D attention map
- `access_prediction/{access}_{side}/combined_plot.png` - Combined visualization

**Skip flag:** `-sap` / `--skip_access_prediction`

---

## Skip Flags

| Flag | Long Form | Effect |
|------|-----------|--------|
| `-ss` | `--skip_segmentation` | Skip nnU-Net inference |
| `-sce` | `--skip_centerline_extraction` | Skip VMTK centerline extraction |
| `-sb` | `--skip_branching` | Skip branch model extraction |
| `-sc` | `--skip_clipping` | Skip surface model clipping |
| `-svl` | `--skip_vessel_labelling` | Skip GNN vessel classification |
| `-sfe` | `--skip_feature_extraction` | Skip feature extraction |
| `-sap` | `--skip_access_prediction` | Skip access prediction |
| `-sld` | `--skip_landmark_detection` | Skip landmark detection |

---

## Expected Outputs

### Complete Output Structure

```
case_dir/
├── cta.nii.gz                              # Input CTA image
├── {mode}/segmentation.nii.gz              # Vessel segmentation mask
└── {mode}/
    ├── centerlines/
    │   ├── centerlines_0.vtk
    │   └── ...
    ├── segmentations/
    │   ├── segmentation_0.vtk
    │   └── ...
    ├── branch_models/
    │   ├── branch_model_0.vtk
    │   └── ...
    ├── branch_model.vtk                    # Merged branch model
    ├── centerline_segments_array.npy       # Centerline array
    ├── landmarks/
    │   ├── landmarks.json
    │   └── landmarks_slicer.json
    ├── individual_centerlines/
    │   ├── individual_centerline_l-ica.vtk
    │   ├── individual_centerline_l-ica.pickle
    │   └── ...
    ├── segments_graph.pickle
    ├── segments_graph.png
    ├── segments_graph_pred.pickle
    ├── segments_graph_pred.png
    ├── local_graph.pickle                  # Featurized graph
    ├── local_graph.png
    ├── single_segments/
    │   ├── AA.pickle
    │   ├── LCCA.pickle
    │   └── ...
    ├── single_segments.png
    ├── supersegments/
    │   ├── femoral + left + anterior.pickle
    │   ├── femoral + right + anterior.pickle
    │   └── ... (8 total)
    ├── supersegments.png
    └── access_prediction/
        ├── femoral_left/
        │   ├── access_prediction.json
        │   ├── attention_map.pickle
        │   ├── attention_map.png
        │   └── ...
        └── femoral_right/
            └── ...
```

---

## Configuration Options

### ArterialProcessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **required** | Path to case directory |
| `cta_nifti_path` | str | `case_dir/cta.nii.gz` | Path to CTA NIfTI |
| `mode` | str | `extracranial_vessels` | Analysis mode |
| `sampling_distance_mm` | float | `2` | Centerline sampling distance |
| `fast_segmentation` | bool | `False` | Use single-resolution segmentation |
| `use_vanilla_nnunet` | bool | `True` | Use standard nnU-Net (vs clDice) |
| `no_slicing` | bool | `False` | Skip head/neck slicing |
| `set_threshold_099` | bool | `False` | Use 0.99 segmentation threshold |

### Analysis Modes

| Mode | Landmarks | Feature Extraction | Access Prediction |
|------|-----------|-------------------|-------------------|
| `extracranial_vessels` | 6 landmarks | ✓ Full | ✓ |
| `intracranial_vessels` | 4 landmarks | Individual centerlines only | ✗ |

---

## Module Components

```
run/
├── __init__.py
└── processor.py          # ArterialProcessor class

perform_analysis.py       # Command-line entry point
```

### ArterialProcessor Methods

| Method | Description |
|--------|-------------|
| `perform_analysis()` | Run complete pipeline, return timing dict |
| `perform_segmentation()` | Run segmentation step |
| `perform_centerline_extraction()` | Run centerline extraction |
| `perform_landmark_detection()` | Run landmark detection |
| `perform_vessel_labelling()` | Run vessel labelling |
| `perform_feature_extraction()` | Run feature extraction |
| `perform_access_prediction()` | Run access prediction |

### Timing Output

The `perform_analysis()` method returns a dictionary with execution times:

```python
{
    "segmentation_time": float,           # seconds
    "centerline_extraction_time": float,
    "landmark_detection_time": float,
    "vessel_labelling_time": float,
    "feature_extraction_time": float,     # 0 for intracranial
    "access_prediction_time": float,      # 0 for intracranial
    "total_time": float
}
```

---

## See Also

- [Segmentation Module](segmentation.md) - Vessel segmentation details
- [Centerline Extraction Module](centerline_extraction.md) - Centerline computation
- [Landmark Detection Module](landmark_detection.md) - Anatomical landmark detection
- [Vessel Labelling Module](vessel_labelling.md) - Vessel classification
- [Feature Extraction Module](feature_extraction.md) - Feature computation


