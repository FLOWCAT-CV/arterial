# Centerline Extraction Module

The centerline extraction module computes vascular centerlines from segmentation masks using the **VMTK** (Vascular Modeling Toolkit) framework. It supports both automatic full-tree extraction and targeted extraction between user-defined endpoints.

---

## Table of Contents

- [Overview](#overview)
- [Extraction Methods](#extraction-methods)
  - [Standard Pipeline (Automatic)](#standard-pipeline-automatic)
  - [Endpoint-to-Endpoint Extraction](#endpoint-to-endpoint-extraction)
- [Usage](#usage)
  - [Minimal Example (Standard Pipeline)](#minimal-example-standard-pipeline)
  - [Extracting Centerlines Between Endpoints](#extracting-centerlines-between-endpoints)
  - [Full Pipeline with Branch Model](#full-pipeline-with-branch-model)
- [Processing Pipeline](#processing-pipeline)
- [Expected Outputs](#expected-outputs)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)
- [References](#references)

---

## Overview

The centerline extraction module transforms binary vascular segmentations into geometric centerline representations. Centerlines are computed as geodesic paths through the vessel lumen, with each point annotated with the **Maximum Inscribed Sphere Radius (MISR)**, providing local vessel diameter information.

| Feature | Description |
|---------|-------------|
| **Automatic endpoint detection** | Identifies vessel endpoints from network topology |
| **Aortic arch handling** | Special logic for ascending/descending aorta |
| **Robust endpoint relocation** | Ensures endpoints lie within segmentation |
| **Branch model extraction** | Decomposes centerlines into labeled arterial segments |
| **Surface clipping** | Clips vessel surface along branch boundaries |

---

## Extraction Methods

The module provides two fundamentally different approaches to centerline extraction:

### Standard Pipeline (Automatic)

Automatically extracts the **complete vascular tree** from a segmentation mask:

1. Converts segmentation to surface mesh (Marching Cubes)
2. Extracts network topology (VMTK network extraction)
3. Auto-detects all vessel endpoints
4. Computes centerlines from a common origin to all endpoints
5. Optionally extracts branch models and clipped surfaces

**Best for:** Full vascular analysis, when all vessel branches are needed.

### Endpoint-to-Endpoint Extraction

Extracts a **single centerline** between two user-specified points:

1. Uses an existing or computed surface mesh
2. Takes a startpoint and endpoint as input (RAS coordinates)
3. Computes the geodesic centerline path between them

**Best for:** Targeted analysis, measuring specific vessel segments, or when landmarks are available.

---

## Usage

### Minimal Example (Standard Pipeline)

```python
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

# Initialize the extractor
extractor = CenterlineExtractor(
    case_dir="/path/to/case",
    mode="extracranial_vessels",
    segmentation_nifti_path="/path/to/segmentation.nii.gz"
)

# Run preprocessing (convert segmentation to surface mesh)
extractor.perform_preprocessing(save=True)

# Extract centerlines automatically
extractor.perform_centerline_extraction(save=True)

# Access results
centerline_models = extractor.centerline_model_list
```

### Extracting Centerlines Between Endpoints

For targeted extraction between known anatomical points:

```python
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

extractor = CenterlineExtractor(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)

# Define endpoints in RAS coordinates (mm)
startpoint = [-50.0, -120.0, 80.0]  # e.g., carotid bifurcation
endpoint = [-30.0, -80.0, 150.0]    # e.g., MCA bifurcation

# Extract single centerline between the two points
extractor.extract_centerline_between_endpoints(
    startpoint=startpoint,
    endpoint=endpoint,
    centerline_id="ica_to_mca",
    save=True
)

# Access the extracted centerline
individual_centerline = extractor.individual_centerlines_list[-1]
```

### Full Pipeline with Branch Model

For comprehensive vascular analysis with labeled segments:

```python
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor

extractor = CenterlineExtractor(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)

# Step 1: Preprocessing
extractor.perform_preprocessing(save=True)

# Step 2: Centerline extraction
extractor.perform_centerline_extraction(save=True)

# Step 3: Branch model extraction (segments individual arteries)
extractor.perform_branch_model_extraction(save=True)

# Step 4: Surface clipping (optional - divides surface by branches). WARNING: This step takes very long to run and is not necessary for most applications.
extractor.perform_clipped_model_extraction(save=True)

# Step 5: Postprocessing (generates segment arrays)
extractor.perform_centerline_postprocessing(save=True)

# Access unified models
branch_model = extractor.branch_model
clipped_model = extractor.clipped_model
segments_array = extractor.centerline_segments_array
```

---

## Processing Pipeline

### Standard Pipeline Flow

```
┌──────────────────────┐
│  Segmentation NIfTI  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────────────┐
│    Preprocessing     │────▶│  Surface Mesh (VTK)     │
│  • Split islands     │     │  • segmentation.vtk     │
│  • Marching Cubes    │     │  • segmentation_N.vtk   │
└──────────┬───────────┘     └─────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────────────┐
│  Network Extraction  │────▶│  Topology Network       │
│  • Endpoint detect   │     │  • network_N.vtk        │
│  • Aortic arch check │     │  • endpoints_N.json     │
└──────────┬───────────┘     └─────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────────────┐
│ Centerline Compute   │────▶│  Centerline Models      │
│  • VMTK geodesics    │     │  • centerlines_N.vtk    │
│  • MISR annotation   │     │                         │
└──────────┬───────────┘     └─────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────────────┐
│  Branch Extraction   │────▶│  Labeled Branches       │
│  • Segment labeling  │     │  • branch_model.vtk     │
│  • GroupIds/TractIds │     │  • branch_model_N.vtk   │
└──────────┬───────────┘     └─────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────────────┐
│   Postprocessing     │────▶│  Segment Arrays         │
│  • Remove overlaps   │     │  • centerline_segments  │
│  • Clean duplicates  │     │    _array.npy           │
└──────────────────────┘     └─────────────────────────┘
```

### Endpoint-to-Endpoint Pipeline Flow

```
┌──────────────────────┐     ┌─────────────────────────┐
│  Segmentation NIfTI  │     │  User-Defined Endpoints │
└──────────┬───────────┘     └────────────┬────────────┘
           │                              │
           ▼                              │
┌──────────────────────┐                  │
│    Preprocessing     │                  │
│  • Surface mesh      │                  │
└──────────┬───────────┘                  │
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Endpoint Relocation         │
           │  • Snap to segmentation      │
           │  • Center of mass adjustment │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐     ┌─────────────────────────┐
           │  Centerline Computation      │────▶│  Individual Centerline  │
           │  • Single geodesic path      │     │  • individual_centerline│
           │  • MISR annotation           │     │    _{id}.vtk            │
           └──────────────────────────────┘     └─────────────────────────┘
```

---

## Expected Outputs

### Directory Structure

After running the full pipeline, the following structure is created:

```
{case_dir}/{mode}/
├── segmentation.vtk              # Full surface mesh
├── segmentation.stl              # STL export of surface
├── segmentations/
│   ├── segmentation_0.vtk        # Main vessel island
│   ├── segmentation_1.vtk        # Secondary islands
│   └── ...
├── centerlines/
│   ├── centerlines_0.vtk         # Centerlines for island 0
│   ├── centerlines_1.vtk
│   └── ...
├── networks/
│   ├── network_0.vtk             # Topology network (skeletonized)
│   └── ...
├── endpoints/
│   ├── endpoints_0.json          # Detected endpoints (Compatible for visualization in 3D Slicer)
│   └── ...
├── branch_models/
│   ├── branch_model_0.vtk        # Branch model per island
│   └── ...
├── branch_model.vtk              # Unified branch model
├── clipped_models/
│   ├── clipped_model_0.vtk
│   └── ...
├── clipped_model.vtk             # Unified clipped surface
├── individual_centerlines/       # For endpoint-to-endpoint extraction
│   ├── individual_centerline_0.vtk
│   └── ...
└── centerline_segments_array.npy # Postprocessed segments
```

### Centerline Point Data

Each centerline VTK file contains the following point data arrays:

| Array Name | Description |
|------------|-------------|
| `MaximumInscribedSphereRadius` | Local vessel radius (mm) |

### Branch Model Cell Data

Branch models include additional cell-level annotations:

| Array Name | Description |
|------------|-------------|
| `CenterlineIds` | Unique ID for each centerline path |
| `TractIds` | Segment index along each centerline (0 = proximal) |
| `GroupIds` | Branch/vessel group identifier |
| `Blanking` | Indicates bifurcation transition zones |

### Centerline Segments Array

The `centerline_segments_array.npy` is a numpy array with shape `(N, 2)` where:
- `N` = number of unique (non-overlapping) segments
- Column 0 = coordinates array `(M, 3)` for segment points
- Column 1 = radius array `(M,)` for each point

---

## Configuration Options

### CenterlineExtractor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **Required** | Output directory for results |
| `mode` | str | `"extracranial_vessels"` | Analysis mode: `"extracranial_vessels"` or `"intracranial_vessels"` |
| `segmentation_nifti_path` | str | `None` | Path to segmentation. If `None`, expects `{case_dir}/{mode}/segmentation.nii.gz` |
| `fast_segmentation` | bool | `False` | Removes upper 10% of bounding box (simplifies intracranial processing) |

### perform_preprocessing() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_check` | bool | `True` | Validate segmentation volume is within expected range |
| `save` | bool | `True` | Save surface meshes to disk |
| `apply_bottom_cutting_to_first_model` | bool | `False` | Remove bottom region of main island (aortic arch fix) |
| `bottom_height_mm` | float | `5.0` | Height to remove when bottom cutting is enabled |

### extract_centerline_between_endpoints() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `startpoint` | list | **Required** | Source point in RAS coordinates `[x, y, z]` |
| `endpoint` | list | **Required** | Target point in RAS coordinates `[x, y, z]` |
| `centerline_id` | str | `None` | Identifier for the centerline (auto-incremented if `None`) |
| `save` | bool | `True` | Save the individual centerline to disk |

---

## Module Components

### `centerline_extractor.py`

Main orchestrator class `CenterlineExtractor` that manages the extraction pipeline.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `perform_preprocessing()` | Converts segmentation to surface mesh(es) |
| `perform_centerline_extraction()` | Automatic full-tree centerline extraction |
| `extract_centerline_between_endpoints()` | Targeted extraction between two points |
| `perform_branch_model_extraction()` | Decomposes centerlines into labeled segments |
| `perform_clipped_model_extraction()` | Clips surface mesh along branch boundaries |
| `perform_centerline_postprocessing()` | Generates non-overlapping segment arrays |

### `centerline_extraction.py`

Core extraction functions:

| Function | Description |
|----------|-------------|
| `extract_centerlines_full_cta()` | Automatic endpoint detection and centerline computation |
| `extract_centerline_between_endpoints()` | Geodesic path between specified points |

### `utils.py`

Contains the `CenterlineComputationLogic` class (adapted from VMTK Slicer extension):

| Method | Description |
|--------|-------------|
| `extract_centerline_full_cta()` | Full extraction with network-based endpoint detection |
| `extract_centerline_between_endpoints()` | Extraction with user-provided endpoints |
| `prepare_model()` | Clean, triangulate, smooth, and cap surface |
| `extract_network()` | VMTK network extraction for topology |
| `autodetect_endpoints()` | Endpoint detection from network |
| `compute_centerlines()` | Core VMTK centerline computation |

**Utility Functions:**

| Function | Description |
|----------|-------------|
| `robust_endpoint_relocation()` | Snaps endpoints to segmentation center of mass |
| `aortic_arch_endpoint_check()` | Ensures aortic arch has proper start/endpoints |
| `build_endpoints_json()` | Creates Slicer-compatible markup JSON |
| `clean_centerline()` | Consolidates overlapping points |

### `preprocessing/preprocessing.py`

Segmentation-to-surface conversion:

| Function | Description |
|----------|-------------|
| `preprocess_segmentation_for_centerline_extraction()` | Main preprocessing pipeline |
| `compute_segmentation_model()` | Marching Cubes surface extraction |

### `postprocessing/postprocessing.py`

Centerline segment analysis:

| Function | Description |
|----------|-------------|
| `compute_centerline_segments_array()` | Splits centerlines into non-overlapping segments |
| `clean_centerlines()` | Removes duplicate/invalid segments |
| `remove_intracranial_arteries()` | Filters out small cerebral vessels (extracranial mode) |

### `postprocessing/branch_and_clipped_model_extraction.py`

Branch-level processing:

| Function | Description |
|----------|-------------|
| `extract_branch_model()` | VMTK branch extraction |
| `unify_branch_models()` | Merges multiple branch models |
| `extract_clipped_model()` | VMTK surface clipping |
| `unify_clipped_models()` | Merges clipped surfaces |

---

## Error Handling

The module includes robust error handling for common issues:

| Issue | Handling |
|-------|----------|
| No endpoints found | Returns `None` for secondary islands; raises error for first island |
| Single centerline cell (first model) | Attempts preprocessing with bottom cutting |
| Branch extraction crash | Runs in subprocess to prevent segfault propagation |
| Endpoints outside segmentation | Automatic relocation to nearest valid position |

---

## References

1. **VMTK**: The Vascular Modeling Toolkit - [http://www.vmtk.org/](http://www.vmtk.org/)
2. Antiga, L., Piccinelli, M., Botti, L., Ene-Iordache, B., Remuzzi, A., & Steinman, D. A. (2008). **An image-based modeling framework for patient-specific computational hemodynamics.** *Medical & Biological Engineering & Computing*, 46(11), 1097-1112.

---

## See Also

- [Segmentation Module](../segmentation/README.md) - Vascular segmentation
- [Landmark Detection Module](../landmark_detection/README.md) - Anatomical landmark detection for endpoint specification
- [Feature Extraction Module](../feature_extraction/README.md) - Geometric feature analysis from centerlines


