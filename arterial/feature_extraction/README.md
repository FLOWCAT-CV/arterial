# Feature Extraction Module

The feature extraction module computes multi-scale vascular features from centerline graphs. It transforms raw centerline geometry into rich feature representations across three scales: **local** (node-level), **segment** (vessel-level), and **global** (anatomy-level), supporting downstream analysis such as catheter pathway planning.

---

## Table of Contents

- [Overview](#overview)
- [Feature Hierarchy](#feature-hierarchy)
  - [Local Features](#local-features)
  - [Segment Features](#segment-features)
  - [Global Features](#global-features)
- [Usage](#usage)
  - [Minimal Example](#minimal-example)
  - [Full Pipeline](#full-pipeline)
  - [Individual Centerline Analysis](#individual-centerline-analysis)
- [Processing Pipeline](#processing-pipeline)
- [Catheter Pathway Mapping](#catheter-pathway-mapping)
- [Expected Outputs](#expected-outputs)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)
- [References](#references)

---

## Overview

The feature extraction module bridges centerline geometry with clinical analysis by computing:

| Scale | Description | Use Case |
|-------|-------------|----------|
| **Local** | Per-node features (curvature, direction, intensity) | Detailed navigation analysis |
| **Segment** | Per-vessel features (length, tortuosity, diameters) | Vessel characterization |
| **Global** | Anatomical variants (aortic arch type, bovine arch) | Patient classification |

The module operates on the output of the **centerline extraction** and **vessel labelling** modules, requiring:
1. A centerline segments array (`.npy`)
2. A predicted vessel graph (`.pickle`)
3. A branch model (`.vtk`)
4. The original CTA image (`.nii.gz`)

---

## Feature Hierarchy

### Local Features

Local features are computed at each node of the centerline graph, separately for **femoral** and **radial** access configurations.

| Feature | Description | Units |
|---------|-------------|-------|
| `pos r`, `pos a`, `pos s` | Node position in RAS coordinates | mm |
| `radius` | Inscribed sphere radius | mm |
| `segment length` | Average edge length to neighbors | mm |
| `curvature` | Local curvature at node | 1/mm |
| `torsion` | Local torsion at node | 1/mm² |
| `direction module` | Direction vector magnitude | mm |
| `direction polar` | Polar angle of direction | radians |
| `direction azimuth` | Azimuthal angle of direction | radians |
| `blanking` | Bifurcation blanking indicator | boolean |
| `HU intensity` | CTA Hounsfield Unit at node | HU |
| `vessel_type` | Predicted vessel label | categorical |

**Cumulative features** are also computed along the catheter path:
- `cumulative segment length`: Total path length from access point
- `cumulative curvature`: Integrated curvature along path

### Segment Features

Segment features characterize entire vascular segments by vessel type.

| Feature | Description | Units |
|---------|-------------|-------|
| `length` | Total segment length | mm |
| `mean_diameter` | Average vessel diameter | mm |
| `std_diameter` | Diameter standard deviation | mm |
| `min_diameter` | Minimum diameter | mm |
| `max_diameter` | Maximum diameter | mm |
| `proximal_diameter` | Diameter at proximal end | mm |
| `distal_diameter` | Diameter at distal end | mm |
| `min_max_diameter_ratio` | Ratio of min to max diameter | ratio |
| `tortuosity_index` | Path length / Euclidean distance | ratio |
| `bending_length` | Effective bending length | mm |
| `cumulative_curvature` | Integrated curvature | radians |
| `tortuosity_index_5_cm` | Tortuosity of first 5 cm | ratio |
| `min_polar_angle` | Minimum polar angle along path | radians |
| `accumulated_polar_angle_differential` | Cumulative angular change | radians |
| `polar_angle` | Overall polar direction | radians |
| `azimuthal_angle` | Overall azimuthal direction | radians |
| `diameter_last_5mm` | Diameter at last 5 mm | mm |
| `diameter_last_10mm` | Diameter at last 10 mm | mm |
| `tortuosity_index_last_10mm` | Tortuosity of last 10 mm | ratio |
| `tortuosity_index_last_20mm` | Tortuosity of last 20 mm | ratio |
| `tortuosity_index_last_30mm` | Tortuosity of last 30 mm | ratio |
| `curvature_energy` | Integrated squared curvature | 1/mm |

### Global Features

Global features describe anatomical variants at the patient level.

| Feature | Description | Values |
|---------|-------------|--------|
| `aortic_arch_type` | Aortic arch classification | Type I, II, III |
| `bovine_arch` | Bovine arch variant | True/False |
| `arsa` | Aberrant Right Subclavian Artery | True/False |

---

## Usage

### Minimal Example

```python
from arterial.feature_extraction import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)

# Build local graph from centerlines
extractor.build_local_graph()

# Extract features at all scales
extractor.extract_local_features()
extractor.extract_segment_features()
extractor.extract_global_features()
```

### Full Pipeline

```python
from arterial.feature_extraction import FeatureExtractor

# Initialize with custom paths
extractor = FeatureExtractor(
    case_dir="/path/to/case",
    mode="extracranial_vessels",
    sampling_distance_mm=2,
    cta_nifti_path="/path/to/cta.nii.gz",
    centerline_segments_array_path="/path/to/centerline_segments_array.npy",
    branch_model_path="/path/to/branch_model.vtk",
    segments_graph_pred_path="/path/to/segments_graph_pred.pickle"
)

# Build resampled local graph
extractor.build_local_graph(resample=True, save=True)

# Extract all feature levels
extractor.extract_local_features(save=True)
extractor.extract_segment_features(save=True)
extractor.extract_global_features(save=True)

# Map catheter pathways (supersegments)
extractor.extract_supersegments(save=True)

```

### Individual Centerline Analysis

For targeted vessel analysis (e.g., intracranial vessels), use the individual centerline method:

```python
from arterial.feature_extraction import FeatureExtractor
from arterial.io.load_and_save_operations import load_vtkpolydata

# Initialize extractor
extractor = FeatureExtractor(
    case_dir="/path/to/case",
    mode="intracranial_vessels",
    sampling_distance_mm=0.5
)

# Load a single centerline model
centerline_model = load_vtkpolydata("/path/to/centerline.vtk")

# Build and featurize individual centerline
extractor.build_and_featurize_individual_centerline_graph(
    centerline_model=centerline_model,
    radius_array_name="MaximumInscribedSphereRadius",
    centerline_id="LMCA_segment",
    save=True
)

# Access the featurized graph
individual_graph = extractor.individual_centerline_graph
```

---

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  centerline_segments │    │   segments_graph     │    │   branch_model  │
│     _array.npy       │    │    _pred.pickle      │    │      .vtk       │
└──────────┬───────────┘    └──────────┬───────────┘    └────────┬────────┘
           │                           │                         │
           └───────────────────────────┼─────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │    build_local_graph    │
                         │   (resampling, merging) │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │   extract_local_features│◄───── CTA image (intensity)
                         │   (node-level features) │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │ extract_segment_features│
                         │  (vessel-level features)│
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │ extract_global_features │
                         │  (anatomical variants)  │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │  extract_supersegments  │
                         │  (catheter pathways)    │
                         └───────────┬─────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │  local_graph     │              │   supersegments/ │
          │    .pickle       │              │     (8 configs)  │
          └──────────────────┘              └──────────────────┘
```

---

## Catheter Pathway Mapping

The module maps all possible **catheter pathways** (supersegments) for endovascular procedures by combining three binary variables:

| Variable | Options | Description |
|----------|---------|-------------|
| **Access** | Femoral / Radial | Entry point for catheter |
| **Laterality** | Right / Left | Target hemisphere |
| **Circulation** | Anterior / Posterior | Target circulation |

This produces **8 unique configurations**:

```
                    SUPERSEGMENT CONFIGURATIONS
                    
Access      Laterality      Circulation      Output File
─────────────────────────────────────────────────────────
Femoral     Right           Anterior         femoral + right + anterior.pickle
Femoral     Right           Posterior        femoral + right + posterior.pickle
Femoral     Left            Anterior         femoral + left + anterior.pickle
Femoral     Left            Posterior        femoral + left + posterior.pickle
Radial      Right           Anterior         radial + right + anterior.pickle
Radial      Right           Posterior        radial + right + posterior.pickle
Radial      Left            Anterior         radial + left + anterior.pickle
Radial      Left            Posterior        radial + left + posterior.pickle
```

Each supersegment is a subgraph of the local_graph containing only the nodes traversed along that specific catheter pathway.

---

## Expected Outputs

### Directory Structure

```
case_dir/
└── {mode}/
    ├── local_graph.pickle           # Featurized centerline graph
    ├── local_graph.png              # Graph visualization
    ├── single_segments/             # Per-cell-id segment graphs
    │   ├── AA.pickle
    │   ├── LCCA.pickle
    │   ├── LICA.pickle
    │   └── ...
    ├── single_segments.png          # Segment visualization
    ├── supersegments/               # Catheter pathway graphs
    │   ├── femoral + right + anterior.pickle
    │   ├── femoral + left + anterior.pickle
    │   └── ... (8 total)
    └── supersegments.png            # Pathway visualization
```

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `local_graph.pickle` | NetworkX Graph | Full featurized centerline graph |
| `local_graph.png` | PNG | Sagittal view visualization |
| `single_segments/*.pickle` | NetworkX Graph | Individual vessel segments |
| `single_segments.png` | PNG | All segments colored by type |
| `supersegments/*.pickle` | NetworkX Graph | Catheter pathway subgraphs |
| `supersegments.png` | PNG | All 8 pathway configurations |

### Accessing Features

```python
import pickle

# Load featurized graph
with open("local_graph.pickle", "rb") as f:
    local_graph = pickle.load(f)

# Access node features
node_0_features = local_graph.nodes[0]["features femoral"]
print(f"Curvature: {node_0_features['curvature']}")
print(f"Radius: {node_0_features['radius']}")

# Access segment features
segment_features = local_graph.graph["segment_features"]
print(f"LCCA length: {segment_features['LCCA']['length']} mm")
print(f"LCCA tortuosity: {segment_features['LCCA']['tortuosity_index']}")

# Access global features
print(f"Aortic arch type: {local_graph.graph['aortic_arch_type']}")
print(f"Bovine arch: {local_graph.graph['bovine_arch']}")
```

---

## Configuration Options

### FeatureExtractor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **required** | Path to case directory |
| `mode` | str | `"extracranial_vessels"` | `"extracranial_vessels"` or `"intracranial_vessels"` |
| `sampling_distance_mm` | float | `2` | Distance between resampled nodes |
| `cta_nifti_path` | str | `None` | Path to CTA NIfTI file |
| `centerline_segments_array_path` | str | `None` | Path to centerline array |
| `branch_model_path` | str | `None` | Path to branch model VTK |
| `segments_graph_pred_path` | str | `None` | Path to predicted graph |

### Method Parameters

| Method | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `build_local_graph` | `resample` | `True` | Resample to `sampling_distance_mm` |
| `build_local_graph` | `save` | `True` | Save outputs to disk |
| `extract_*` | `save` | `True` | Save updated graph to disk |

### Sampling Distance Recommendations

| Analysis Type | Recommended Distance | Rationale |
|--------------|---------------------|-----------|
| Extracranial vessels | 2 mm | Sufficient for large vessels |
| Intracranial vessels | 0.5 mm | Higher resolution for small vessels |
| Tortuosity analysis | 1 mm | Balance of resolution and smoothing |

---

## Module Components

```
feature_extraction/
├── __init__.py
├── feature_extractor.py      # Main FeatureExtractor class
├── graph_builder.py          # Graph construction functions
├── utils.py                  # Graph utilities and resampling
├── local_features/
│   ├── __init__.py
│   ├── feature_extraction.py # Local feature computation
│   └── utils.py              # Node featurization helpers
├── segment_features/
│   ├── __init__.py
│   ├── feature_extraction.py # Segment feature computation
│   └── utils.py              # Segment extraction and features
├── global_features/
│   ├── __init__.py
│   ├── feature_extraction.py # Global feature computation
│   └── utils.py              # Anatomical variant detection
└── mapping/
    ├── __init__.py
    ├── mapping.py            # Supersegment extraction
    └── utils.py              # Pathway building utilities
```

### Key Functions

| Function | Location | Description |
|----------|----------|-------------|
| `build_local_graph` | `graph_builder.py` | Constructs dense centerline graph |
| `perform_local_feature_extraction` | `local_features/` | Computes per-node features |
| `perform_segment_feature_extraction` | `segment_features/` | Computes per-vessel features |
| `perform_global_feature_extraction` | `global_features/` | Computes anatomical variants |
| `extract_arterial_mapping` | `mapping/` | Generates supersegments |
| `resample_centerline_segments_array` | `utils.py` | Resamples centerlines |

---

## References

- The feature extraction module is designed for endovascular thrombectomy planning
- Curvature and torsion are computed using Frenet-Serret formulas
- Tortuosity index follows standard vascular analysis conventions
- Aortic arch classification follows radiological guidelines (Type I, II, III)

---

## See Also

- [Centerline Extraction Module](../centerline_extraction/README.md) - Generates input centerlines
- [Vessel Labelling Module](../vessel_labelling/README.md) - Provides vessel type predictions
- [Segmentation Module](../segmentation/README.md) - Generates initial vessel masks

