# Vessel Labelling Module

The vessel labelling module performs automatic anatomical labelling of vascular segments using Graph Neural Networks (GNNs). It classifies each vessel segment in the centerline graph with its corresponding anatomical name (e.g., "LCCA" for Left Common Carotid Artery).

---

## Table of Contents

- [Overview](#overview)
- [Vessel Classes](#vessel-classes)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Minimal Example](#minimal-example)
  - [Ensemble Prediction](#ensemble-prediction)
  - [Visualization](#visualization)
- [Processing Pipeline](#processing-pipeline)
- [Expected Outputs](#expected-outputs)
- [Graph Features](#graph-features)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)
- [References](#references)

---

## Overview

The vessel labelling module transforms centerline segments into a graph representation where:
- **Nodes** represent bifurcation points
- **Edges** represent vessel segments

A **GATv2** (Graph Attention Network v2) model performs node classification on a transformed version of the graph, assigning anatomical labels to each vessel segment.

| Feature | Description |
|---------|-------------|
| **14 vessel classes** | Covers major head-and-neck arteries |
| **Graph Neural Network** | GATv2 architecture for relational learning |
| **5-fold ensemble** | Improved robustness through model averaging |
| **24 geometric features** | Rich segment-level feature representation |

---

## Vessel Classes

The model classifies vessel segments into 14 anatomical categories:

| ID | Label | Full Name |
|----|-------|-----------|
| 0 | `other` | Unclassified/other vessels |
| 1 | `AA` | Aortic Arch |
| 2 | `BT` | Brachiocephalic Trunk |
| 3 | `RCCA` | Right Common Carotid Artery |
| 4 | `LCCA` | Left Common Carotid Artery |
| 5 | `RSA` | Right Subclavian Artery |
| 6 | `LSA` | Left Subclavian Artery |
| 7 | `RVA` | Right Vertebral Artery |
| 8 | `LVA` | Left Vertebral Artery |
| 9 | `RICA` | Right Internal Carotid Artery |
| 10 | `LICA` | Left Internal Carotid Artery |
| 11 | `RECA` | Right External Carotid Artery |
| 12 | `LECA` | Left External Carotid Artery |
| 13 | `BA` | Basilar Artery |

---

## Architecture

### Graph Representation

The module uses a dual graph representation:

**Edge Form (Original)**
```
Nodes = Bifurcation points
Edges = Vessel segments (with features)
```

**Node Form (For GNN)**
```
Nodes = Vessel segments (with features)
Edges = Connectivity between adjacent vessels
```

### Model Architecture

```
Input Graph (Node Form)
        │
        ▼
┌───────────────────┐
│  GATv2 Layer 1    │  ─── Attention-based message passing
│  (hidden: 128)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  GATv2 Layer 2    │
│  (hidden: 128)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  GATv2 Layer N    │
│  (out: 14 classes)│
└─────────┬─────────┘
          │
          ▼
   Class Predictions
   (per vessel segment)
```

### Ensemble Inference

For improved accuracy, the module supports 5-fold ensemble:

```
┌──────────────┐  ┌──────────────┐       ┌──────────────┐
│   Fold 0     │  │   Fold 1     │  ...  │   Fold 4     │
│   Model      │  │   Model      │       │   Model      │
└──────┬───────┘  └──────┬───────┘       └──────┬───────┘
       │                 │                      │
       ▼                 ▼                      ▼
   Softmax           Softmax               Softmax
       │                 │                      │
       └────────────┬────┴──────────────────────┘
                    │
                    ▼
              Mean Logits
                    │
                    ▼
              Argmax → Final Prediction
```

---

## Usage

### Minimal Example

```python
from arterial.vessel_labelling.vessel_labeller import VesselLabeller

# Initialize the labeller
labeller = VesselLabeller(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)

# Step 1: Build graph from centerline segments
labeller.build_segments_graph(save=True)

# Step 2: Predict vessel types
labeller.predict_vessel_types(ensemble=True, save=True)

# Access results
predicted_graph = labeller.segments_graph_pred
```

### Ensemble Prediction

Using the 5-fold ensemble for improved accuracy (default):

```python
from arterial.vessel_labelling.vessel_labeller import VesselLabeller

labeller = VesselLabeller(
    case_dir="/path/to/case",
    mode="extracranial_vessels"
)

# Build and predict with ensemble (recommended)
labeller.build_segments_graph(save=True)
labeller.predict_vessel_types(
    ensemble=True,  # Use 5-fold ensemble
    save=True
)

# Access vessel labels from the graph
for src, dst, data in labeller.segments_graph_pred.edges(data=True):
    print(f"Segment {data['cell_id']}: {data['vessel_type_name']}")
```

### Visualization

Generate graph visualization plots:

```python
from arterial.vessel_labelling.vessel_labeller import VesselLabeller

labeller = VesselLabeller(case_dir="/path/to/case")

# Generate visualization of the input graph
labeller.build_segments_graph(save=True)
labeller.make_segments_graph_plot()  # Saves segments_graph.png

# Generate visualization with predictions
labeller.predict_vessel_types(save=True)
labeller.make_segments_graph_pred_plot()  # Saves segments_graph_pred.png
```

### Full Pipeline Integration

Using vessel labelling after centerline extraction:

```python
from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.vessel_labelling.vessel_labeller import VesselLabeller

# Step 1: Extract centerlines
extractor = CenterlineExtractor(case_dir="/path/to/case")
extractor.perform_preprocessing(save=True)
extractor.perform_centerline_extraction(save=True)
extractor.perform_centerline_postprocessing(save=True)  # Creates segments array

# Step 2: Label vessels
labeller = VesselLabeller(case_dir="/path/to/case")
labeller.build_segments_graph(save=True)
labeller.predict_vessel_types(ensemble=True, save=True)
```

---

## Processing Pipeline

```
┌─────────────────────────────┐
│  centerline_segments_array  │
│         (.npy)              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Graph Construction       │
│  • Nodes = bifurcations     │
│  • Edges = vessel segments  │
│  • Merge coincident nodes   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Feature Extraction       │
│  • 24 geometric features    │
│  • Per-segment computation  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Node Transform           │
│  • Edges → Nodes            │
│  • Add connectivity edges   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Feature Normalization    │
│  • Z-score for continuous   │
│  • Min-max for discrete     │
│  • Mean-centering for pos   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    GATv2 Inference          │
│  • 5-fold ensemble          │
│  • Softmax averaging        │
│  • Argmax prediction        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Output Graph             │
│  • vessel_type (ID)         │
│  • vessel_type_name (label) │
└─────────────────────────────┘
```

---

## Expected Outputs

### Directory Structure

```
{case_dir}/{mode}/
├── centerline_segments_array.npy   # Input (from centerline extraction)
├── segments_graph.pickle           # Input graph (NetworkX)
├── segments_graph.png              # Input graph visualization
├── segments_graph_pred.pickle      # Output graph with predictions
└── segments_graph_pred.png         # Predicted graph visualization
```

### Graph Structure

Both input and output graphs are NetworkX Graph objects:

**Node Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `pos` | ndarray (3,) | 3D position (RAS coordinates) |
| `radius` | float | Vessel radius at bifurcation |

**Edge Attributes (Input):**
| Attribute | Type | Description |
|-----------|------|-------------|
| `cell_id` | int | Segment identifier |
| `coordinate_array` | ndarray (N, 3) | Centerline point coordinates |
| `radius_array` | ndarray (N,) | Radius along segment |
| `features` | ndarray (24,) | Geometric feature vector |
| `features_dict` | dict | Named feature dictionary |
| `pos` | ndarray (3,) | Segment center position |

**Edge Attributes (Output, additional):**
| Attribute | Type | Description |
|-----------|------|-------------|
| `vessel_type` | int | Predicted class ID (0-13) |
| `vessel_type_name` | str | Anatomical name (e.g., "LCCA") |

---

## Graph Features

The model uses 24 geometric features per vessel segment:

### Radius Features (6)

| Feature | Description |
|---------|-------------|
| `mean radius` | Average radius along segment |
| `proximal radius` | Radius at proximal end |
| `distal radius` | Radius at distal end |
| `proximal/distal radius ratio` | Taper ratio |
| `minimum radius` | Smallest radius along segment |
| `maximum radius` | Largest radius along segment |

### Length Features (3)

| Feature | Description |
|---------|-------------|
| `distance` | Euclidean distance between endpoints |
| `relative length` | Euclidean distance / centerline length |
| `number of points` | Number of centerline points |

### Directional Features (6)

| Feature | Description |
|---------|-------------|
| `direction r/a/s` | Normalized direction vector (R, A, S components) |
| `departure angle r/a/s` | Direction of first 10mm from bifurcation |

### Positional Features (9)

| Feature | Description |
|---------|-------------|
| `proximal bifurcation position r/a/s` | Start point coordinates |
| `distal bifurcation position r/a/s` | End point coordinates |
| `pos r/a/s` | Segment center coordinates |

---

## Configuration Options

### VesselLabeller Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **Required** | Output directory for results |
| `mode` | str | `"extracranial_vessels"` | Labelling mode (only `"extracranial_vessels"` currently supported) |
| `centerline_segments_array_path` | str | `None` | Path to segments array. If `None`, uses `{case_dir}/{mode}/centerline_segments_array.npy` |

### build_segments_graph() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save` | bool | `True` | Save graph and visualization to disk |

### predict_vessel_types() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ensemble` | bool | `True` | Use 5-fold ensemble (recommended) |
| `save` | bool | `True` | Save predicted graph and visualization |

---

## Module Components

### `vessel_labeller.py`

Main orchestrator class `VesselLabeller` for vessel labelling.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `build_segments_graph()` | Construct graph from centerline segments |
| `predict_vessel_types()` | Run GNN inference |
| `segments_graph_sanity_check()` | Validate graph has minimum edges |
| `make_segments_graph_plot()` | Generate input graph visualization |
| `make_segments_graph_pred_plot()` | Generate predicted graph visualization |

### `inference.py`

GNN inference functions:

| Function | Description |
|----------|-------------|
| `perform_inference()` | Main inference dispatcher |
| `predict_extracranial_vessel_types()` | Single model inference |
| `predict_extracranial_vessel_types_ensemble()` | 5-fold ensemble inference |

### `utils.py`

Dataset and visualization utilities:

| Class/Function | Description |
|----------------|-------------|
| `EVCDatasetInference` | PyTorch Geometric dataset for inference |
| `node_transform()` | Convert edge-form to node-form graph |
| `make_graph_plot()` | Generate matplotlib visualization |

### `preprocessing/preprocessing.py`

Graph construction:

| Function | Description |
|----------|-------------|
| `build_nx_graph_from_segments_array()` | Create NetworkX graph from segments |

### `preprocessing/utils.py`

Feature extraction:

| Function | Description |
|----------|-------------|
| `extract_features_for_labelling()` | Compute 24 geometric features per segment |

---

## Model Files

The weights are downloaded by `download_models.sh` (see the main README, "Model Weights")
and live under `<models directory>/vessel_labelling/`:

```
<models directory>/vessel_labelling/
└── extracranial_vessels/
    ├── dataset.json           # Edge labels and normalisation statistics
    ├── model_weights.pth      # Single model (ensemble=False)
    └── fold_{0..4}/
        └── model_weights.pth  # Five folds for the ensemble (ensemble=True)
```

Only extracranial vessel labelling is available.

---

## Environment Requirements

Weights are resolved through `arterial.model_registry`: `ARTERIAL_MODELS_DIR` if set, otherwise
`$arterial_dir/models`, where `arterial_dir` points at the *package* directory:

```bash
export arterial_dir="/path/to/arterial/arterial"     # the inner package directory
# or, to keep the weights elsewhere:
export ARTERIAL_MODELS_DIR="/data/arterial-models"
```

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | CUDA-compatible GPU recommended |
| VRAM | ~2GB minimum |
| CPU | Fallback available |

---

## Limitations

- **Intracranial mode** is not yet implemented
- Requires minimum 2 edges in the graph for valid inference
- Performance depends on quality of centerline extraction

---

## References

1. Brody, S., Alon, U., & Yahav, E. (2022). **How Attentive are Graph Attention Networks?** *ICLR 2022*.
2. PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)

---

## See Also

- [Centerline Extraction Module](centerline_extraction.md) - Provides input segments array
- [Feature Extraction Module](feature_extraction.md) - Uses labelled vessels for analysis
- [Segmentation Module](segmentation.md) - Upstream segmentation


