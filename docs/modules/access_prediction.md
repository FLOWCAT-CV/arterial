# Access Prediction Module

The access prediction module predicts catheter accessibility for endovascular intervention using a multi-scale Graph Neural Network (ArterialGNet). It processes catheter pathway graphs (supersegments) to predict procedural feasibility and generates interpretable attention maps highlighting anatomically challenging regions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [ArterialGNet Model](#arterialgnet-model)
  - [Multi-Scale Input](#multi-scale-input)
- [Usage](#usage)
  - [Minimal Example](#minimal-example)
  - [Full Pipeline with Attention Maps](#full-pipeline-with-attention-maps)
  - [Custom Access/Side Configuration](#custom-accessside-configuration)
- [Processing Pipeline](#processing-pipeline)
- [Expected Outputs](#expected-outputs)
- [Prediction Interpretation](#prediction-interpretation)
- [Configuration Options](#configuration-options)
- [Module Components](#module-components)
- [References](#references)

---

## Overview

The access prediction module answers the clinical question: **"How feasible is catheter navigation along this vascular pathway?"**

| Feature | Description |
|---------|-------------|
| **Input** | Supersegment graphs from feature extraction |
| **Output** | Accessibility probability (0-1) with confidence interval |
| **Attention Maps** | Per-node attention weights identifying difficult segments |
| **Model** | ArterialGNet (multi-scale GATv2 architecture) |
| **Ensemble** | 5-fold cross-validation for robust predictions |

The module supports multiple configurations:
- **Access sites**: Femoral (default), Radial
- **Target sides**: Left, Right (both by default)

---

## Architecture

### ArterialGNet Model

ArterialGNet is a multi-scale graph neural network that processes vascular information at three levels:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ArterialGNet                                  │
│                                                                         │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│   │  Global Features │   │  Segment Graph   │   │   Dense Graph    │    │
│   │      (MLP)       │   │     (GATv2)      │   │     (GATv2)      │    │
│   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘    │
│            │                      │                      │              │
│            │                      ▼                      ▼              │
│            │              ┌──────────────┐      ┌──────────────┐        │
│            │              │  Pooling     │      │  Pooling     │        │
│            │              │  (mean/max)  │      │  (mean/max)  │        │
│            │              └──────┬───────┘      └──────┬───────┘        │
│            │                     │                     │                │
│            └─────────────────────┴─────────────────────┘                │
│                                  │                                      │
│                                  ▼                                      │
│                        ┌─────────────────┐                              │
│                        │  Concatenation  │                              │
│                        └────────┬────────┘                              │
│                                 │                                       │
│                                 ▼                                       │
│                        ┌─────────────────┐                              │
│                        │   Output MLP    │                              │
│                        │   (Softmax)     │                              │
│                        └────────┬────────┘                              │
│                                 │                                       │
│                                 ▼                                       │
│                        ┌─────────────────┐                              │
│                        │   Prediction    │                              │
│                        │   (0 = easy,    │                              │
│                        │    1 = difficult)│                             │
│                        └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Multi-Scale Input

| Scale | Graph Type | Nodes | Features |
|-------|------------|-------|----------|
| **Global** | Tensor | 1 | Supersegment-level: side, tortuosity, length, diameters, angles, vessel types |
| **Segment** | Graph | Per vessel | Vessel-level: tortuosity, length, diameter stats, polar/azimuthal angles |
| **Dense** | Graph | Per point | Point-level: position, radius, curvature, direction, hierarchy |

---

## Usage

### Minimal Example

```python
from arterial.access_prediction.access_predictor import AccessPredictor

# Initialize predictor
predictor = AccessPredictor(
    case_dir="/path/to/case"
)

# Predict accessibility (preprocessing + inference)
predictor.predict_accessibility()

# Access results
for (access, side), prediction in predictor.predictions_dict.items():
    print(f"{access} {side}: {prediction['mean']:.2f} ± {prediction['std']:.2f}")
```

### Full Pipeline with Attention Maps

```python
from arterial.access_prediction.access_predictor import AccessPredictor

# Initialize with custom configuration
predictor = AccessPredictor(
    case_dir="/path/to/case",
    cta_nifti_path="/path/to/cta.nii.gz",
    supersegments_dir_path="/path/to/supersegments",
    access=["femoral"],
    side=["left", "right"]
)

# Step 1: Preprocess supersegments
predictor.preprocess_supersegments(save=True)

# Step 2: Predict with attention maps
predictor.predict_accessibility(
    return_attention_map=True,
    save=True
)

# Access predictions
prediction = predictor.predictions_dict[("femoral", "left")]
print(f"Prediction: {prediction['mean']:.3f}")
print(f"95% CI: [{prediction['mean'] - 1.96*prediction['std']/2.236:.3f}, "
      f"{prediction['mean'] + 1.96*prediction['std']/2.236:.3f}]")

# Access attention map graph
attention_graph = predictor.attention_maps_dict[("femoral", "left")]
for node in attention_graph.nodes():
    weight = attention_graph.nodes[node].get("attention_weight", 0)
    if weight > 0.1:  # High attention nodes
        print(f"Node {node}: attention = {weight:.3f}")
```

### Custom Access/Side Configuration

```python
from arterial.access_prediction.access_predictor import AccessPredictor

# Predict only for femoral left
predictor = AccessPredictor(
    case_dir="/path/to/case",
    access=["femoral"],
    side=["left"]
)
predictor.predict_accessibility()

# Predict for both femoral and radial (when radial models available)
predictor = AccessPredictor(
    case_dir="/path/to/case",
    access=["femoral", "radial"],
    side=["left", "right"]
)
predictor.predict_accessibility()
```

---

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ACCESS PREDICTION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT: Supersegment graphs from Feature Extraction                      │
│         (e.g., femoral + left + anterior.pickle)                         │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING (preprocess_supersegments)                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  • Clean supersegment graph                                        │  │
│  │  • Extract global features (tortuosity, length, diameters, etc.)   │  │
│  │  • Build segment graph (one node per vessel type)                  │  │
│  │  • Build dense graph (one node per centerline point)               │  │
│  │  • Compute node/edge features for both graphs                      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Output: preprocessed_supersegment_dict.pickle                           │
│          global_features.json                                            │
│          segment_supersegment.pickle                                     │
│          dense_supersegment.pickle                                       │
│          combined_plot.png                                               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 2: INFERENCE (predict_accessibility)                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  • Load 5-fold ArterialGNet models                                 │  │
│  │  • Convert graphs to PyTorch Geometric format                      │  │
│  │  • Normalize features using training statistics                    │  │
│  │  • Run inference on each fold                                      │  │
│  │  • Average predictions and attention weights                       │  │
│  │  • Build attention map graph                                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Output: access_prediction.json                                          │
│          attention_map.pickle                                            │
│          attention_map.png                                               │
│          attention_map.vtk                                               │
│          attention_map_points.json                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Expected Outputs

### Directory Structure

```
case_dir/
└── extracranial_vessels/
    └── access_prediction/
        ├── femoral_left/
        │   ├── preprocessed_supersegment_dict.pickle
        │   ├── global_features.json
        │   ├── segment_supersegment.pickle
        │   ├── dense_supersegment.pickle
        │   ├── combined_plot.png
        │   ├── access_prediction.json
        │   ├── attention_map.pickle
        │   ├── attention_map.png
        │   ├── attention_map.vtk
        │   └── attention_map_points.json
        └── femoral_right/
            └── ... (same structure)
```

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `preprocessed_supersegment_dict.pickle` | Pickle | Complete preprocessed data dictionary |
| `global_features.json` | JSON | Global supersegment features |
| `segment_supersegment.pickle` | NetworkX Graph | Segment-level graph |
| `dense_supersegment.pickle` | NetworkX Graph | Dense point-level graph |
| `combined_plot.png` | PNG | Visualization of both graphs |
| `access_prediction.json` | JSON | Prediction mean and std |
| `attention_map.pickle` | NetworkX Graph | Graph with attention weights |
| `attention_map.png` | PNG | Attention map visualization |
| `attention_map.vtk` | VTK PolyData | 3D attention map for visualization |
| `attention_map_points.json` | JSON | Point coordinates with weights |

### Accessing Results

```python
import json
import pickle

# Load prediction
with open("access_prediction.json", "r") as f:
    prediction = json.load(f)
print(f"Mean: {prediction['mean']:.3f}")
print(f"Std: {prediction['std']:.3f}")

# Load attention map
with open("attention_map.pickle", "rb") as f:
    attention_graph = pickle.load(f)

# Find highest attention nodes
attention_weights = [
    (node, attention_graph.nodes[node].get("attention_weight", 0))
    for node in attention_graph.nodes()
]
attention_weights.sort(key=lambda x: x[1], reverse=True)
print("Top 5 attention nodes:", attention_weights[:5])
```

---

## Prediction Interpretation

### Prediction Values

| Value Range | Interpretation | Clinical Meaning |
|-------------|----------------|------------------|
| 0.0 - 0.3 | Low difficulty | Straightforward catheterization expected |
| 0.3 - 0.7 | Moderate difficulty | Some anatomical complexity, but should be feasible |
| 0.7 - 1.0 | High difficulty | Significant anatomical complexity, consider alternative access |

### Confidence Intervals

The module reports predictions with 95% confidence intervals:

```
prediction ± 1.96 × (std / √5)
```

Where:
- `prediction`: Mean across 5 folds
- `std`: Standard deviation across folds
- `√5`: Correction for 5 independent predictions

### Attention Maps

Attention weights identify which regions of the vascular pathway the model focuses on:

| Attention Weight | Interpretation |
|-----------------|----------------|
| Low (< 0.05) | Normal vessel, no difficulty expected |
| Medium (0.05 - 0.15) | Moderate complexity |
| High (> 0.15) | Challenging region, potential difficulty |

High-attention regions typically correspond to:
- Sharp bends or high curvature
- Narrow vessel segments
- Complex bifurcations
- Tortuous segments

---

## Configuration Options

### AccessPredictor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_dir` | str | **required** | Path to case directory |
| `cta_nifti_path` | str | `None` | Path to CTA NIfTI (default: `case_dir/cta.nii.gz`) |
| `supersegments_dir_path` | str | `None` | Path to supersegments directory |
| `access` | list | `["femoral"]` | Access sites to predict |
| `side` | list | `["left", "right"]` | Target sides to predict |

### Method Parameters

| Method | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `preprocess_supersegments` | `save` | `True` | Save preprocessed data |
| `predict_accessibility` | `return_attention_map` | `True` | Compute attention maps |
| `predict_accessibility` | `save` | `True` | Save predictions and maps |

### Model Configuration

The ArterialGNet model uses the following architecture (loaded from checkpoints):

| Parameter | Description |
|-----------|-------------|
| `hidden_dim` | Hidden dimension for global/segment paths |
| `hidden_dim_dense` | Hidden dimension for dense path |
| `num_global_layers` | Number of MLP layers for global features |
| `num_segment_layers` | Number of GATv2 layers for segment graph |
| `num_dense_layers` | Number of GATv2 layers for dense graph |
| `attn_heads` | Number of attention heads (default: 8) |
| `aggregation` | Pooling method (mean/max/add) |

---

## Module Components

```
access_prediction/
├── __init__.py
├── access_predictor.py       # Main AccessPredictor class
├── inference.py              # Model inference logic
├── models.py                 # ArterialGNet architecture
├── utils.py                  # Dataset, transforms, utilities
├── preprocessing/
│   ├── preprocessing.py      # Supersegment preprocessing
│   └── utils.py              # Preprocessing utilities
└── (weights are not in the package; see below)
```

Weights are downloaded by `download_models.sh` to `<models directory>/access_prediction/`
(`dataset.json` with the normalisation statistics and `fold_{0..4}/model_weights.pth`), resolved
through `arterial.model_registry`: `ARTERIAL_MODELS_DIR` if set, otherwise `$arterial_dir/models`.

### Key Classes

| Class | Location | Description |
|-------|----------|-------------|
| `AccessPredictor` | `access_predictor.py` | Main interface for predictions |
| `ArterialGNet` | `models.py` | Multi-scale GNN architecture |
| `GATv2Layer` | `models.py` | Graph Attention layer |
| `ArterialGNetDatasetInference` | `utils.py` | PyG dataset for inference |
| `DenseRadiusGraph` | `utils.py` | Radius graph transform |

### Key Functions

| Function | Location | Description |
|----------|----------|-------------|
| `preprocess_supersegment` | `preprocessing/` | Prepare supersegment for inference |
| `perform_inference` | `inference.py` | Run 5-fold ensemble inference |
| `build_final_attention_map` | `utils.py` | Construct attention map graph |

---

## References

- The ArterialGNet architecture combines GATv2 (Graph Attention Networks v2) with multi-scale processing
- Attention weights are extracted from the first GATv2 layer of the dense graph path
- Feature normalization uses training set statistics stored in `dataset.json`

---

## See Also

- [Feature Extraction Module](feature_extraction.md) - Generates supersegments input
- [Vessel Labelling Module](vessel_labelling.md) - Provides vessel type labels
- [Run Module](run.md) - Pipeline orchestration
- [Arterial GNet repo](https://github.com/perecanals/arterial_gnet.git) - ArterialGNet model repository

