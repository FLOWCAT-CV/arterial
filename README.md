# Arterial

**An AI framework for automated vascular analysis and endovascular intervention planning**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Arterial is a comprehensive AI framework for fully automated vascular tortuosity analysis from CT Angiography (CTA) images. Originally developed to support mechanical thrombectomy (MT) planning for acute ischemic stroke (AIS), Arterial provides:

- **Automated vessel segmentation** using deep learning (nnU-Net v2)
- **Centerline extraction** via VMTK
- **Anatomical landmark detection** using 3D U-Net
- **Vessel labelling** with Graph Neural Networks (GNN)
- **Multi-scale feature extraction** for vascular characterization
- **Catheter pathway mapping** and **accessibility prediction**

The framework processes a single CTA image and outputs a complete vascular analysis including segmentation masks, labeled centerlines, geometric features, and procedural predictions—all without manual input.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Deep Learning Segmentation** | nnU-Net v2 models for extracranial and intracranial vessels |
| **Automatic Centerlines** | VMTK-based centerline extraction with branch splitting |
| **Landmark Detection** | Automatic identification of ICA and MCA bifurcations |
| **Vessel Labelling** | GNN-based classification of 14 vessel types |
| **Feature Extraction** | Local, segment, and global vascular features |
| **Pathway Mapping** | 8 catheter configurations (femoral/radial × left/right × anterior/posterior) |
| **End-to-End Pipeline** | Single command from CTA to complete analysis |

---

## Pipeline Architecture

```
                              ┌─────────────┐
                              │  CTA Image  │
                              │  (.nii.gz)  │
                              └──────┬──────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            ▼                            │
        │               ┌────────────────────────┐                │
        │               │     SEGMENTATION       │                │
        │               │   (nnU-Net v2 models)  │                │
        │               └───────────┬────────────┘                │
        │                           │                             │
        │                           ▼                             │
        │          ┌─────────────────────────────────┐            │
        │          │    CENTERLINE EXTRACTION        │            │
        │          │   (VMTK preprocessing + vmtk)   │            │
        │          └────────────────┬────────────────┘            │
        │                           │                             │
        │          ┌────────────────┴────────────────┐            │
        │          ▼                                 ▼            │
        │  ┌───────────────────┐          ┌──────────────────┐    │
        │  │ LANDMARK DETECTION│          │  VESSEL LABELLING│    │
        │  │   (3D U-Net)      │          │      (GNN)       │    │
        │  └─────────┬─────────┘          └────────┬─────────┘    │
        │            │                             │              │
        │            └──────────────┬──────────────┘              │
        │                           ▼                             │
        │               ┌────────────────────────┐                │
        │               │   FEATURE EXTRACTION   │                │
        │               │  (local/segment/global)│                │
        │               └───────────┬────────────┘                │
        │                           │                             │
        │                           ▼                             │
        │               ┌────────────────────────┐                │
        │               │   ACCESS PREDICTION    │                │
        │               │   (catheter pathways)  │                │
        │               └────────────────────────┘                │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
```

---

## Modules

Arterial is organized into specialized modules, each with detailed documentation:

| Module | Description | Documentation |
|--------|-------------|---------------|
| **Segmentation** | nnU-Net v2 vessel segmentation (extracranial/intracranial) | [README](arterial/segmentation/README.md) |
| **Centerline Extraction** | VMTK-based centerline computation and branching | [README](arterial/centerline_extraction/README.md) |
| **Landmark Detection** | Automatic anatomical landmark identification | [README](arterial/landmark_detection/README.md) |
| **Vessel Labelling** | GNN-based anatomical vessel classification | [README](arterial/vessel_labelling/README.md) |
| **Feature Extraction** | Multi-scale vascular feature computation | [README](arterial/feature_extraction/README.md) |
| **Access Prediction** | Catheter accessibility prediction with attention maps | [README](arterial/access_prediction/README.md) |
| **Run (Processor)** | Pipeline orchestration and CLI | [README](arterial/run/README.md) |

---

## Installation

### System Requirements

- **Operating System**: Linux (Ubuntu 22.04 tested), macOS (14.x, 15.x)
- **Python**: 3.9+ (3.11 recommended for Linux)
- **GPU**: NVIDIA GPU with CUDA support (recommended for inference)

### Linux (Ubuntu)

```bash
# Create conda environment
conda create -n arterial_env python=3.11
conda activate arterial_env

# Install VMTK
conda install -c conda-forge vmtk

# Install PyTorch and dependencies
pip install torch==2.6.0 torch_geometric==2.6.1 monai torchio nnunetv2

# Install PyG dependencies (CUDA 12.4)
pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### macOS

```bash
# Create conda environment (Python 3.9 required for VMTK on macOS)
conda create -n arterial_env python=3.9
conda activate arterial_env

# Install VMTK
conda install -c conda-forge vmtk

# Install PyTorch and dependencies
pip install --upgrade pip
pip install torch==2.2.2 torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

### Install Arterial

```bash
# Clone the repository
git clone https://github.com/FLOWCAT-CV/arterial.git
cd arterial

# Install as editable package
pip install -e .
```

### Environment Configuration

Add the Arterial directory to your environment:

```bash
# Add to ~/.bashrc (Linux) or ~/.zshrc (macOS)
export arterial_dir="/path/to/arterial/arterial"
```

---

## Quick Start

### Command-Line Usage

```bash
# Full analysis pipeline (extracranial vessels)
python perform_analysis.py -cd /path/to/case_dir

# With explicit CTA path
python perform_analysis.py -cd /path/to/case_dir -cnp /path/to/cta.nii.gz

# Intracranial vessel analysis
python perform_analysis.py -cd /path/to/case_dir -m intracranial_vessels

# Fast segmentation mode
python perform_analysis.py -cd /path/to/case_dir -fast

# Skip segmentation (if already computed)
python perform_analysis.py -cd /path/to/case_dir -ss
```

### Python API

```python
from arterial.segmentation import VesselSegmenter
from arterial.centerline_extraction import CenterlineExtractor
from arterial.vessel_labelling import VesselLabeller
from arterial.feature_extraction import FeatureExtractor

case_dir = "/path/to/case"

# Step 1: Segment vessels
segmenter = VesselSegmenter(case_dir, mode="extracranial_vessels")
segmenter.segment_vessels_from_cta()

# Step 2: Extract centerlines
extractor = CenterlineExtractor(case_dir, mode="extracranial_vessels")
extractor.perform_centerline_extraction()
extractor.perform_branch_model_extraction()
extractor.perform_centerline_postprocessing()

# Step 3: Label vessels
labeller = VesselLabeller(case_dir, mode="extracranial_vessels")
labeller.build_segments_graph()
labeller.predict_vessel_types()

# Step 4: Extract features
feature_ext = FeatureExtractor(case_dir, mode="extracranial_vessels")
feature_ext.build_local_graph()
feature_ext.extract_local_features()
feature_ext.extract_segment_features()
feature_ext.extract_global_features()
feature_ext.extract_supersegments()
```

### Using the Processor

```python
from argparse import Namespace
from arterial.run.processor import ArterialProcessor

args = Namespace(
    case_dir="/path/to/case",
    cta_nifti_path=None,
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

processor = ArterialProcessor(args)
timing = processor.perform_analysis()

print(f"Total analysis time: {timing['total_time']:.2f}s")
```

---

## Expected Input

| Requirement | Format | Description |
|-------------|--------|-------------|
| CTA Image | NIfTI (`.nii.gz`) | CT Angiography volume |
| Case Directory | Folder | Working directory for outputs |

**Default naming convention**: `case_dir/cta.nii.gz`

---

## Expected Outputs

After running the full pipeline:

```
case_dir/
├── cta.nii.gz                              # Input
├── extracranial_vessels_segmentation.nii.gz # Vessel mask
└── extracranial_vessels/
    ├── centerlines/                         # Individual centerline models
    ├── branch_model.vtk                     # Merged branch model
    ├── centerline_segments_array.npy        # Centerline data
    ├── landmarks/
    │   └── landmarks.json                   # Detected landmarks
    ├── segments_graph_pred.pickle           # Labeled vessel graph
    ├── local_graph.pickle                   # Featurized graph
    ├── supersegments/                       # Catheter pathways (8 configs)
    └── access_prediction/                   # Accessibility predictions
```

---

## Analysis Modes

| Mode | Vessels | Landmarks | Features |
|------|---------|-----------|----------|
| `extracranial_vessels` | Aortic arch through Circle of Willis | 6 (bilateral ICA, EICA, MCA) | Full pipeline |
| `intracranial_vessels` | Circle of Willis and branches | 4 (bilateral TICA, MCA) | Individual centerlines |

---

## Command-Line Options

| Flag | Description |
|------|-------------|
| `-cd`, `--case_dir` | Path to case directory (required) |
| `-cnp`, `--cta_nifti_path` | Path to CTA NIfTI file |
| `-m`, `--mode` | Analysis mode (`extracranial_vessels` or `intracranial_vessels`) |
| `-sd`, `--sampling_distance_mm` | Centerline sampling distance (default: 2) |
| `-fast`, `--fast_segmentation` | Use fast (single-resolution) segmentation |
| `-ss`, `--skip_segmentation` | Skip segmentation step |
| `-sce`, `--skip_centerline_extraction` | Skip centerline extraction |
| `-sb`, `--skip_branching` | Skip branch model extraction |
| `-svl`, `--skip_vessel_labelling` | Skip vessel labelling |
| `-sfe`, `--skip_feature_extraction` | Skip feature extraction |
| `-sap`, `--skip_access_prediction` | Skip access prediction |
| `-sld`, `--skip_landmark_detection` | Skip landmark detection |

---

## Citation

If you use Arterial in your research, please cite:

```bibtex
@article{canals2024arterial,
  title={Arterial: An AI framework for automated vascular analysis},
  author={Canals, Pere and others},
  journal={...},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Arterial was developed at the Stroke Research group at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

**Key dependencies:**
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - Deep learning segmentation
- [VMTK](http://www.vmtk.org/) - Vascular Modeling Toolkit
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph Neural Networks
- [MONAI](https://monai.io/) - Medical image analysis

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions and support, please open an issue on GitHub.
