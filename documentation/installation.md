# Installation guide

## System requirements

## Operating System
Arterial has been developed and tested on Linux (Ubuntu 22.04), and MacOS (14.X, 15.X).

## Installation

For Ubuntu:

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n arterial_env python=3.11
conda activate arterial_env
conda install -c conda-forge vmtk

pip install torch==2.6.0 torch_geometric==2.6.1 monai torchio nnunetv2
pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

For MacOS (outdated, not tested):

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n arterial_env python=3.9
conda activate arterial_env
conda install -c conda-forge vmtk

pip install --upgrade pip
pip install torch==2.2.2 torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

To clone repo and install arterial as a Python package:

```bash
git clone --branch no_slicer https://github.com/FLOWCAT-CV/arterial.git
cd arterial
pip install -e .
```

## Setting up paths

```bash
nano ~/.bashrc
```

```bash
export arterial_dir="/path/to/arterial/arterial"
```

## Copy models

Raw `.pth` files stay ignored in the repo. To move them with plain `git`, export them into chunk files plus a manifest, commit those artifacts, and rebuild the original model paths after cloning.

Source machine:

```bash
python scripts/export_pth_chunks.py --clean
```

This writes:

- `model_chunks/manifest.json`
- chunk files under `model_chunks/arterial/.../*.partNNN`

Destination machine:

```bash
python scripts/rebuild_pth_chunks.py --skip-existing
```

This reconstructs the original `.pth` files under `$arterial_dir/...` and verifies their SHA256 checksums.

## Suggested Git branch strategy

Use two branch roles:

- `main` for normal code, docs, and the chunking scripts
- `models/<version>` for chunk payload commits

Recommended workflow:

```bash
git checkout main
git pull

# after the chunking scripts are already in main
git checkout -b models/2026-03
python scripts/export_pth_chunks.py --clean
```

Commit payloads in module-sized batches:

1. access prediction plus landmark detection
2. vessel labelling
3. extracranial segmentation
4. intracranial segmentation
5. mandible segmentation

On the destination machine:

```bash
git checkout models/2026-03
git pull
python scripts/rebuild_pth_chunks.py --skip-existing
```