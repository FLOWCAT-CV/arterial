# Installation guide

## System requirements

## Operating System
Arterial has been developed and tested on Linux (Ubuntu 22.04), and MacOS (14.X, 15.X).

## Installation

For Ubuntu:

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n foo python=3.9
conda activate foo
conda install -c conda-forge vmtk

pip install torch torch_geometric==2.5.2
pip install pyg_lib torch_scatter==2.1.2+pt22cu121 torch_sparse==0.6.18+pt22cu121 torch_cluster==1.6.3+pt22cu121 torch_spline_conv==1.2.2+pt22cu121 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

For MacOS:

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n foo python=3.9
conda activate foo
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

This step is currently performed manually.