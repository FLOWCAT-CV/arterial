# Installation guide

## System requirements

## Operating System
Arterial has been developed and tested on Linux (Ubuntu 22.04), and MacOS.

## Installation

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n foo python=3.9
conda activate foo
conda install -c conda-forge vmtk

pip install --upgrade pip
pip install torch torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

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