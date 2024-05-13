#!/bin/bash

# Get the directory path of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export environmental variables based on the directory path
export arterial_dir="$DIR/arterial"

# Install necessary packages before arterial installation
conda install -c conda-forge vmtk
pip install --upgrade pip
pip install torch torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html