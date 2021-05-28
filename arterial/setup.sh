#!/bin/zsh

# Setting up environment variables
echo "export arterialDir=`pwd`" >> ~/.zshrc
echo "export nnUNet_raw_data_base=`pwd`/segmentation/nnUNet_base/nnUNet_raw_data" >> ~/.zshrc
echo "export nnUNet_preprocessed=`pwd`/segmentation/nnUNet_base/nnUNet_preprocessed" >> ~/.zshrc
echo "export RESULTS_FOLDER=`pwd`/segmentation/models" >> ~/.zshrc

source ~/.zshrc