#---------------------------------------------------------------#
#---------------------------------------------------------------#
# To call this script using the slicer python 
# ./Applications/Slicer.app/Contents/MacOS/Slicer --no-main-window --python-script /Users/pere/opt/anaconda3/envs/vmtkenv/slicer/performSegmentationAndCenterlineExtraction.py
#---------------------------------------------------------------#
#---------------------------------------------------------------#

import os
import slicer
import numpy as np
import argparse

from performSegmentationsFromBinaryMask import performSegmentationsFromBinaryMask
from centerlineExtraction import centerlineExtraction

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument('-casePath', '--casePath', type=str, required=True, 
    help='path binary nifti to be processed. Required.')

args = parser.parse_args()

casePath = args.casePath

caseDir = os.path.abspath(os.path.dirname(casePath))

# Load volume and associate to node
slicer.util.loadLabelVolume(casePath)
masterVolumeNode = getNode(os.path.basename(casePath[:-7]))

# Perform and save segmentations as vtk files (segmentation.vtk and decimatedSegmentation.vtk)
segmentationNode = performSegmentationsFromBinaryMask(caseDir, masterVolumeNode)
# Perform centerline extraction. Creates centerlines.vtk
centerlineExtraction(caseDir, segmentationNode)