import os
import numpy as np
import argparse

from segmentation.nnunetSegmentation import nnUNetInference

from centerlineExtraction.branchAndClippedModelUnification import branchAndClippedModelUnification
from centerlineExtraction.segmentSplitting import segmentSplitting
from centerlineExtraction.segmentsArray import centerlineSegmentsArray

from graphProcessing.centerlineGraph import centerlineGraphOperator
from graphProcessing.graphToVisualizationModels.graphBranchModelLink import graphBranchModelLink

import time

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument('-casePath', '--casePath', type=str, required=True,
    help='path to dir containing the nifti (assumes that the binary map has the basename of the dir). Required.')
parser.add_argument('-no_display', '--no_display', type=bool, required=False, default=False, 
    help='If using a remote Linux, this should be used following correct Slicer installation, and should be coulpled '
    'with the use of `xvfb-run --auto-servernum --server-num=1` upon use before calling this script (prior to the python command). '
    'Not required, default=False.')

args = parser.parse_args()

casePath = args.casePath
no_display = args.no_display

print("                                                          ")
print(f"Processing case {os.path.basename(casePath)} ({casePath})")
print("                                                          ")

caseDir = os.path.abspath(os.path.dirname(casePath))
slicerPath = os.environ["slicerPath"]
segmentationAndCenterlineCode = os.path.join(os.environ["arterialDir"], "centerlineExtraction/performSegmentationAndCenterlineExtraction.py")

times = []

start0 = time.time()

print("Performing segmentation from CTA volume...")

# Performs inference with trained nnU-Net. Changes name of original CTA NIfTI (adds _CTA to caseId) and generates predicted binary map at casePath
# nnUNetInference(casePath)

start1 = time.time()
# Segmentation timestamp
times.append(start1 - start0)

print("Segmentation completed              ")
print(f"Time (parcial): {start1 - start0} s")
print(f"Time (total): {start1 - start0} s  ")
print("                                    ")

print("Starting segmentation and centerline extraction...")

# Perform segmentation and centerline extraction. This generates decimatedSegmentations and centerlines in caseDir
# if no_display: # Use if remote server is used, in combination with xvfb
#     os.system(f"{slicerPath} --disable-terminal-outputs --python-script {segmentationAndCenterlineCode} -casePath {casePath} --exit-after-startup")
# else:
#     os.system(f"{slicerPath} --no-main-window --no-splash  --python-script {segmentationAndCenterlineCode} -casePath {casePath} --exit-after-startup")

start2 = time.time()
# Centerline extraction timestamp
times.append(start2 - start1)

print("Segmentation and centerline extraction completed")
print(f"Time (parcial): {start2 - start1} s            ")
print(f"Time (total): {start2 - start0} s              ")
print("                                                ")

centerlineList = [centerlineFile for centerlineFile in os.listdir(os.path.join(caseDir, "centerlines")) if centerlineFile.endswith(".vtk")]

for idx, _ in enumerate(centerlineList):

    if not os.path.isdir(os.path.join(caseDir, "branchModels")): os.mkdir(os.path.join(caseDir, "branchModels"))
    if not os.path.isdir(os.path.join(caseDir, "clippedModels")): os.mkdir(os.path.join(caseDir, "clippedModels"))

    start20 = time.time()

    print(f"Starting branch extraction from centerline model {idx}...")
    centerlineModel = os.path.join(caseDir,"centerlines", f"centerlines{idx}.vtk")
    branchModel = os.path.join(caseDir, "branchModels", f"branchModel{idx}.vtk")
    surfaceModel = os.path.join(caseDir, "decimatedSegmentations", f"decimatedSegmentation{idx}.vtk")
    clippedModel = os.path.join(caseDir, "clippedModels", f"clippedModel{idx}.vtk")
    radiusArrayName = "Radius"

    # os.system(f"vmtkbranchextractor -ifile {centerlineModel} -ofile {branchModel} -radiusarray {radiusArrayName}")

    start21 = time.time()

    print("Branch extraction completed           ")
    print(f"Time (parcial): {start21 - start20} s")
    print(f"Time (total): {start21 - start0} s   ")
    print("                                      ")

    print("Clipping branch of decimated segmentation surface model...")

    # os.system(f"vmtkbranchclipper -ifile {surfaceModel} -centerlinesfile {branchModel} -ofile {clippedModel} -radiusarray {radiusArrayName}")

    start22 = time.time()

    print("Branch clipping completed             ")
    print(f"Time (parcial): {start22 - start21} s")
    print(f"Time (total): {start22 - start0} s   ")
    print("                                      ")

# Branch and centerline model extraction timestamp
times.append(start22 - start2)

print("Unifying branch and clipped models...")

# branchAndClippedModelUnification(caseDir)

start3 = time.time()
# Branch and clipped model unificaiton timestamp
times.append(start3 - start22)

print("Branch and clipped models unification completed")
print(f"Time (parcial): {start3 - start22} s          ")
print(f"Time (total): {start3 - start0} s             ")
print("                                               ")

print("Splitting clipped model into individual segments...")

# segmentSplitting(caseDir)

start4 = time.time()
# Segment spliting timestamp
times.append(start4 - start3)

print("Segment splitting completed         ")
print(f"Time (parcial): {start4 - start3} s")
print(f"Time (total): {start4 - start0} s  ")
print("                                    ")

print("Generating graph...")

# Generate segmentsArray
# segmentsArray = centerlineSegmentsArray(caseDir, make_plot=False)   

start5 = time.time()
# Graph generation timestamp
times.append(start5 - start4)

print("Graph generation completed          ")
print(f"Time (parcial): {start5 - start4} s")
print(f"Time (total): {start5 - start0} s  ")
print("                                    ")

print("Performing supersegment extraction...")

# Supersegment extraction
centerlineGraph = centerlineGraphOperator(caseDir)
print("     Making centerline dense graph...")
centerlineGraph.makeCenterlineGraph()
print("     done")
print("     Performing sanity check for random islands...")
centerlineGraph.sanityCheckForRandomIslands()
print("     done")
print("     Making simple centerline dense graph...")
centerlineGraph.makeSimpleCenterlineGraph()
print("     done")
print("     Predicting vessel types...")
centerlineGraph.predictVesselTypes()
print("     done")
print("     Unifying subgraphs...")
centerlineGraph.unifySubgraphs()
print("     done")
print("     Performing feature extraction...")
centerlineGraph.performFeatureExtraction()
print("     done")
print("     Extracting supersegments...")
centerlineGraph.supersegmentExtraction()
print("     done")
print()

start6 = time.time()
# Supersegment extraction timestamp
times.append(start6 - start5)

print("Supersegment extraction completed ")
print(f"Time (parcial): {start6 - start5} s")
print(f"Time (total): {start6 - start0} s  ")
print("                                    ")

print("Linking branch model to vesselType prediction...")

# Linked labeled graph with branchModel and clipped model segments (assumes graph_pred.pickle exists in caseDir)
# graphBranchModelLink(caseDir)

start7 = time.time()
# Graph and branch model link timestamp
times.append(start7 - start6)

# Total time
times.append(start7 - start0)
# Save time stamps in text file
# np.savetxt(os.path.join(caseDir, "timeStamps.txt"), times)

print("Branch model link to vesselType prediction completed")
print(f"Time (parcial): {start7 - start6} s                ")
print("                                                    ")

print("Analysis completed                ")
print(f"Time (total): {start7 - start0} s")
print("                                  ")