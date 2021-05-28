import os
import vtk
import argparse

from vesselLabelling.segmentsArray import centerlineSegmentsArray
from vesselLabelling.centerlineGraph import generateCenterlineGraph
from vesselLabelling.segmentSplitting import segmentSplitting
from vesselLabelling.graphBranchModelLink import graphBranchModelLink

from centerlineExtraction.branchAndClippedModelUnification import branchAndClippedModelUnification

import time

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument('-casePath', '--casePath', type=str, required=True, 
    help='path to dir containing the nifti (assumes that the binary map has the basename of the dir). Required.')

args = parser.parse_args()

casePath = args.casePath

print("                                                          ")
print(f"Processing case {os.path.basename(casePath)} ({casePath})")
print("                                                          ")

caseDir = os.path.abspath(os.path.dirname(casePath))

# import shutil

# # for element in os.listdir(caseDir):
# #     if not element.endswith(".nii.gz"):
# #         if os.path.isdir(os.path.join(caseDir, element)):
# #             shutil.rmtree(os.path.join(caseDir, element))
# #         if os.path.isfile(os.path.join(caseDir, element)):
# #             os.remove(os.path.join(caseDir, element))

# segmentationAndCenterlineCode = os.path.join(os.path.abspath(""), "centerlineExtraction/performSegmentationAndCenterlineExtraction.py")
segmentationAndCenterlineCode = os.path.join("/Users/pere/GitHub/arterial/v2/arterial", "centerlineExtraction/performSegmentationAndCenterlineExtraction.py")

start0 = time.time()

print("Starting segmentation and centerline extraction...")

# Perform segmentation and centerline extraction. This generates segmentation.vtk, decimatedSegmentation.vtk and centerlines.vtk in caseDir
os.system(f"/Applications/Slicer.app/Contents/MacOS/Slicer --disable-terminal-outputs --no-main-window --python-script {segmentationAndCenterlineCode} -pth {casePath} --exit-after-startup")

start1 = time.time()

print("Segmentation and centerline extraction completed")
print(f"Time (parcial): {start1 - start0} s            ")
print(f"Time (total): {start1 - start0} s              ")
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

    os.system(f"vmtkbranchextractor -ifile {centerlineModel} -ofile {branchModel} -radiusarray {radiusArrayName}")

    start21 = time.time()

    print("Branch extraction completed           ")
    print(f"Time (parcial): {start21 - start20} s")
    print(f"Time (total): {start21 - start0} s   ")
    print("                                      ")

    print("Clipping branch of decimated segmentation surface model...")

    os.system(f"vmtkbranchclipper -ifile {surfaceModel} -centerlinesfile {branchModel} -ofile {clippedModel} -radiusarray {radiusArrayName}")

    start22 = time.time()

    print("Branch clipping completed             ")
    print(f"Time (parcial): {start22 - start21} s")
    print(f"Time (total): {start22 - start0} s   ")
    print("                                      ")

print("Unifying branch and clipped models...")

branchAndClippedModelUnification(caseDir)

start3 = time.time()

print("Branch and clipped models unification completed")
print(f"Time (parcial): {start3 - start22} s          ")
print(f"Time (total): {start3 - start0} s             ")
print("                                               ")

print("Splitting clipped model into individual segments...")

segmentSplitting(caseDir)

start4 = time.time()

print("Segment splitting completed         ")
print(f"Time (parcial): {start4 - start3} s")
print(f"Time (total): {start4 - start0} s  ")
print("                                    ")

print("Generating graph...")

# Generate segmentsArray
segmentsArray = centerlineSegmentsArray(caseDir, make_plot=False)
        
# Generate centerline graph
G = generateCenterlineGraph(segmentsArray, caseDir, make_plot=True)

start5 = time.time()

print("Graph generation completed          ")
print(f"Time (parcial): {start5 - start4} s")
print(f"Time (total): {start5 - start0} s  ")
print("                                    ")

####### Now the GNN would continue

# Linked labeled graph with branchModel and clipped model segments (assumes graph_label.pickle exists in caseDir)
# Output could be the relation between graph labels and groupIds in branchModel and clippedModel
# graphBranchModelLink(caseDir)

# start6 = time.time()

# print("Branch model linkage to labeled graph completed")
# print(f"Time (parcial): {start6 - start5} s           ")
# print(f"Time (total): {start6 - start0} s             ")
# print("                                               ")