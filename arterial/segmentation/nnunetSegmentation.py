#    Copyright 2021 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import shutil
import numpy as np
import argparse

from time import time

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument("-mode", "--mode", type=str, required=True, 
    help="choose between `inference` or `ensemble`. Required.")

parser.add_argument("-input", "--inputImage", type=str, required=True,
    help="absolute path to the input image (has to be a nifti). Required")


args = parser.parse_args()

mode       = args.mode
inputImage = args.inputImage
    
##############################################################################################
#--------------------------------------------------------------------------------------------#    

print("                                                                                     ")
print("Model optimization for nnUNet (Isensee et al. 2020) for segmentation of 3D CTA images")
print("By Pere Canals (2020)                                                                ")
print("                                                                                     ")

#--------------------------------------------------------------------------------------------#    
##############################################################################################

if mode == "inference":
    inference(inputImage)
elif mode == "ensemble":
    ensemble(inputImage)
else:
    ValueError("Please introduce one of the possible modes. See main.py -h for more information.")


def inference(inputImage):
    ''' Performs inference of inputImage (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    outputPath.

    Arguments:
        - inputImage <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''
    arterialSegmentationDir = os.path.join(os.environ["arterialDir"], "segmentation")
    modelDir = os.path.join(arterialSegmentationDir, "models/nnUNet/3d_lowres/Task01_Arterial/nnUNetTrainerV2__nnUNetPlansv2.1/all")

    patId = inputImage[:-7] # Name of the nifti inputImage except the .nii.gz extension
    inputDir = os.path.dirname(inputImage)

    # Paths used by nnUNet
    inputPath = os.path.join(inputDir, patId, "input")
    outputPath = os.path.join(inputDir, patId, "output")
    if not os.path.isdir(inputPath): os.mkdir(inputPath)
    if not os.path.isdir(outputPath): os.mkdir(outputPath)
    
    # The inputImage will be placed in a newly created dir with the patId as name
    if not os.path.isfile(os.path.join(inputPath, patId + "_0000.nii.gz")):
        os.rename(inputImage, os.path.join(inputPath, patId + "_0000.nii.gz"))

    # Only use if running on Colab
    if inputPath[:8] == "/content": # If we are working on Colab and Drive, we need to get rid of spaces in the path
        inputPath = inputPath[:17] + "\ " + inputPath[18:]
        outputPath = outputPath[:17] + "\ " + outputPath[18:]

    start = time()

    os.system("nnUNet_predict -i " + inputPath + " -o " + outputPath + f" -t Task01_Arterial -m 3d_lowres -f all")

    print(f"Inference took {time() - start} s")
    print("                                  ")


def ensemble(inputImage):
    ''' Performs inference of inputImage (nifti, CTA) by ensembling all folds of 
    the best performing model to output a binary mask in a nifti format. The resulting 
    nifti will be placed in outputDir.

    Arguments:
        - inputImage <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''

   
    arterialSegmentationDir = os.path.join(os.environ["arterialDir"], "segmentation")
    modelDir = os.path.join(arterialSegmentationDir, "models/nnUNet/3d_lowres/Task01_Arterial/nnUNetTrainerV2__nnUNetPlansv2.1")

    patId = inputImage[:-7]
    inputDir = os.path.dirname(inputImage)

    inputPath = os.path.join(inputDir, patId, "input")
    if not os.path.isdir(inputPath): os.mkdir(inputPath)

    # Only use if running on Colab. Make sure inputPath and outputPath do not contain spaces
    if inputPath[:8] == "/content": # If we are working on Colab and Drive, we need to get rid of spaces in the path
        inputPath = inputPath[:17] + "\ " + inputPath[18:]
    
    if not os.path.isfile(os.path.join(inputPath, patId + "_0000.nii.gz")):
        os.rename(inputImage, os.path.join(inputPath, patId + "_0000.nii.gz"))

    npzDirs = []

    start = time()

    for fold in range(5):
        print(f"Predicting {patId} fold {fold}")
        print("                               ")

        outputPath = os.path.join(inputDir, patId, f"fold_{fold}")
        if not os.path.isdir(outputPath): os.mkdir(outputPath)

        foldDir = os.path.join(modelDir, f"fold_{fold}")

        # Only use if running on Colab. Make sure inputPath and outputPath do not contain spaces
        if outputPath[:8] == "/content": # If we are working on Colab and Drive, we need to get rid of spaces in the path
            outputPathAux = outputPath[:17] + "\ " + outputPath[18:]
        else:
            outputPathAux = outputPath
            
        os.system("nnUNet_predict -i " + inputPath + " -o " + outputPathAux + " -t Task100_grid -z -m 3d_lowres -f " + str(fold))

        npzDirs.append(outputPath)

        print(f"Fold {fold} completed")
        print("                      ")

    outputDir = os.path.join(inputDir, patId, "output")
    if not os.path.isdir(outputDir): os.mkdir(outputDir)

    if outputDir[:8] == "/content":
        outputDir = outputDir[:17] + "\ " + outputDir[18:]

    print("Starting ensembling")

    os.system(f"nnUNet_ensemble -f {npzDirs[0]} {npzDirs[1]} {npzDirs[2]} {npzDirs[3]} {npzDirs[4]} -o {outputDir}")

    print(f"Ensembling took {time() - start} s")
    print("                                   ")