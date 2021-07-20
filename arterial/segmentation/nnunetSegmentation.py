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
import argparse

from time import time

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument("-mode", "--mode", type=str, required=True, 
    help="choose between `inference` or `ensemble`. Required.")

parser.add_argument("-casePath", "--casePath", type=str, required=True,
    help="absolute path to the input image (has to be a nifti). Required")


args = parser.parse_args()

mode     = args.mode
casePath = args.casePath
    
##############################################################################################
#--------------------------------------------------------------------------------------------#    

print("                                                                                     ")
print("Model optimization for nnUNet (Isensee et al. 2020) for segmentation of 3D CTA images")
print("By Pere Canals (2020)                                                                ")
print("                                                                                     ")

#--------------------------------------------------------------------------------------------#    
##############################################################################################

def inference(casePath):
    ''' Performs inference of casePath (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    outputPath.

    Arguments:
        - casePath <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''

    patId = os.path.basename(casePath)[:-7] # Name of the nifti casePath except the .nii.gz extension
    caseDir = os.path.dirname(casePath)

    # Paths used by nnUNet
    inputPath = os.path.join(caseDir, "input")
    outputPath = os.path.join(caseDir, "segmentation")
    if not os.path.isdir(inputPath): os.mkdir(inputPath)
    if not os.path.isdir(outputPath): os.mkdir(outputPath)
    
    # The casePath will be placed in a newly created dir with the patId as name
    if not os.path.isfile(os.path.join(inputPath, patId + "_0000.nii.gz")):
        os.rename(casePath, os.path.join(inputPath, patId + "_0000.nii.gz"))

    # Only use if running on Colab
    if inputPath[:8] == "/content": # If we are working on Colab and Drive, we need to get rid of spaces in the path
        inputPath = inputPath[:17] + "\ " + inputPath[18:]

    start = time()

    os.system("nnUNet_predict -i " + inputPath + " -o " + outputPath + f" -t Task001_Arterial -m 3d_lowres -f all")

    shutil.copyfile(os.path.join(inputPath, patId + "_0000.nii.gz"), os.path.join(caseDir, patId + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(outputPath, patId + ".nii.gz"), os.path.join(caseDir, patId + ".nii.gz"))

    print(f"Inference took {time() - start} s")
    print("                                  ")


def ensemble(casePath):
    ''' Performs inference of casePath (nifti, CTA) by ensembling all folds of 
    the best performing model to output a binary mask in a nifti format. The resulting 
    nifti will be placed in outputDir.

    Arguments:
        - casePath <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''

    patId = os.path.basename(casePath)[:-7]
    caseDir = os.path.dirname(casePath)

    inputPath = os.path.join(caseDir, "input")
    if not os.path.isdir(inputPath): os.mkdir(inputPath)
    if not os.path.isdir(os.path.join(caseDir, "ensemble")): os.mkdir(os.path.join(caseDir, "ensemble"))

    # Only use if running on Colab. Make sure inputPath and outputPath do not contain spaces
    if inputPath[:8] == "/content": # If we are working on Colab and Drive, we need to get rid of spaces in the path
        inputPath = inputPath[:17] + "\ " + inputPath[18:]
    
    if not os.path.isfile(os.path.join(inputPath, patId + "_0000.nii.gz")):
        os.rename(casePath, os.path.join(inputPath, patId + "_0000.nii.gz"))

    npzDirs = []

    start = time()

    for fold in range(5):
        print(f"Predicting {patId} fold {fold}")
        print("                               ")

        outputPath = os.path.join(caseDir, "ensemble", f"fold_{fold}")
        if not os.path.isdir(outputPath): os.mkdir(outputPath)

        # Only use if running on Colab. Make sure inputPath and outputPath do not contain spaces
        if outputPath[:8] == "/content": # If we are working on Colab and Drive, we need to get rid of spaces in the path
            outputPathAux = outputPath[:17] + "\ " + outputPath[18:]
        else:
            outputPathAux = outputPath
            
        os.system("nnUNet_predict -i " + inputPath + " -o " + outputPathAux + " -t Task001_Arterial -z -m 3d_lowres -f " + str(fold) + "--part_id=0 --part_id=1 --part_id=2 --part_id=3 --num_parts=4")

        npzDirs.append(outputPath)

        print(f"Fold {fold} completed")
        print("                      ")

    outputDir = os.path.join(caseDir, "ensemble", "output")
    if not os.path.isdir(outputDir): os.mkdir(outputDir)

    if outputDir[:8] == "/content":
        outputDir = outputDir[:17] + "\ " + outputDir[18:]

    print("Starting ensembling")

    os.system(f"nnUNet_ensemble -f {npzDirs[0]} {npzDirs[1]} {npzDirs[2]} {npzDirs[3]} {npzDirs[4]} -o {outputDir}")

    shutil.copyfile(os.path.join(inputPath, patId + "_0000.nii.gz"), os.path.join(caseDir, patId + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(outputDir, patId + ".nii.gz"), os.path.join(caseDir, patId + ".nii.gz"))

    print(f"Ensembling took {time() - start} s")
    print("                                   ")

##############################################################################################

if mode == "inference":
    inference(casePath)
elif mode == "ensemble":
    ensemble(casePath)
else:
    ValueError("Please introduce one of the possible modes. See main.py -h for more information.")