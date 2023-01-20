# Arterial: an AI framework for the automated analysis of vascular tortuosity

Mechanical thrombectomy (MT) is considered as the gold standard treatment for acute ischemic stroke (AIS). Studies show that up to 30% of MT patients register abnormally long procedural times, and in about 3% of cases, catheterization through femoral access is impossible. Most of these long procedures are linked to complications due to the presence of vascular tortuosity in supra-aortic and cerebral arteries, which oppose difficulties upon navigation. In an attempt to minimize time loss in these cases we propose ARTERIAL, an artificial intelligence (AI) framework for the fully automatic assessment of vascular tortuosity and operation planning for MT. ARTERIAL is born with the promise to deliver accurate and robust predictions of procedural times from all possible access sites and recognition of potential intra-operation difficulties for endovascular treatment based on machine learning (ML) models, enabling a powerful, objective and personalized analysis for each patient prior to intervention. 

Arterial is conceived to be deployed as an operation planning and decision support tool prior to endovascuklar intervention for AIS patients. This is a challenging medical emergency setting which requires a robust, rapid and objective analysis, only taking the protocolary imaging (namely non-contrast CT (NCCT) and angio-CT (CTA)) as well as patient metadata that can be gathered before the patient arrival. Therefore, we adjust the analysis as much as possible taking this into account. Moreover, we believe that the only way to achieve the described qualities that the analysis requires is to achieve a fully automated process, without the need of any kind of manual input on the analysis.

In this repository we will be posting the development of Arterial, which is the core project of the Pere Canals' doctoral thesis.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
  * [nnU-Net](#nnU-Net)
  * [Slicer](#Slicer)
  * [VMTK](#VMTK)
  * [GraphNets](#GNNs)
- [Usage](#usage)
    * [FullAnalysis](#FullAnalysis)
    * [SegmentationOnly](#SegmentationOnly)

<!-- # Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  * [How to run nnU-Net on a new datasets](#how-to-run-nnu-net-on-a-new-datasets)
    + [Dataset conversion](#dataset-conversion)
    + [Experiment planning and preprocessing](#experiment-planning-and-preprocessing)
    + [Model training](#model-training)
      - [2D U-Net](#2d-u-net)
      - [3D full resolution U-Net](#3d-full-resolution-u-net)
      - [3D U-Net cascade](#3d-u-net-cascade)
        * [3D low resolution U-Net](#3d-low-resolution-u-net)
        * [3D full resolution U-Net](#3d-full-resolution-u-net-1)
      - [Multi GPU training](#multi-gpu-training)
    + [Identifying the best U-Net configuration(s)](#identifying-the-best-u-net-configuration)
    + [Run inference](#run-inference)
  * [How to run inference with pretrained models](#how-to-run-inference-with-pretrained-models)
  * [Examples](#Examples)
- [Extending/Changing nnU-Net](#extending-changing-nnu-net)
- [FAQ](#faq) 

ecotrust-canada.github.io/markdown-toc/ -->

## Description of the process

Arterial is composed by several external packages for the different tasks that have to be implemented in order to run the complete analysis. As the first step, Isensee's et al. nnU-Net [1] has been implemented and optimized for the segmentation of the arterial anatomy of the patient from the CTA images. This deep learning network inputs a CTA volume (nifti format) and outputs the corresponding binary mask with a background (not vessel) as zeros and a foreground (vessel) as ones, also in nifti format. 

We include the possibility to run inference with a single model or to ensemble 5 folds of the same model (varying the training dataset for each fold) and averaging the resulting probabilities for each voxel. This can improve the quality of the segmentation in exchange of a larger computational cost. 

After segmentation, the binary map is processed using open-source software Slicer [2] and the Vascular Modelling ToolKit (VMTK). First, the binary map is segmented by thresholding and VMTK is used to automatically extract the centerline model of the vascular object. In parallel, VMTK is also used to split the surface model of the segmentation into individual clipped branches, which are linked to the centerline model as well. 

The centerline model is used to generate a graph used for vessel labeling. We have implemented a graph neural net (GNN) for the 


    [1] Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein "Automated Design of Deep Learning Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).
    [2] Fedorov A., Beichel R., Kalpathy-Cramer J., Finet J., Fillion-Robin J-C., Pujol S., Bauer C., Jennings D., Fennessy F.M., Sonka M., Buatti J., Aylward S.R., Miller J.V., Pieper S., Kikinis R. 3D Slicer as an Image Computing Platform for the Quantitative Imaging Network. Magn Reson Imaging. 2012 Nov;30(9):1323-41. PMID: 22770690. PMCID: PMC3466397.

## Installation

We strongly recommend the creation of a virtual environment for the installation of Arterial along with the rest of external packages.

### Setting up environment paths

Defining a series of environment paths

### nnU-Net

For installation of the nnU-Net framework please refer to the [nnU-Net GitHub site](https://github.com/MIC-DKFZ/nnUNet), and follow the steps for installation for running inference with pre-trained models. You have to first install [PyTorch](https://pytorch.org/) (> 1.6), then `pip install nnunet` and finally set up the 

After the out-of-the-box nnU-Net setup is completed

### Slicer

SlicerPython path should be set as an environmental variable or sth?

### VMTK

### GraphNets