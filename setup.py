#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="arterial",
      version="1.0",
      description="Arterial: framework for vascular tortuosity analysis",
      author="Pere Canals",
      author_email="perecanalscanals@gmail.com",
      packages=find_packages(),
      package_data={"arterial": ["segmentation/models/nnUNet/3d_lowres/Task001_Arterial/nnUNetTrainerV2__nnUNetPlansv2.1/*.pkl",
                                 "segmentation/models/nnUNet/3d_lowres/Task001_Arterial/nnUNetTrainerV2__nnUNetPlansv2.1/all/*.model",
                                 "segmentation/models/nnUNet/3d_lowres/Task001_Arterial/nnUNetTrainerV2__nnUNetPlansv2.1/all/*.pkl",
                                 "segmentation/models/nnUNet/3d_lowres/Task001_Arterial/nnUNetTrainerV2__nnUNetPlansv2.1/all/*.json",
                                 "segmentation/models/nnUNet/3d_fullres/Task002_Cerebral/nnUNetTrainerV2__nnUNetPlansv2.1/*.pkl",
                                 "segmentation/models/nnUNet/3d_fullres/Task002_Cerebral/nnUNetTrainerV2__nnUNetPlansv2.1/all/*.model",
                                 "segmentation/models/nnUNet/3d_fullres/Task002_Cerebral/nnUNetTrainerV2__nnUNetPlansv2.1/all/*.pkl",
                                 "segmentation/models/nnUNet/3d_fullres/Task002_Cerebral/nnUNetTrainerV2__nnUNetPlansv2.1/all/*.json"]},
      install_requires=[
        'nnunet',
        'nibabel',
        'networkx',
        'torch_geometric',
        'torch_scatter',
        'torch_sparse',
        'torch_cluster',
        'numpy',
        'scipy',
        'scikit-image',
        'mycolorpy'
    ]
)
