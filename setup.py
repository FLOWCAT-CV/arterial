#!/usr/bin/env python
#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from setuptools import setup, find_packages

setup(
    name="arterial",
    version="2.1",
    description="Arterial: framework for vascular tortuosity analysis",
    author="Pere Canals",
    author_email="perecanalscanals@gmail.com",
    license="PolyForm-Noncommercial-1.0.0",
    license_files=["LICENSE"],
    url="https://github.com/FLOWCAT-CV/arterial",
    python_requires=">=3.11",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    # Trained weights are not packaged: they are downloaded to arterial/models/
    # by download_models.sh and resolved through arterial.model_registry.
    install_requires=[
        "nibabel",
        "networkx",
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "vtk",
        "connected-components-3d",
        # Deep-learning stack. The README installs torch and the PyTorch Geometric
        # wheels first so that the CUDA build is chosen; these entries only make
        # the dependency explicit for an already prepared environment.
        "torch",
        "torch_geometric",
        "monai",
        "torchio",
        "nnunetv2",
        # vmtk is conda-only (conda install -c conda-forge vmtk) and cannot be listed here.
    ],
    extras_require={
        "dicom": ["SimpleITK", "dicom2nifti"],   # arterial.io.dicom_and_nifti converters
        "registration": ["antspyx"],             # arterial.io.registration
    },
)
