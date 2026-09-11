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
    packages=find_packages(),
    package_data={
        "arterial": [
            "segmentation/models/*",
            "vessel_labelling/models/*",
            "landmark_extraction/models/*"],
        "tests": ["test_data/*"]
        },
    install_requires=[
        'nibabel',
        'networkx',
        'numpy',
        'scipy',
        'scikit-image',
        'mycolorpy',
        'vtk',
        'connected-components-3d'
        ]
)
