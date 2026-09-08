#!/usr/bin/env python
#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0

from setuptools import setup, find_packages

setup(
    name="arterial",
    version="2.1",
    description="Arterial: framework for vascular tortuosity analysis",
    author="Pere Canals",
    author_email="perecanalscanals@gmail.com",
    license="CC-BY-NC-4.0",
    license_files=["LICENSE"],
    url="https://github.com/FLOWCAT-CV/arterial",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
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
        'connected-components-3d',
        'opencv-python'
        ]
)
