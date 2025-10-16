#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="arterial",
    version="2.1",
    description="Arterial: framework for vascular tortuosity analysis",
    author="Pere Canals",
    author_email="perecanalscanals@gmail.com",
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
