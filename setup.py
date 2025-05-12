#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="arterial",
    version="2.0",
    description="Arterial: framework for vascular tortuosity analysis",
    author="Pere Canals",
    author_email="perecanalscanals@gmail.com",
    packages=find_packages(),
    package_data={
        "arterial": [
            "segmentation/models/*",
            "vessel_labelling/models/*"],
        "tests": ["test_data/*"]
        },
    install_requires=[
        'nnunetv2',
        'nibabel',
        'networkx',
        'numpy',
        'scipy',
        'scikit-image',
        'mycolorpy',
        'vtk',
        'torchio'
        ]
)
