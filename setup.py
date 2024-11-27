# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:21:47 2024

@author: austa
"""

from setuptools import setup, find_packages

setup(
    name="pfinder",  # The name of your package
    version="1.0.0",  # Version number
    author="Your Name",  # Replace with your name
    author_email="your_email@example.com",  # Replace with your email
    description="A Python module for image registration, keypoint selection, and pressure calculation",
    packages=find_packages(),  # Automatically find all packages in the directory
    install_requires=[
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "matplotlib>=3.5.3",
        "opencv-python>=4.10.0.84",
        "imageio>=2.31.2",
        "scikit-image>=0.19.3",
        "tqdm>=4.66.4",
        "pillow>=9.5.0",
        "scipy>=1.7.3",
        "cython>=3.0.10",
        "pytest>=7.4.4",
        "pywavelets>=1.3.0",
        "seaborn>=0.12.2",
        "joblib>=1.3.2",
    ],
    python_requires=">=3.7",  # Specify the Python version compatibility
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",  # License type
    url="https://github.com/your_username/pfinder",  # Replace with your GitHub repo URL
    long_description=open("README.md").read(),  # Include the README as the long description
    long_description_content_type="text/markdown",
)
