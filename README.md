# Unsupervised phase discovery with deep anomaly detection

This repository contains the code to our paper [Unsupervised phase discovery via anomaly detection](https://github.com/Qottmann/phase-discovery-anomaly-detection/edit/master/README.md), where we use deep neural networks auto encoders (AE) to find phase transitions in the one dimensional extended Bose-Hubbard model.

The repository provides the code to generate the training data with DMRG and the whole training pipeline for the AE.

## Run the code

You will have to install the following packages:

- tensorflow
- numpy
- matplotlib
- TenPy (`pip install physics-tenpy`)

To check if the TenPy installation was succesful run `AD_tools.py` for a test run.

The Jupyter Notebook `Bose_Hubbard.ipynb` contains all the necessary elements to draw the phase diagram along $U=5$ with $V \in [0,5]$.

The Jupyter Notebook `Bose_Hubbard.ipynb` allows to load data and plot the whole phase diagram along $U$ and $V$.
