# Unsupervised phase discovery with deep anomaly detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3601485.svg)](https://doi.org/10.5281/zenodo.3601485)

This repository contains the code to our paper [Unsupervised phase discovery via anomaly detection](https://github.com/Qottmann/phase-discovery-anomaly-detection/edit/master/README.md), where we use deep neural networks auto encoders (AE) to find phase transitions in the one dimensional extended Bose-Hubbard model.

The repository provides the code to generate the training data with DMRG and the whole training pipeline for the AE.

## Run the code

You will have to install the following packages:

- tensorflow (v1.12.1)
- numpy (1.17.3)
- matplotlib (3.1.1)
- TenPy (> v0.4.0) (`pip install physics-tenpy`)

The versions in brackets indicate with which version the code was tested.
To check if the TenPy installation was succesful run `AD_tools.py` for a test run.

The Jupyter Notebook `Bose_Hubbard.ipynb` contains all the necessary elements to draw the phase diagram along 

<a href="https://www.codecogs.com/eqnedit.php?latex=U=5&space;\text{&space;with&space;}&space;V&space;\in&space;[0,5]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U=5&space;\text{&space;with&space;}&space;V&space;\in&space;[0,5]" title="U=5 \text{ with } V \in [0,5]" /></a>.

The Jupyter Notebook `Bose_Hubbard_precalc.ipynb` allows to load data and plot the whole phase diagram in 2D with

<a href="https://www.codecogs.com/eqnedit.php?latex=$U&space;\in&space;[0,8]$&space;\text{&space;and&space;}&space;$V&space;\in&space;[0,5]$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$U&space;\in&space;[0,8]$&space;\text{&space;and&space;}&space;$V&space;\in&space;[0,5]$" title="$U \in [0,8]$ \text{ and } $V \in [0,5]$" /></a>.
