# predict_nasst
Train Neural Networks to Predict North Atlantic Sea Surface Temperature


## Introduction

Code for the manuscript *Physical Insights from the Multidecadal Prediction of North Atlantic Sea Surface Temperature Variability Using Explainable Neural Networks* (in preparation). Written in Python.

The scripts train Fully-connected Neural Networks (NNs) to predict the state of North Atlantic Sea Surface Temperature (NASST) Index, given snapshots of a given predictor. It is written to work with Community Earth System Model 1 (CESM1) Large Ensemble output from the historical period (1920-2005), though could be adapted to work with other model large ensembles and scenarios. Layerwise Relevance Propagation (LRP) is used to create relevance composites for each NASST state.

## Directory Structure and Notes
Scripts assume the raw data (CESM1 output) is placed in \[../CESM_data\]
Additional dependencies include the Pytorch-LRP package (https://github.com/moboehle/Pytorch-LRP) and functions located in the \[amv\] repository (https://github.com/glennliu265/amv). These are assumed to be placed in the directory outside the repository (relative paths: ../amv/ and ../Pytorch-LRP/).

A subset of the data and results (~1.6 GB) used in Manuscript_Figures.ipynb is available on Google Drive (https://drive.google.com/drive/folders/12VljJ5iYGL08hkE3pQBS-C28kAmu2Fbi?usp=sharing), but will be uploaded to Zenodo after the peer review process (doi will be added here).

## Set-up
*note: add script names later*
1. Run the yml file to set-up the Python environment with required packages
2. Place downloaded CESM1 data into target folder (../CESM_data/) and run preprocessing scripts.
3. Run NN training scripts
4. Run Analysis script to compute test metrics
5. Visualize output using Analysis scripts.

## Contents
![Draft Workflow Image](https://github.com/glennliu265/predict_nasst/blob/main/Figures/Draft_Workflow.pdf)
### Preprocessing
Scripts for preparing raw CESM1 data into the predictors and targets.

### Models
Scripts for training and testing the NNs

### Analysis
Scripts for computing test metrics, performing LRP, and visualizing output.

### Working_Copies
Additional analyses and work in progress--documentation not complete.

