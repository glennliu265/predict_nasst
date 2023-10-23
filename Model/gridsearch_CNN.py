#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Gridsearch Test Script for CNNs

Copied section from train_gridsearch.py (for FNNs)

Created on Mon Oct 23 12:43:26 2023

@author: gliu
"""


import numpy as np
import itertools

import sys
import numpy as np
import os
import time
import tqdm
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset

# <Start copy from train_NN_CESM1.py>  ========================================
#%% Load custom packages and setup parameters

machine = 'stormtrack' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd     = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

# Load Predictor Information
bbox          = pparams.bbox
# Import general utilities from amv module
pkgpath = pparams.machine_paths[machine]['amv_path']
sys.path.append(pkgpath)
from amv import proc

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "CNN2_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames            = ['SSH','SST',]#"SST","SSS","SLP","NHFLX",]       # Names of predictor variables
leads               = np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,50,1)    # Which runs to do

# Other toggles
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = True                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# ----------------------------------------------
#%% 02. Data Loading, Classify Targets
# ----------------------------------------------

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Debug messages
if debug:
    print("Loaded data of size: %s" % (str(data.shape)))

"""
# Output: 
    predictors       :: [channel x ens x year x lat x lon]
    target_class     :: [ens x year]
"""


#%% Make a Class

def build_simplecnn(param_dict,num_classes,nlat=224,nlon=224,num_inchannels=3):
    
    # 2 layer CNN settings
    nchannels      = param_dict['nchannels']#
    filtersizes    = param_dict['filtersizes']
    filterstrides  = param_dict['filterstrides']
    poolsizes      = param_dict['poolsizes']#[[2,3],[2,3]]
    poolstrides    = param_dict['poolstrides']#[[2,3],[2,3]]
    activations    = param_dict['activations']
    dropout        = param_dict['dropout']
    firstlineardim = am.calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
    
    layers = []
    nlayers = len(nchannels)
    
    for l in range(nlayers):
        if l == 0: # 1st Layer
            # Make + Append Convolutional Layer
            conv_layer = nn.Conv2d(in_channels=num_inchannels,out_channels=nchannels[l], kernel_size=filtersizes[l], stride=filterstrides[l])
            layers.append(conv_layer)
        else: # All other layers
            # Make + Append Convolutional Layer
            nn.Conv2d(in_channels=nchannels[l-1], out_channels=nchannels[l], kernel_size=filtersizes[l], stride=filterstrides[l])
            layers.append(conv_layer)
        
        # Append Activation
        layers.append(activations[l])
        
        # Make+Append Pooling layer
        pool_layer = nn.MaxPool2d(kernel_size=poolsizes[l], stride=poolstrides[l])
        layers.append(pool_layer)
        
        if l == (nlayers-1): # Final Layer (Flatten and add Fully Connected)
            layers.append(nn.Flatten())
            layers.append(nn.Dropout(p=dropout))
            linear_layer = nn.Linear(in_features=firstlineardim,out_features=num_classes)
            layers.append(linear_layer)
    return layers

#%%

nlat = 69
nlon = 65

# 2-Layer CNN (original)
cnn_param_dict_ori = {
    "nchannels"     : [32,64],
    "filtersizes"   : [[2,3],[3,3]],
    "filterstrides" : [[1,1],[1,1]],
    "poolsizes"     : [[2,3],[2,3]],
    "poolstrides"   : [[2,3],[2,3]],
    }

# 2-Layer CNN (corrected, basedon diagram)
cnn_param_dict = {
    "nchannels"     : [32,64],
    "filtersizes"   : [[2,3],[2,3]],
    "filterstrides" : [[1,1],[1,1]],
    "poolsizes"     : [[3,3],[3,3]],
    "poolstrides"   : [[2,3],[2,3]],
    }

# Set the Test Values
nchannels     = ([32,],[32,64],[32,64,128]) 
filtersizes   = ([2,2],[3,3],[4,4])
filterstrides = (2,3,4)
poolsizes     = copy.deepcopy(filtersizes)
poolstrides   = copy.deepcopy(filterstrides)

# Set up testing dictionary
test_param_names  = ["nchannels","filtersizes","filterstrides","poolsizes","poolstrides"]
test_param_values = [nchannels  ,filtersizes  ,filterstrides  ,poolsizes  ,poolstrides]
test_params       = dict(zip(test_param_names,test_param_values))
param_combinations = list(itertools.product(*test_param_values))
ncombos            = len(param_combinations)

#%%  A Simpler Test

"""
Let's set up a simpler test, where:
    (1) strides + filter sizes are symmetric, and 
    (2) the pool and filter sizes are also the same...
"""

in_channels = 1
num_classes = 3

# Set the Test Values
nchannels     = ([32,],[32,64],[32,64,128]) 
filtersizes   = ([2,2],[3,3],[4,4])
filterstrides = (2,3,4)
poolsizes     = copy.deepcopy(filtersizes)
poolstrides   = copy.deepcopy(filterstrides)

# Set up testing dictionary (reduced complexity)
test_param_names   = ["nchannels","filtersizes","filterstrides",]
test_param_values  = [nchannels  ,filtersizes  ,filterstrides  ,]
test_params        = dict(zip(test_param_names,test_param_values))
param_combinations = list(itertools.product(*test_param_values))
ncombos            = len(param_combinations)

# Set up dictionaries and build CNNs
all_cnns = []
for n in range(ncombos):
    
    nchannels,filtersize,stridesize=param_combinations[n]
    stridesize_in = [stridesize,stridesize]
    nlayers = len(nchannels)
    cnn_param_dict = {
        "nchannels"     : nchannels,
        "filtersizes"   : [filtersize,]*nlayers,
        "filterstrides" : [stridesize_in,]*nlayers,
        "poolsizes"     : [filtersize,]*nlayers,
        "poolstrides"   : [stridesize_in,]*nlayers,
        "activations"   : [nn.ReLU(),]*nlayers,
        "dropout"       : 0,
        }
    
    cnnmod = build_simplecnn(cnn_param_dict,num_classes,
                        nlat=nlat,nlon=nlon,num_inchannels=in_channels)
    
    all_cnns.append(cnnmod)
    


#%%


