#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

===============
Training Params
===============

File containing input training parameters for the training script..
Imports into train_NN_CESM1.py, such that the data is loaded there.

Created on Thu Mar 23 09:21:09 2023

@author: gliu

-------------------
Current Experiments
-------------------
    
        ~ Set 1: Cross-Validation Experiments
    FNN4_128_Rewrite                                        : Testing (20 epochs) for rewritten NN training script in late March 2023. Uses [train_NN_CESM1.py]
    FNN4_128_SingleVar_CV                                   : k-fold Cross Validation script, where testing % = 0.30. Uses [train_NN_CESM1_CV.py], copies FNN4_128_SingleVar_Rewrite
    FNN4_128_SingleVar_CV_consistent                        : Use consistent sample. Copies FNN4_128_SingleVar_Rewrite.
    
        ~ Set 2: Old Parametersfrom before the script rewrite
    FNN4_128_SingleVar                                      : Old script training FNN4_128 for single predictors, preior to rewrite. Uses [NN_test_lead_ann_ImageNet_classification_singlevar.py]
    FNN4_128_SingleVar_detrend                              : Same as above, detrended. Copies FNN4_128_SingleVar
    FNN4_128_SingleVar_Rerun100                             : Rerun old script by with 100 networks instead of 50.
    
         ~ Set 3: Debugging new script (large inter-model differences, testing consistent sampling)
    FNN4_128_SingleVar_Rerun100_consistent                  : Same as FNN4_128_SingleVar_Rerun100 but uses consistent sampling
    FNN4_128_SingleVar_debug1_shuffle_all                   : Same as FNN4_128_SingleVar_Rerun100, but with shuffled samples
    FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_32bs     : Copies FNN4_128_SingleVar_debug1_shuffle_all, but switch to old number of epochs, earlystopping, batch size
    FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs     : Copies above, but reduce batchsize to 16
    FNN4_128_SingleVar_debug1_shuffle_all_no_val            : Copies above, but remove validation set
    FNN4_128_SingleVar_debug1_shuffle_all_no_val_8020       : Copies above, but changes Test-Train split to the old 20-80
    
         ~ Set 4: Trying PiC training
    FNN4_128_SingleVar_PIC                                  : Training script for PiC data, same settings as original FNN4_128_SingleVar
    
         ~ Set 5: Regaining my senses in June after writing up the draft....
    FNN4_128_SingleVar_Rewrite_June                         : Copies FNN4_128_SingleVar exactly to rerun in June 2023
    FNN4_128_SingleVar_Testing                              : Copies FNN4_128_SingleVar, trying new Train.Val split of 60-10
    
         ~ Set 6: Testing Index Normalization
    FNN4_128_SingleVar_Norm0                                : FNN4 run with UNnormalized index, 32 ensemble members
    FNN4_128_SingleVar_Norm1                                : FNN4 run with normalized index, 32 ensemble members
    
         ~ Set 7: Reducing training to 32 ens members for training run
    FNN4_128_SingleVar_PaperRun                             : 100-epoch run for Predict AMV Draft
    FNN4_128_SingleVar_PaperRun_detrended                   : Detrended version of above run
    CNN2_PaperRun                                           : Corresponding CNN2 run with architecture that works with Captum
    FNN6_128_SingleVar_PaperRun                             : 100 epoch run for 6-layer FNN
    
"""

# Load my own custom modules
import sys
import os
from torch import nn
import predict_amv_params as pparams
import numpy as np

# Import Custom Packages
cwd = os.getcwd()
sys.path.append(cwd+"/../")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
sys.path.append("../")
import predict_amv_params as pparams

# Dictionary Containing All Parameters
train_params_all = {}

# --------------------------
#%% Experiment dictionaries 
# =============================
# Note: Copy section below and change parameters. These are stored in a dict [expdict]
#       that can be accessed via a unique key of your choosing [expname]

#%% FNN4_128_SingleVar_Rewrite
"""

FNN4_128_SingleVar_Rewrite

Single Variable Training for 4 Layer Fully-Connected NN

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_Rewrite"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
#expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['norm']            = True     # Indicate if target was normalized
expdict['stdspace']        = False    # Set to true to standardize predictors in space
expdict['mask']            = True     # True for land-ice masking
expdict["PIC"]             = False    # Use PiControl Data

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']             = 42      # Number of ensemble members to limit to
expdict['ystart']          = 1920    # Start year of processed dataset
expdict['yend']            = 2005    # End year of processed dataset
expdict['bbox']            = pparams.bbox #  Bounding box for data

# Label Determination
expdict['quantile']        = False   # Set to True to use quantiles
expdict['thresholds']      = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']             = 300               # Number of samples from each class to train with. None = size of minimum class
expdict['percent_train']        = 0.60              # Training percentage
expdict['percent_val']          = 0.10              # Validation Percentage
expdict['shuffle_class']        = True              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit']   = True             # Set to False to maintain same set for train/test/val split

# Cross Validation Options
expdict['cv_loop']              = False             # Repeat for cross-validation
expdict['cv_offset']            = 0                 # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
expdict['netname']       = "FNN4_128"           # Key for Architecture Hyperparameters
expdict['loss_fn']       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']   = False                # Set to true to change final layer to softmax
expdict['reduceLR']      = False                # c
expdict['LRpatience']    = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']    = 3                    # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 20                   # Maximum # of Epochs to train for
expdict['batch_size']    = 16                   # Pairs of predictions
expdict['unfreeze_all']  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer


train_params_all[expname] = expdict.copy()
#%% FNN4_128_SingleVar_CV
"""
FNN4_128_SingleVar_CV

Single Variable Training for 4 Layer Fully-Connected NN, Cross Validation Test
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                         = "FNN4_128_SingleVar_CV"

# Copy dictionary from above
expdict = train_params_all["FNN4_128_SingleVar_Rewrite"].copy()

# Cross Validation Options
expdict['cv_loop']              = True    # Repeat for cross-validation
percent_test                    = 1 - (expdict['percent_train'] + expdict['percent_val'])
expdict['cv_offset']            = percent_test       # Set cv option. Default is test size chunk
expdict['shuffle_class']        = False              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit']   = True             # Set to False to maintain same set for train/test/val split


train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_CV_consistent
"""
FNN4_128_SingleVar_CV_consistent

Single Variable Training for 4 Layer Fully-Connected NN, Cross Validation Test, wuith consistent sampling
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_CV_consistent"

# Copy dictionary from above
expdict = train_params_all["FNN4_128_SingleVar_Rewrite"].copy()

# Cross Validation Options
expdict['cv_loop']         = True    # Repeat for cross-validation
percent_test               = 1 - (expdict['percent_train'] + expdict['percent_val'])
expdict['cv_offset']       = percent_test       # Set cv option. Default is test size chunk
expdict['shuffle_class']        = False              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit']   = False             # Set to False to maintain same set for train/test/val split


train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar

"""

FNN4_128_SingleVar

Old Singlevar Script, prior to rewrite

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
#expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['norm']            = True     # Indicate if target was normalized
expdict['stdspace']        = False    # Set to true to standardize predictors in space
expdict['mask']            = True     # True for land-ice masking
expdict["PIC"]             = False    # Use PiControl Data

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']             = 40      # Number of ensemble members to limit to
expdict['ystart']          = 1920    # Start year of processed dataset
expdict['yend']            = 2005    # End year of processed dataset
expdict['bbox']            = pparams.bbox #  Bounding box for data

# Label Determination
expdict['quantile']        = False   # Set to True to use quantiles
expdict['thresholds']      = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']        = 300     # Number of samples from each class to train with. None = size of minimum class
expdict['percent_train']   = 0.80    # Training percentage
expdict['percent_val']     = 0.00    # Validation Percentage
expdict['shuffle_class']      = True              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit'] = True             # Set to False to maintain same set for train/test/val split

# Cross Validation Options
expdict['cv_loop']         = False    # Repeat for cross-validation
expdict['cv_offset']       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
expdict['netname']       = "FNN4_128"           # Key for Architecture Hyperparameters
expdict['loss_fn']       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']   = False                # Set to true to change final layer to softmax
expdict['reduceLR']      = False                # Set to true to use LR scheduler
expdict['LRpatience']    = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']    = 3                    # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 20                   # Maximum # of Epochs to train for
expdict['batch_size']    = 16                   # Pairs of predictions
expdict['unfreeze_all']  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer

train_params_all[expname] = expdict.copy()

#%% FNN4_128_detrend
"""
FNN4_128_SingleVar_detrend

Old Singlevar Script (withdetrend), prior to rewrite
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_detrend"
expdict                    = {}

# Copy dictionary from above
expdict                         = train_params_all["FNN4_128_SingleVar"].copy()

expdict['detrend']         = 1        # True if the target was detrended

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_Rerun100
"""

FNN4_128_SingleVar_Rerun100

Re-running the SingleVar. 100 Networks perleadtime, no shuffle

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_Rerun100"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
#expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['norm']            = True     # Indicate if target was normalized
expdict['stdspace']        = False    # Set to true to standardize predictors in space
expdict['mask']            = True     # True for land-ice masking
expdict["PIC"]             = False    # Use PiControl Data

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']             = 42      # Number of ensemble members to limit to
expdict['ystart']          = 1920    # Start year of processed dataset
expdict['yend']            = 2005    # End year of processed dataset
expdict['bbox']            = pparams.bbox #  Bounding box for data

# Label Determination
expdict['quantile']        = False   # Set to True to use quantiles
expdict['thresholds']      = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']        = 400     # Number of samples from each class to train with. None = size of minimum class
expdict['percent_train']   = 0.60    # Training percentage
expdict['percent_val']     = 0.10    # Validation Percentage
expdict['shuffle_class']        = True              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit']   = True             # Set to False to maintain same set for train/test/val split

# Cross Validation Options
expdict['cv_loop']         = False    # Repeat for cross-validation
expdict['cv_offset']       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------
# Network Hyperparameters
expdict['netname']       = "FNN4_128"           # Key for Architecture Hyperparameters
expdict['loss_fn']       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']   = False                # Set to true to change final layer to softmax
expdict['reduceLR']      = False                # Set to true to use LR scheduler
expdict['LRpatience']    = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']    = 10                   # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 100                  # Maximum # of Epochs to train for
expdict['batch_size']    = 32                   # Pairs of predictions
expdict['unfreeze_all']  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_Rerun100_consistent
"""
FNN4_128_SingleVar_Rerun100_consistent

Same as above, but with a consistent sample to assess weight initilization effects
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                         = "FNN4_128_SingleVar_Rerun100_consistent"

# Copy dictionary from above
expdict                         = train_params_all["FNN4_128_SingleVar_Rerun100"].copy()

# Cross Validation Options
expdict['shuffle_class']        = False            # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit']   = False             # Set to False to maintain same set for train/test/val split


train_params_all[expname]       = expdict.copy()

#%% FNN4_128_SingleVar_debug1_shuffle_all
"""
FNN4_128_SingleVar_debug1_shuffle_all

Try rerunning the script above, but with shuffled data.
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                         = "FNN4_128_SingleVar_debug1_shuffle_all"

# Copy dictionary from above
expdict                         = train_params_all["FNN4_128_SingleVar_Rerun100"].copy()

# Cross Validation Options
expdict['shuffle_class']        = False              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit']   = False              # Set to False to maintain same set for train/test/val split


# Runids 
train_params_all[expname]       = expdict.copy()


#%% FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_32bs
""" Same as above, but adapt the old amount of epochs"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname= "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_32bs"

# Copy dictionary from above
expdict                  = train_params_all["FNN4_128_SingleVar_debug1_shuffle_all"].copy()

expdict['early_stop']    = 3#10                   # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 20#100                 # Maximum # of Epochs to train for
expdict['batch_size']    = 32                     # Pairs of predictions

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs
"""Sme as above, but reduce batch size"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                  = "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs"

# Copy dictionary from above
expdict                  = train_params_all["FNN4_128_SingleVar_debug1_shuffle_all"].copy()

expdict['early_stop']    = 3#10                   # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 20#100                  # Maximum # of Epochs to train for
expdict['batch_size']    = 16                   # Pairs of predictions

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_debug1_shuffle_all_no_val

"""Sme as above, but remove validation set"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                  = "FNN4_128_SingleVar_debug1_shuffle_all_no_val"

# Copy dictionary from above
expdict                  = train_params_all["FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs"].copy()

expdict['percent_val']   = 0

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_debug1_shuffle_all_no_val_8020
"""Sme as above, but increase training size"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                  = "FNN4_128_SingleVar_debug1_shuffle_all_no_val_8020"

# Copy dictionary from above
expdict                  = train_params_all["FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs"].copy()

expdict['percent_train']   = 0.80
expdict['percent_val']     = 0

train_params_all[expname] = expdict.copy()


#%% FNN4_128_SingleVar_PIC
"""
FNN4_128_SingleVar_PIC

Old Singlevar Script, but for PiC Data

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PIC"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
#expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['norm']            = True     # Indicate if target was normalized
expdict['stdspace']        = False    # Set to true to standardize predictors in space
expdict['mask']            = True     # True for land-ice masking
expdict["PIC"]             = True     # Use PiControl Data

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']             = 1      # Number of ensemble members to limit to
expdict['ystart']          = 400    # Start year of processed dataset
expdict['yend']            = 2200    # End year of processed dataset
expdict['bbox']            = pparams.bbox #  Bounding box for data

# Label Determination
expdict['quantile']        = False   # Set to True to use quantiles
expdict['thresholds']      = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']           = 264     # Number of samples from each class to train with. None = size of minimum class
expdict['percent_train']      = 0.80    # Training percentage
expdict['percent_val']        = 0.00    # Validation Percentage
expdict['shuffle_class']      = True              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit'] = True             # Set to False to maintain same set for train/test/val split

# Cross Validation Options
expdict['cv_loop']         = False    # Repeat for cross-validation
expdict['cv_offset']       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
expdict['netname']       = "FNN4_128"           # Key for Architecture Hyperparameters
expdict['loss_fn']       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']   = False                # Set to true to change final layer to softmax
expdict['reduceLR']      = False                # Set to true to use LR scheduler
expdict['LRpatience']    = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']    = 3                    # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 20                   # Maximum # of Epochs to train for
expdict['batch_size']    = 16                   # Pairs of predictions
expdict['unfreeze_all']  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer


train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_Rewrite_June 
"""

FNN4_128_SingleVar_Rewrite_June
2023.06.02 Trying to Debug Again 
Forgot what is going on, so I'm going to try again...

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_Rewrite_June" 

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar"].copy()
train_params_all[expname] = expdict.copy()


#%% FNN4_128_SingleVar_Testing
"""

FNN4_128_SingleVar_Testing

Forgot what is going on, so I'm going to try again...

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_Testing" 

# Copy dictionary from above
expdict                    = train_params_all["FNN4_128_SingleVar"].copy()

# Set validation and test size
expdict["percent_train"]   = 0.60
expdict["percent_val"]     = 0.10

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_PaperRun

"""

FNN4_128_SingleVar_PaperRun
2023.06.06 Rerun
Old Singlevar Script, prior to rewrite

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PaperRun"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
#expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['norm']            = True     # Indicate if target was normalized
expdict['stdspace']        = False    # Set to true to standardize predictors in space
expdict['mask']            = True     # True for land-ice masking
expdict["PIC"]             = False    # Use PiControl Data

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']               = 32      # Number of ensemble members to limit to
expdict['ystart']            = 1920    # Start year of processed dataset
expdict['yend']              = 2005    # End year of processed dataset
expdict['bbox']              = pparams.bbox #  Bounding box for data

# Label Determination
expdict['quantile']          = False   # Set to True to use quantiles
expdict['thresholds']        = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']           = 300     # Number of samples from each class to train with. None = size of minimum class
expdict['percent_train']      = 0.90    # Training percentage
expdict['percent_val']        = 0.00    # Validation Percentage
expdict['shuffle_class']      = True              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit'] = True             # Set to False to maintain same set for train/test/val split

# Cross Validation Options
expdict['cv_loop']         = False    # Repeat for cross-validation
expdict['cv_offset']       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
expdict['netname']        = "FNN4_128"           # Key for Architecture Hyperparameters
expdict['loss_fn']        = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']            = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']    = False                # Set to true to change final layer to softmax
expdict['reduceLR']       = False                # Set to true to use LR scheduler
expdict['LRpatience']     = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']     = 5                    # Number of epochs where validation loss increases before stopping
expdict['max_epochs']     = 50                   # Maximum # of Epochs to train for
expdict['batch_size']     = 32                   # Pairs of predictions
expdict['unfreeze_all']   = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_PaperRun_detrended

"""
FNN4_128_SingleVar_PaperRun_detrended

2023.06.06 Rerun
Same as above, but with detrended data

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PaperRun_detrended"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
expdict['detrend']        = 1
train_params_all[expname] = expdict.copy()

#%% FNN6_128_SingleVar_PaperRun

"""
FNN6_128_SingleVar_PaperRun

Run with 6 layers

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN6_128_PaperRun"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
expdict['netname']        = "FNN6_128"
train_params_all[expname] = expdict.copy()

#%% FNN6_128_PaperRun_Detrended

"""
FNN6_128_PaperRun_Detrended

Run with 6 layers

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN6_128_PaperRun_Detrended"

# Copy dictionary from above
expdict                   = train_params_all["FNN6_128_PaperRun"].copy()

# Set custom/new params
expdict['detrend']        = 1
train_params_all[expname] = expdict.copy()


#%% FNN10_128_SingleVar_PaperRun

"""
FNN10_128_SingleVar_PaperRun

Run with 10 layers

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN10_128_PaperRun"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
expdict['netname']        = "FNN10_128"
train_params_all[expname] = expdict.copy()

#%% FNN10_128_SingleVar_PaperRun_Detrended

"""
FNN10_128_SingleVar_PaperRun_Detrended

Run with 10 layers

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN10_128_PaperRun_Detrended"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
expdict['netname']        = "FNN10_128"
expdict['detrend']        = 1
train_params_all[expname] = expdict.copy()

#%% FNN4_128_NoIceMask
"""
FNN4_128_NoIceMask

Run FNN4_128 with no Ice Mask

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PaperRun_NoIceMask"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
expdict['mask'] = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90_lf030_if100.nc"
train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_PaperRun_erule
"""
FNN4_128_SingleVar_PaperRun_erule

Dummy Experiment... for comparing output test metrics
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PaperRun_erule"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
#expdict['mask'] = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90_lf030_if100.nc"
train_params_all[expname] = expdict.copy()
#%% FNN4_128_SingleVar_PaperRun_erule_exp1
"""
FNN4_128_SingleVar_PaperRun_erule_exp1

Dummy Experiment... for comparing output test metrics
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PaperRun_erule_exp1"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
#expdict['mask'] = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90_lf030_if100.nc"
train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_Norm0

"""

FNN4_128_SingleVar_Norm0
2023.06.08 Rerun with new parameters, test normalization
"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_Norm0"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar"].copy()

# Set new parameters
expdict['norm']          = False
expdict['ens']           = 32    # Rest for testing
expdict['percent_train'] = 0.90
expdict['percent_val']   = 0.00 

train_params_all[expname] = expdict.copy()

#%% FNN4_128_SingleVar_Norm1

"""

FNN4_128_SingleVar_Norm1
2023.06.08 Rerun with new parameters, test normalization
Copy above, but with normalization in index

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_Norm1"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_Norm0"].copy()

# Set new parameters
expdict['norm']           = True
train_params_all[expname] = expdict.copy()



#%% CNN2_PaperRun
"""

CNN2_PaperRun
CNN2_LRP Paper Run
Old Singlevar Script, prior to rewrite

"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "CNN2_PaperRun"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
#expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['norm']            = True     # Indicate if target was normalized
expdict['stdspace']        = False    # Set to true to standardize predictors in space
expdict['mask']            = True     # True for land-ice masking
expdict["PIC"]             = False    # Use PiControl Data

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']               = 32      # Number of ensemble members to limit to
expdict['ystart']            = 1920    # Start year of processed dataset
expdict['yend']              = 2005    # End year of processed dataset
expdict['bbox']              = pparams.bbox #  Bounding box for data

# Label Determination
expdict['quantile']          = False   # Set to True to use quantiles
expdict['thresholds']        = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']           = 300     # Number of samples from each class to train with. None = size of minimum class
expdict['percent_train']      = 0.90    # Training percentage
expdict['percent_val']        = 0.00    # Validation Percentage
expdict['shuffle_class']      = True              # Set to True to sample DIFFERENT subsets prior to class subsetting
expdict['shuffle_trainsplit'] = True             # Set to False to maintain same set for train/test/val split

# Cross Validation Options
expdict['cv_loop']         = False    # Repeat for cross-validation
expdict['cv_offset']       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
expdict['netname']        = "CNN2_LRP"           # Key for Architecture Hyperparameters
expdict['loss_fn']        = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']            = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']    = False                # Set to true to change final layer to softmax
expdict['reduceLR']       = False                # Set to true to use LR scheduler
expdict['LRpatience']     = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']     = 5                    # Number of epochs where validation loss increases before stopping
expdict['max_epochs']     = 50                   # Maximum # of Epochs to train for
expdict['batch_size']     = 32                   # Pairs of predictions
expdict['unfreeze_all']   = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer

train_params_all[expname] = expdict.copy()


#%% FNN4_128_SingleVar_PaperRun_stdspace

"""
"FNN4_128_SingleVar_PaperRun_stdspace"
    Copies FNN4_128_SingleVar_PaperRun, but trains networks with spatially-standardized predictors


"""

# # Create Experiment Directory (note that expname = expdir in the original script)
expname                    = "FNN4_128_SingleVar_PaperRun_stdspace"

# Copy dictionary from above
expdict                   = train_params_all["FNN4_128_SingleVar_PaperRun"].copy()

# Set custom/new params
expdict['stdspace']       = True
train_params_all[expname] = expdict.copy()


#%% Some Notes on Parameters and subdivisions

"""

Some Notes on Parameters and subdivisions

Thinking of 3 major categories

---------------------------------
(1) Data Preprocessing Parameters
---------------------------------

Decisions made in the data preprocessing step

    - detrend
    - regrid
    - region of index
    - ice masking
    - predictor (varnames)
    - dataset
    - season
    - lowpass
    - dataset/cmipver

---------------------------------
(2) Subsetting
---------------------------------

Determining how the data is subsetting for training

    - Thresholding Type (quantile, stdev)
    - Test/Train/Val Split Percentage
    - Crossfold Offset
    - # of Ensemble Members to Use
    - Time Period of Trainining (ystart, yend)
    - Bounding Box
    - Training Sample Size (nsamples)
    
    
---------------------------------
(3) Machine Learning Parameters
---------------------------------

Specific Machine Learning Parameters (regularization, training options, etc)

    - epochs
    - assorted hyperparameters
    - early stop
    - architecture
    - batch size
    - 
    
"""
