#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to Test the Rewritten train_ResNet script and to evaluate the effect of early stopping on training accuracy

Copied train_NN_CESM1.py on 2023 . 11 . 01


Created on Wed Nov  1 12:23:31 2023

@author: gliu
"""

import sys
import numpy as np
import os
import time
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset

import matplotlib.pyplot as plt
import copy

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
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Make new output path
outpath             = "../../CESM_data/%s/EpochTesting/"  % expdir
proc.makedir(outpath)

# Set some looping parameters and toggles
varnames            = ['SST','SSH']#"SST","SSS","SLP","NHFLX",]       # Names of predictor variables
leads               = np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(1)    # Which runs to do

# Other toggles
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = True                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# ============================================================
# End User Edits ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ============================================================

# ------------------------------------------------------------
# %% 01. Check for existence of experiment directory and create it
# ------------------------------------------------------------
allstart = time.time()

proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)

# Check if there is gpu
if checkgpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

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
# ----------------------------------------------------
# %% Retrieve a consistent sample if the option is set
# ----------------------------------------------------

if eparams["shuffle_trainsplit"] is False:
    print("Pre-selecting indices for consistency")
    output_sample=am.consistent_sample(data,target_class,leads,eparams['nsamples'],leadmax=leads.max(),
                          nens=None,ntime=None,
                          shuffle_class=eparams['shuffle_class'],debug=False)
    
    target_indices,target_refids,predictor_indices,predictor_refids = output_sample
else:
    print("Indices will be shuffled for each training iteration")
    target_indices     = None
    predictor_indices  = None
    target_refids      = None
    predictor_refids   = None

"""
Output

shuffidx_target  = [nsamples*nclasses,]        - Indices of target
predictor_refids = [nlead][nsamples*nclasses,] - Indices of predictor at each leadtime

tref --> array of the target years
predictor_refids --> array of the predictor refids
"""

# ------------------------------------------------------------
# %% Prepare the Data for Input
# ------------------------------------------------------------

# ------------------------
# 04. Loop by predictor...
# ------------------------
v = 0
varname = varnames[v]
vt = time.time()
predictors = data[[v],...] # Get selected predictor

# --------------------
# 05. Loop by runid...
# --------------------
nr = 0
runid = runids[nr]
rt = time.time()

# ---------------------
# 07. Loop by Leadtime
# ---------------------
l = 25
lead = leads[l]

# ####################### Probably Irrelevant #################################
# ---------------------------------------
# 06. Set experiment name and preallocate
# ---------------------------------------
# Set experiment save name (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
expname = ("AMVClass%i_%s_nepoch%02i_" \
           "nens%02i_maxlead%02i_"\
           "detrend%i_run%02i_"\
           "quant%i_res%s" % (nclasses,eparams['netname'],eparams['max_epochs'],
                                 eparams['ens'],leads[-1],eparams['detrend'],runid,
                                 eparams['quantile'],eparams['regrid']))

    
# Set names for intermediate saving, based on leadtime
if (lead == leads[-1]) and (len(leads)>1): # Output all files together
    outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
else: # Output individual lead times while training
    outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
# #############################################################################



if target_indices is None:
    # --------------------------
    # 08. Apply lead/lag to data
    # --------------------------
    # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
    X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=eparams['ens'],tstep=ntime)
    
    # ----------------------
    # 09. Select samples
    # ----------------------
    if (eparams['shuffle_trainsplit'] is True) or (l == 0):
        if eparams['nsamples'] is None: # Default: nsamples = smallest class
            threscount = np.zeros(nclasses)
            for t in range(nclasses):
                threscount[t] = len(np.where(y_class==t)[0])
            eparams['nsamples'] = int(np.min(threscount))
            print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
        y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
    
    else:
        
        print("Select the pre-sampled indices")
        shuffidx = sampled_idx[l-1]
        y_class  = y_class[shuffidx,...]
        X        = X[shuffidx,...]
        am.count_samples(eparams['nsamples'],y_class)
    shuffidx = shuffidx.astype(int)
else:
    print("Using preselected indices")
    pred_indices = predictor_indices[l]
    nchan        = predictors.shape[0]
    y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
    X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
    X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
    shuffidx     = target_indices    

# -----------------------------------------------------------------------------
#%% Experiment 1 (50 epoch (with early stopping), 50 Epochs , 100 Epochs)
# -----------------------------------------------------------------------------
# Preallocate Evaluation Metrics...
train_loss_grid = [] #np.zeros((max_epochs,nlead))
test_loss_grid  = [] #np.zeros((max_epochs,nlead))
val_loss_grid   = [] 

train_acc_grid  = []
test_acc_grid   = [] # This is total_acc
val_acc_grid    = []

acc_by_class    = []
total_acc       = []
yvalpred        = []
yvallabels      = []
sampled_idx     = []
thresholds_all  = []
sample_sizes    = []

for c in range(3):
    
    eparams_in = copy.deepcopy(eparams)
    if c == 0:
        eparams_in['early_stop'] - eparams['early_stop']
    elif c == 1:
        eparams_in['early_stop'] = False
    elif c == 2:
        eparams_in['early_stop'] = False
        eparams_in['max_epochs'] = 100
    
    # # --------------------------------------------------------------------------------
    # # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
    # # --------------------------------------------------------------------------------
    output = am.train_NN_lead(X,y_class,eparams_in,pparams,debug=False,checkgpu=checkgpu,verbose=False)
    model,trainloss,valloss,testloss,trainacc,valacc,testacc,y_predicted,y_actual,class_acc,lead_acc = output
    
    # Append outputs for the leadtime
    train_loss_grid.append(trainloss)
    val_loss_grid.append(valloss)
    test_loss_grid.append(testloss)
    
    train_acc_grid.append(trainacc)
    val_acc_grid.append(valacc)
    test_acc_grid.append(testacc)
    
    acc_by_class.append(class_acc)
    total_acc.append(lead_acc)
    yvalpred.append(y_predicted)
    yvallabels.append(y_actual)
    sampled_idx.append(shuffidx) # Save the sample indices
    sample_sizes.append(eparams['nsamples'])

#%% Visualize Train/Test/Val Loss and Accuracies

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,4))

ax = axs[0]
ax.set_ylabel("Loss")
ax.plot(trainloss,color='dodgerblue',label="Train",marker="d",markersize=2)
ax.plot(valloss,color='magenta',label="Val",marker="d",markersize=2)
ax.plot(testloss,color='orange',label="Test",marker="d",markersize=2)
ax.axvline(np.nanargmin(testloss),label="Test Min (%i)" % (np.nanargmin(testloss)+1),color="orange",ls='dashed')
ax.grid(True,ls='dashed')
ax.legend()

ax = axs[1]
ax.set_ylabel("Acc")
ax.plot(trainacc,color='dodgerblue',label="Train",marker=".",markersize=5)
ax.plot(valacc,color='magenta',label="Val",marker=".",markersize=5)
ax.plot(testacc,color='orange',label="Test",marker=".",markersize=5)
ax.axvline(np.nanargmin(testacc),label="Test Min (%i)" % (np.nanargmin(testacc)+1),color="orange",ls='dashed')
ax.grid(True,ls='dashed')
ax.set_ylim([0,1.1])
ax.legend()
ax.set_ylabel("Epochs")

# -----------------------------------------------------------------------------
#%% Experiment (2): How many models do "early stopping" before 50 epochs
# -----------------------------------------------------------------------------

niter = 100

# Preallocate Evaluation Metrics...
train_loss_grid = [] #np.zeros((max_epochs,nlead))
test_loss_grid  = [] #np.zeros((max_epochs,nlead))
val_loss_grid   = [] 

train_acc_grid  = []
test_acc_grid   = [] # This is total_acc
val_acc_grid    = []

acc_by_class    = []
total_acc       = []
yvalpred        = []
yvallabels      = []
sampled_idx     = []
thresholds_all  = []
sample_sizes    = []

for c in range(niter):
    
    eparams_in = copy.deepcopy(eparams)
    
    # # --------------------------------------------------------------------------------
    # # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
    # # --------------------------------------------------------------------------------
    output = am.train_NN_lead(X,y_class,eparams_in,pparams,debug=False,checkgpu=checkgpu,verbose=False)
    model,trainloss,valloss,testloss,trainacc,valacc,testacc,y_predicted,y_actual,class_acc,lead_acc = output
    
    # Append outputs for the leadtime
    train_loss_grid.append(trainloss)
    val_loss_grid.append(valloss)
    test_loss_grid.append(testloss)
    
    train_acc_grid.append(trainacc)
    val_acc_grid.append(valacc)
    test_acc_grid.append(testacc)
    
    acc_by_class.append(class_acc)
    total_acc.append(lead_acc)
    yvalpred.append(y_predicted)
    yvallabels.append(y_actual)
    sampled_idx.append(shuffidx) # Save the sample indices
    sample_sizes.append(eparams['nsamples'])


savename = "%s%s_EpochTesting_EarlyStop_niter%i_%s_lead%02i.npz" % (outpath,expdir,niter,varname,leads[l])
savedict = {'trainloss' : train_loss_grid,
            'testloss'  : test_loss_grid,
            'valloss'   : val_loss_grid,
            'trainacc'  : train_acc_grid,
            'testacc'   : test_acc_grid,
            'valacc'    : val_acc_grid,
            "acc_by_class" : acc_by_class}
np.savez(savename,**savedict,allow_pickle=True)


#%% Load dict from above and do some calculations

# Load some essential variables
savename = "%s%s_EpochTesting_EarlyStop_niter%i_%s_lead%02i.npz" % (outpath,expdir,niter,varname,leads[l])
ld = np.load(savename,allow_pickle=True)
trainloss = ld['trainloss']
testloss  = ld['testloss']

# For each loop, compute when early stopping happened along epoch dimension (1)
stop_epoch     = np.argmax(np.isnan(trainloss),1)
percent_past50 = len(np.where(stop_epoch == 0)[0]) / len(stop_epoch)



# %% Visualize the Testing and Training Loss of All Networks




fig,axs = plt.subplots(2,1,sharex=True)

for a in range(2):
    
    ax = axs[a]
    
    if a == 0:
        plotloss = trainloss
        md       = "Train"
    elif a == 1:
        plotloss = testloss
        md       = "Validation"
        ax.set_xlabel("Epochs")
    
    for n in range(niter):
        ax.plot(plotloss[n],alpha=0.5)
        ax.set_ylabel("%s Loss" % (md))
        
        
#%% Compute percentage of samples doing early stopping


        
        
    

#%% Loop Version of above (Run for 2 variables at each leadtime)

varnames = ["SST","SSH"]
leads    = np.arange(0,26,1)
niter    = 100

for v in range(len(varnames)):
    varname    = varnames[v]
    predictors = data[[v],...] # Get selected predictor
    
    for l in range(len(leads)):
        lead = leads[l]
        
        
        if target_indices is None:
            # --------------------------
            # 08. Apply lead/lag to data
            # --------------------------
            # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
            X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=eparams['ens'],tstep=ntime)
            
            # ----------------------
            # 09. Select samples
            # ----------------------
            if (eparams['shuffle_trainsplit'] is True) or (l == 0):
                if eparams['nsamples'] is None: # Default: nsamples = smallest class
                    threscount = np.zeros(nclasses)
                    for t in range(nclasses):
                        threscount[t] = len(np.where(y_class==t)[0])
                    eparams['nsamples'] = int(np.min(threscount))
                    print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
                y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
            
            else:
                
                print("Select the pre-sampled indices")
                shuffidx = sampled_idx[l-1]
                y_class  = y_class[shuffidx,...]
                X        = X[shuffidx,...]
                am.count_samples(eparams['nsamples'],y_class)
            shuffidx = shuffidx.astype(int)
        else:
            print("Using preselected indices")
            pred_indices = predictor_indices[l]
            nchan        = predictors.shape[0]
            y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
            X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
            X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
            shuffidx     = target_indices    
            
            
            
        # Start Looop
        # Preallocate Evaluation Metrics...
        train_loss_grid = [] #np.zeros((max_epochs,nlead))
        test_loss_grid  = [] #np.zeros((max_epochs,nlead))
        val_loss_grid   = [] 

        train_acc_grid  = []
        test_acc_grid   = [] # This is total_acc
        val_acc_grid    = []

        acc_by_class    = []
        total_acc       = []
        yvalpred        = []
        yvallabels      = []
        sampled_idx     = []
        thresholds_all  = []
        sample_sizes    = []
        
        for c in range(niter):
            
            eparams_in = copy.deepcopy(eparams)
            
            # # --------------------------------------------------------------------------------
            # # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
            # # --------------------------------------------------------------------------------
            output = am.train_NN_lead(X,y_class,eparams_in,pparams,debug=False,checkgpu=checkgpu,verbose=False)
            model,trainloss,valloss,testloss,trainacc,valacc,testacc,y_predicted,y_actual,class_acc,lead_acc = output
            
            # Append outputs for the leadtime
            train_loss_grid.append(trainloss)
            val_loss_grid.append(valloss)
            test_loss_grid.append(testloss)
            
            train_acc_grid.append(trainacc)
            val_acc_grid.append(valacc)
            test_acc_grid.append(testacc)
            
            acc_by_class.append(class_acc)
            total_acc.append(lead_acc)
            yvalpred.append(y_predicted)
            yvallabels.append(y_actual)
            sampled_idx.append(shuffidx) # Save the sample indices
            sample_sizes.append(eparams['nsamples'])
            
            
        savename = "%s%s_EpochTesting_EarlyStop_niter%i_%s_lead%02i.npz" % (outpath,expdir,niter,varname,leads[l])
        savedict = {'trainloss' : train_loss_grid,
                    'testloss'  : test_loss_grid,
                    'valloss'   : val_loss_grid,
                    'trainacc'  : train_acc_grid,
                    'testacc'   : test_acc_grid,
                    'valacc'    : val_acc_grid,
                    "acc_by_class" : acc_by_class}
        np.savez(savename,**savedict,allow_pickle=True)
        
        
        







