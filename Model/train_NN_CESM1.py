#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Neural Networks (NN) for CESM1 Large Ensemble Simulations

 - Copied introductory section from NN_Training_Rewrite.py on 2023.03.20
 - Based on NN_test_lead_ann_ImageNet_classification_singlevar.py

Current Structure:
    - Indicate CESM1 training parameters in [train_cesm_parameters.py]
    - Functions are mostly contained in [amvmod.py]
    - Universal Variables + Architectures are in [predict_amv_params.py]
    - Additional helper function from [amv] module [proc] and [viz]

Updated Procedure:
    01) Create Experiment Directory
    02) Load Data, Determine (and make) AMV Classes based on selected thresholds
    03) Loop by Predictor...
        05) Loop by runid (train [nr] networks)...
            06) Preallocate variables and set experiment output name
            07) Loop by Leadtime...
                08) Apply Lead/Lag to predictors+target
                09) Select N samples from each class
                ---- moved to function amvmod.train_NN_lead (10-12)
                10) Perform Train/Test Split, place into dataloaders
                11) Initialize and train the model
                12) Test the model, compute accuracy by class
                ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- -
                13) Save the model and output

Created on Mon Mar 20 21:34:32 2023
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
expdir              = "FNN10_128_PaperRun_Detrended"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames            = ['SSH','SST','SSS','SLP'] #"SST","SSS","SLP","NHFLX",]       # Names of predictor variables
leads               = np.arange(0,26,1)#np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

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
# %% Looping for runid
# ------------------------------------------------------------

# Print Message
print("Running [train_NN_CESM1.py] with the following settings:")
print("\tNetwork Type   : "+ eparams['netname'])
print("\tPredictor(s)   : "+str(varnames))
print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
print("\tRunids         : %i to %i" % (runids[0],runids[-1]))
print("\tMax Epochs     : " + str(eparams['max_epochs']))
print("\tEarly Stop     : " + str(eparams['early_stop']))
print("\t# Ens. Members : " + str(eparams['ens']))
print("\tDetrend        : " + str(eparams['detrend']))
print("\tShuffle        : " + str(eparams['shuffle_trainsplit']))

# ------------------------
# 04. Loop by predictor...
# ------------------------
for v,varname in enumerate(varnames): 
    vt = time.time()
    predictors = data[[v],...] # Get selected predictor
    
    # --------------------
    # 05. Loop by runid...
    # --------------------
    for nr,runid in enumerate(runids):
        rt = time.time()
        
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
        
        # -----------------------
        # 07. Loop by Leadtime...
        # -----------------------
        for l,lead in enumerate(leads):
            lt = time.time()
            
            # Set names for intermediate saving, based on leadtime
            if (lead == leads[-1]) and (len(leads)>1): # Output all files together
                outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
            else: # Output individual lead times while training
                outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
            
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
            
            # # --------------------------------------------------------------------------------
            # # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
            # # --------------------------------------------------------------------------------
            output = am.train_NN_lead(X,y_class,eparams,pparams,debug=debug,checkgpu=checkgpu)
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
            
            # ------------------------------
            # 13. Save the model and metrics
            # ------------------------------
            if savemodel:
                modout = "../../CESM_data/%s/Models/%s_%s_lead%02i_classify.pt" %(expdir,expname,varname,lead)
                torch.save(model.state_dict(),modout)
            
            # Save Metrics
            savename = "../../CESM_data/"+expdir+"/"+"Metrics"+outname
            np.savez(savename,**{
                      'train_loss'     : train_loss_grid,
                      'test_loss'      : test_loss_grid,
                      'val_loss'       : val_loss_grid,
                      'train_acc'      : train_acc_grid,
                      'test_acc'       : test_acc_grid,
                      'val_acc'        : val_acc_grid,
                      'total_acc'      : total_acc,
                      'acc_by_class'   : acc_by_class,
                      'yvalpred'       : yvalpred,
                      'yvallabels'     : yvallabels,
                      'sampled_idx'    : sampled_idx,
                      'thresholds_all' : thresholds_all,
                      'exp_params'     : eparams,
                      'sample_sizes'   : sample_sizes,
                      'runtime'        : time.time() - lt
                      }
                      )
            
            # Clear some memory
            del model
            torch.cuda.empty_cache()  # Save some memory
            
            print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
            # End Lead Loop >>>
        print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
        # End Runid Loop >>>
    print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
    # End Predictor Loop >>>
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
             


