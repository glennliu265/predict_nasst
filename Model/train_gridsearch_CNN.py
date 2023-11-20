#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

----------------------------------------------
Hyperparameter Gridsearch Test Script for CNNs
----------------------------------------------

Test different filter sizes, stride sizes, pool sizes, and number of
convolutional + poling layers.

Output will be saved to "Param_Testing" within the corresponding experiment folder

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
varnames            = ['SST',] # Names of predictor variables (Currently only supports 1!!)
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

# <End copy from train_NN_CESM1.py>  ==========================================

#%% Load base experiment dictionary -------------------------------------------

nnparams_original = pparams.nn_param_dict[eparams['netname']].copy()
eparams_original  = eparams.copy()

#%% Define Functions

# This has been moved to amvmod
def build_simplecnn_fromdict(param_dict,num_classes,nlat=224,nlon=224,num_inchannels=3):
    
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


#%%  A Simple Test to set up combinations.
# Discard combinations with too many layers (i.e. pooling over 1-pixel feature maps, etc.)
"""
Let's set up a simpler test, where:
    (1) strides + filter sizes are symmetric, and 
    (2) the pool and filter sizes are also the same...
    
Option for future work:
    Develop more extensive hyperparameter tests
    where the filter vs pool sizes are different, or where the strides
    are symmetric...
"""

# Set some dimensions
in_channels = 1 # Input Channels, Let's use single predictors first
num_classes = 3 # Number of output classes (+) or (-) NASST

# Set the Test Values
nchannels     = ([32,],[32,64],[32,64,128]) 
filtersizes   = ([2,2],[3,3],[4,4])
filterstrides = (1,)#(2,3,4)
poolsizes     = copy.deepcopy(filtersizes)
poolstrides   = copy.deepcopy(filterstrides)

# Set up testing dictionary (simple test)
test_param_names   = ["nchannels","filtersizes","filterstrides",]
test_param_values  = [nchannels  ,filtersizes  ,filterstrides  ,]
test_params        = dict(zip(test_param_names,test_param_values))
param_combinations = list(itertools.product(*test_param_values))
ncombos            = len(param_combinations)

# Set up dictionaries and build CNNs ---------
all_cnns    = []
expnames    = []
param_dicts = []
fcsize_fin = []
for n in range(ncombos):
    
    # Create Parameter Dictionary
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
    # Build CNN
    cnnmod = build_simplecnn_fromdict(cnn_param_dict,num_classes,
                        nlat=nlat,nlon=nlon,num_inchannels=in_channels)
    
    # Check Final Layer Size
    fcfinal = am.calc_layerdims(nlon,nlat,
                                cnn_param_dict['filtersizes'],
                                cnn_param_dict['filterstrides'],
                                cnn_param_dict['poolsizes'],
                                cnn_param_dict['poolstrides'],
                                cnn_param_dict['nchannels'],
                                )
    fcsize_fin.append(fcfinal)

        
    
    # Set Name
    expname = "nlayers%i_filtersize%i_stride%i" % (nlayers,filtersize[0],stridesize)
    if fcfinal == 0:
        print("Combo is not valid: %s" % (expname))
        
    # Append Everything
    all_cnns.append(cnnmod)
    expnames.append(expname)
    param_dicts.append(cnn_param_dict)

#%% Now place this into a training loop
# Output will be saved to "Param_Testing" within the corresponding experiment folder
# Set some variables needed
varname    = varnames[0]
predictors = data[[0],...] # Get selected predictor


for nc in range(ncombos):
    
    # Get Information
    pcomb           = param_combinations[nc]
    pdict           = param_dicts[nc]
    pname           = expnames[nc]
    fcsize          = fcsize_fin[nc]
    if fcsize == 0:
        print("%s is not valid, skipping." % (pname))
        continue
    ct              = time.time()
    
    # Copy dictionaries to use for this particular combo
    combo_expdict   = eparams_original.copy()
    combo_paramdict = nnparams_original.copy()
    
    # Make the experiment string and prepare the folder
    expstr = pname
    print(expstr)
    outdir = "%s%s/ParamTesting/%s/" % (pparams.datpath,expdir,expstr)
    proc.makedir(outdir)
    proc.makedir(outdir + "Metrics/")
    proc.makedir(outdir + "Models/")
    
    # Reassign Parameters
    combo_expdict['netname'] = "simplecnn_paramdict" # Assign key to make custom simplecnn
    combo_paramdict.update(pdict) # Merge Original and new Dictionary
    pparams.nn_param_dict    = {combo_expdict['netname']:combo_paramdict}
    eparams                  = combo_expdict.copy()
    
    # Now run the Loop 
    # <START copy from train_NN_CESM1.py>  ====================================
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
                modout = "%s/Models/%s_lead%02i_classify.pt" %(outdir,varname,lead)
                torch.save(model.state_dict(),modout)
            
            # Save Metrics
            savename = outdir+"/"+"Metrics"+outname
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
                      }
                      )
            
            # Clear some memory
            del model
            torch.cuda.empty_cache()  # Save some memory
            
            #print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
            # End Lead Loop >>>
        #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
        # End Runid Loop >>>
    print("Completed combination %s in %.2fs" % (expstr,time.time()-ct))
    # <End Parameter Combination Loop>
    
    
    
    
    
    
    

