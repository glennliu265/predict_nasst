#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test Different XAI Methods using Captum and Pytorch-LRP, Looping Version


For a given Predictor + Leadtime + Sample
Also performs samplewise composites...

Copied Introductory Section of [test_LRP_incorrect]

Created on Wed Nov 15 20:55:00 2023

@author: gliu
"""


import numpy as np
import sys
import glob

import xarray as xr

import torch
from torch import nn

from tqdm import tqdm
import time
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import captum

from torch.utils.data import DataLoader, TensorDataset,Dataset
#%% Load custom packages and setup parameters

machine = 'stormtrack' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd = os.getcwd()
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
from amv import proc,viz

# Import LRP package
lrp_path = pparams.machine_paths[machine]['lrp_path']
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

# Load ML architecture information
nn_param_dict      = pparams.nn_param_dict

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths
figpath = pparams.figpath

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Processing Options
even_sample         = False
standardize_input   = False # Set to True to standardize variance at each point
calc_lrp            = True # Set to True to calculate relevance composites

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
varnames            = ["SST","SSS","SLP","SSH",],       # Names of predictor variables
leads               = [0,25]#np.arange(0,30,5)#[]#np.arange(0,26,1)    # Prediction Leadtimes
runids              = [0,]#np.arange(0,100,1)    # Which runs to do
choose_class        = [0,2]


# LRP Parameters
innexp              = 2
innmethod           ='b-rule'
innbeta             = 0.1
innepsilon          = 1e-2

# Other toggles
save_all_relevances = False                # True to save all relevances (~33G per file...)
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = False                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# -----------------------------------
# %% Get some other needed parameters
# -----------------------------------

# Ensemble members
ens_all        = np.arange(0,42)
ens_train_val  = ens_all[:eparams['ens']]
ens_test       = ens_all[eparams['ens']:]
nens_test      = len(ens_test)

#%% Try Different LRP Methods


def convert_output(rel_torch,nlat=69,nlon=65):
    nsamples = rel_torch.shape[0]
    return rel_torch.reshape(nsamples,nlat,nlon).detach().numpy().squeeze()

def xai_method(pmodel,X_in,iclass):
    
    predout      = convert_output(X_in)
    
    explanations = []
    xai_names    = []
    
    # 1. LRP Alpha-Beta
    inn_model = InnvestigateModel(pmodel, lrp_exponent=2,
                                              method='b-rule',
                                              beta=0.1,
                                              epsilon=1e-6)
    model_prediction, relout = inn_model.innvestigate(in_tensor=X_in)
    #relout  = convert_output(sample_relevances)
    explanations.append(relout)
    xai_names.append(r"$LRP_{\alpha \beta}$, $\beta=0.1$")
    
    # 2. LRP Z
    lrp    = captum.attr.LRP(pmodel)
    relout = lrp.attribute(X_in,target=iclass)
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append(r"$LRP_{z}$")
    
    # 3. Integrated Gradients
    ig           =  captum.attr.IntegratedGradients(pmodel)
    relout = ig.attribute(X_in,target=iclass)
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append("Integrated Gradients")
    
    # 4. Integrated Gradients + Noise Tunnel
    noise_tunnel    = captum.attr.NoiseTunnel(ig)
    relout = noise_tunnel.attribute(X_in,nt_samples=10,nt_type='smoothgrad_sq',target=iclass,
                                             )
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append("Integrated Gradients + Noise Tunnel")
    
    # 5. Occlusion
    occ_stride = 3 # Set stride length
    occ_window = 5 # Set window size
    occlusion  = captum.attr.Occlusion(pmodel)
    relout     = occlusion.attribute(X_in,target=iclass,
                                strides = (occ_stride,),
                                sliding_window_shapes=(occ_window,),
                                baselines=0)
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append("Occlusion (Stride = %i, Win = %i)" % (occ_stride,occ_window))
    
    # 6. GradientShap
    nsamples_gradshap=50
    stdevs_gradshap=1e-4
    gradshap           = captum.attr.GradientShap(pmodel)
    rand_img_dist      = torch.cat([X_in * 0, X_in * 1]) # Set baseline images
    relout             = gradshap.attribute(X_in,target=iclass,
                                            n_samples=nsamples_gradshap,
                                            stdevs=stdevs_gradshap,
                                            baselines=rand_img_dist,)
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append("GradientShap")
    
    # 7. DeepLift
    dl           = captum.attr.DeepLift(pmodel)
    relout = dl.attribute(X_in,target=iclass)
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append("DeepLift")
    
    # 8. Saliency Maps
    saliency = captum.attr.Saliency(pmodel)
    relout    = saliency.attribute(X_in, target=iclass)
    #relout = convert_output(relout)
    explanations.append(relout)
    xai_names.append("Saliency Maps")
    
    
    explanations = [convert_output(tensor) for tensor in explanations]
    explanations = np.array(explanations)
    
    return explanations,xai_names,predout


# ============================================================
#%% Load the data 
# ============================================================
# Copied segment from train_NN_CESM1.py

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True,load_all_ens=False,return_test_set=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']

# Pick just the testing set
data                           = load_dict['data_test']#data[:,ens_test,...]
target_class                   = load_dict['target_class_test']#target_class[ens_test,:]

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)

# --------------------------------------------------------
#%% Option to standardize input to test effect of variance
# --------------------------------------------------------

if standardize_input:
    # Compute standardizing factor (and save)
    std_vars = np.std(data,(1,2)) # [variable x lat x lon]
    for v in range(nchannels):
        savename = "%s%s/Metrics/%s_standardizing_factor_ens%02ito%02i.npy" % (datpath,expdir,varnames[v],ens_test[0],ens_test[-1])
        np.save(savename,std_vars[v,:,:])
    # Apply standardization
    data = data / std_vars[:,None,None,:,:] 
    data[np.isnan(data)] = 0
    std_vars_after = np.std(data,(1,2))
    check =  np.all(np.nanmax(np.abs(std_vars_after)) < 2)
    assert check, "Standardized values are not below 2!"
        


#%% Choose variable and leadtime, preprocess

lon       = load_dict['lon']
lat       = load_dict['lat']


for v in range(len(varnames)):
    varname               = varnames[v]
    predictors            = data[[v],...] # Get selected predictor
    for l in range(len(leads)):
        
        lead                  = leads[l]

        # ===================================
        # I. Data Prep
        # ===================================
        
        # IA. Apply lead/lag to data
        # --------------------------
        X,y_class             = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)
        
        # ----------------------
        # IB. Select samples
        # ----------------------
        _,class_count = am.count_samples(None,y_class)
        if even_sample:
            eparams['nsamples'] = int(np.min(class_count))
            print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
            y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
        
        # ----------------------
        # IC. Flatten inputs for FNN
        # ----------------------
        if "FNN" in eparams['netname']:
            ndat,nchannels,nlat,nlon = X.shape
            inputsize                = nchannels*nlat*nlon
            outsize                  = nclasses
            X_in                     = X.reshape(ndat,inputsize)
        else:
            X_in = X
        
        # -----------------------------
        # ID. Place data into a data loader
        # -----------------------------
        # Convert to Tensors
        X_torch = torch.from_numpy(X_in.astype(np.float32))
        y_torch = torch.from_numpy(y_class.astype(np.compat.long))
        
        # Put into pytorch dataloaders
        test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
        
        # ================================
        #% Get Models
        # ================================
        # Get the model weights [lead][run]
        modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
        nmodels = len(modweights_lead[0])
        
        
        # Loop by model
        for imodel in range(nmodels):
            
            # Load model and weights
            modweights = modweights_lead[l][imodel]
            pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
            pmodel.load_state_dict(modweights)
            pmodel.eval()
            
            for ic in range(len(choose_class)):
                
                iclass = choose_class[ic]
                
                st = time.time()
                explanations,xai_names,predout= xai_method(pmodel,X_torch,iclass)
                
                
                fname    = "XAI_Output_%s_lead%02i_model%02i_class%i.npz" % (varname,lead,imodel,iclass)
                savename = "%s%s/Metrics/%s" % (datpath,expdir,fname)
                np.savez(savename,**{
                    'explanations': explanations, # [Method x Sample x Lat x Lon]
                    'xai_names'   : xai_names,
                    'targets'     : y_torch,
                    'preditor'    : X_torch,
                    'lon'         : lon,
                    'lat'         : lat,
                    },allow_pickle=True)
                print("Finished in %.2fs" % (time.time()-st))
            
            del pmodel