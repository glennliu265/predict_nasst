#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test Different XAI Methods using Captum and Pytorch-LRP

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

machine = 'Astraeus' # Indicate machine (see module packages section in pparams)

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
varnames            = ["SSH",]       # Names of predictor variables
leads               = np.arange(0,30,5)#[]#np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

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

v                     = 0
l                     = -1

varname               = varnames[v]
predictors            = data[[v],...] # Get selected predictor
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

#%% Select a sample and model

# Select Sample and model
isample  = 609
X_sample = X_torch[[isample],:]
iclass   = int(y_torch[isample,:].detach().numpy()[0])
imodel   = 1

# Load Predictor and Model
X_in       = X_torch[[isample],:]
modweights = modweights_lead[l][imodel]
pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
pmodel.load_state_dict(modweights)
pmodel.eval()

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



#%% Compute Explanations

explanations,xai_names,predout= xai_method(pmodel,X_in,iclass)




#%% Prep For Plotting

classes   = pparams.classes
lon       = load_dict['lon']
lat       = load_dict['lat']
predlvls  = np.arange(-10,11,1)
fsz_title = 16
fsz_ticks = 14

# Make a quick landmask
landmask = predout[0,:,:].copy()
landmask[landmask==0.] = np.nan
landmask[~np.isnan(landmask)] = 1
plt.pcolormesh(landmask)

#%%  Plot for single sample

fig,axs         = plt.subplots(2,4,figsize=(18,9),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

norm = True
ia = 0
for a in range(8):
    
    ax = axs.flatten()[a]
    blabel = [0,0,0,0]
    if a%4 == 0:
        blabel[0] = 1
    if a >=4:
        blabel[-1] = 1
    ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
    
    # Plotting
    plotrel = explanations[a] * landmask
    plotvar = predout.squeeze() * landmask
    cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
    ax.clabel(cl,fontsize=fsz_ticks)
    if norm is False:
        pcm     = ax.pcolormesh(lon,lat,plotrel)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    else:
        normfactor = np.nanmax(np.abs(plotrel))
        plotrel = plotrel / normfactor
        # if imeth == 0: # Test what happens if you do absolute value
        #     plotrel = np.abs(plotrel)
        pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
        label = "max=%.0e" % normfactor
    ax.set_title(xai_names[a],fontsize=fsz_title)
    viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle=r"%s) ",)
        
    ia += 1
if norm:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_title)
    
title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Network %i" % (varnames[v],leads[l],isample,classes[iclass],imodel,)
fnstr    = "%s_lead%02i_sample%03i_class%s_modelALL_norm%i" % (varnames[v],leads[l],isample,iclass,norm)
plt.suptitle("%s"%(title_l1),fontsize=24,y=1.04)
    
savename = "%sXAI_Methods_Test_%s.png" % (figpath,fnstr)
#print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Now repeat for everything... (Just 1 Class)


st = time.time()

explanations,xai_names,predout= xai_method(pmodel,X_torch,iclass)
print("Finished in %.2fs" % (time.time()-st))



#%% 

fig,axs         = plt.subplots(2,4,figsize=(18,9),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

norm = True
ia = 0

normed_maps = []
for a in range(8):
    
    ax = axs.flatten()[a]
    blabel = [0,0,0,0]
    if a%4 == 0:
        blabel[0] = 1
    if a >=4:
        blabel[-1] = 1
    ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
    
    # Plotting
    plotrel = explanations[a,:,:,:].mean(0) * landmask
    #plotvar = predout.mean(0).squeeze() * landmask
    #cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
    ax.clabel(cl,fontsize=fsz_ticks)
    
    if norm is False:
        pcm     = ax.pcolormesh(lon,lat,plotrel)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    else:
        normfactor = np.nanmax(np.abs(plotrel))
        plotrel = plotrel / normfactor
        # if imeth == 0: # Test what happens if you do absolute value
        #     plotrel = np.abs(plotrel)
        pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
        label = "max=%.0e" % normfactor
        
        
    normed_maps.append(plotrel)
    ax.set_title(xai_names[a],fontsize=fsz_title)
    viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle=r"%s) ",)
        
    ia += 1
if norm:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_title)
    
title_l1 = "Predictor %s, Leadtime: %02i, ALL SAMPLES, Network %i" % (varnames[v],leads[l],imodel,)
fnstr    = "%s_lead%02i_sampleALL_class%s_modelALL_norm%i" % (varnames[v],leads[l],iclass,norm)
plt.suptitle("%s"%(title_l1),fontsize=24,y=1.04)
    
savename = "%sXAI_Methods_Test_AllSamples_%s.png" % (figpath,fnstr)
#print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%

nm = normed_maps
nmm = xai_names
diff = nm[5] - nm[6]
plt.pcolormesh(diff,vmin=-.010,vmax=.010,cmap='cmo.balance'),plt.colorbar(),plt.title(nmm[2] + ' - ' + nmm[5])
#%%''å