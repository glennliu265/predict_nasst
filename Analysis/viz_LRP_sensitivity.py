#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Output of test_LRP_sensitivity
----------------------------------------

Created on Mon Nov  6 13:12:18 2023

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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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


figpath             = pparams.figpath

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

#%% Load the files

varname = varnames[0]
betas   = [0,0.1,0.3,0.5,0.7,0.9,1]


fns    = []
ds_all = []
nbetas = len(betas)
for bb in range(nbetas):
    innbeta = betas[bb]
    #expdir_new = expdir + "_beta%.1f/"
    dirnew =  "beta%.1f/" % innbeta
    diroutnew = "%s%s/Metrics/%s" % (datpath,expdir,dirnew)
    proc.makedir(diroutnew)
    outname    = "%s/Test_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (diroutnew,dataset_name,varname,even_sample)
    fns.append(outname)
    
    ds_all.append(xr.open_dataset(outname).load())
    
#%% Make for individual leadtime

classname = "NASST+"
lead      = 15
bboxplot  = [-80,0,0,62]
#vlms      = [-1e-3,1e-3]
vlms      = [-1,1]
normalize = True


runid     = 99

fig,axs = plt.subplots(1,nbetas,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(16,4))

for bb in range(nbetas):
    ax = axs[bb]
    
    blabel = [0,0,0,1]
    if bb == 0:
        blabel[0] = 1
    
    viz.add_coast_grid(ax,bbox,fill_color="gray",blabels=blabel)
    
    if runid == "ALL":
        runstr  = "MEAN"
        plotvar = ds_all[bb].relevance_composites.sel({'class':classname,'lead':lead}).mean('runid')
    else:
        runstr = "%03i" % runid
        plotvar = ds_all[bb].relevance_composites.sel({'class':classname,'lead':lead,'runid':runid})
    if normalize:
        plotvar = plotvar / np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1])
    ax.set_title(r"$ \beta$ = %.2f" % (betas[bb]),fontsize=16)
    
plt.suptitle("Relevance Composites for %s, Predictor: %s, Lead: %i years, Runid: %s" % (classname,varname,lead,runstr),fontsize=20,y=0.84)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.006,pad=0.01)


savename = "%s%s_RelevanceCompositeAll_%s_%s_runid%s_lead%02i.png" % (figpath,expdir,varname,classname,runstr,lead)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Make revision figure

classname = "NASST+"
lead      = 15
bboxplot  = [-80,0,0,62]
#vlms      = [-1e-3,1e-3]
vlms      = [-1,1]
normalize = True
no_sp_label = False
fsz_axlbl   = 14

leadschoose =[0,25]
runid     = "ALL"

fig,axs = plt.subplots(2,nbetas,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(16,4))
ia = 0
for ll in range(2):
    lead = leadschoose[ll]
    for bb in range(nbetas):
        ax = axs[ll,bb]
        
        blabel = [0,0,0,0]
        if bb == 0:
            blabel[0] = 1
            ax.text(-0.25, 0.55, "Lead %02i years" % (lead), va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=13)
        if ll == 1:
            blabel[-1] = 1
        else:
            ax.set_title(r"$ \beta$ = %.2f" % (betas[bb]),fontsize=16)
        
        viz.add_coast_grid(ax,bbox,fill_color="gray",blabels=blabel)
        
        if no_sp_label is False:
            ax = viz.label_sp(ia,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
            ia += 1
        if runid == "ALL":
            runstr  = "MEAN"
            plotvar = ds_all[bb].relevance_composites.sel({'class':classname,'lead':lead}).mean('runid')
        else:
            runstr = "%03i" % runid
            plotvar = ds_all[bb].relevance_composites.sel({'class':classname,'lead':lead,'runid':runid})
        if normalize:
            plotvar = plotvar / np.nanmax(np.abs(plotvar))
        
        landmask = plotvar.values.copy()
        landmask[plotvar==0] = np.nan
        landmask[~np.isnan(landmask)] = 1
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * landmask,cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1])
    
    #plt.suptitle("Relevance Composites for %s, Predictor: %s, Lead: %i years, Runid: %s" % (classname,varname,lead,runstr),fontsize=20,y=0.84)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.006,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=14)
savename = "%s%s_RelevanceCompositeAll_%s_%s_runid%s_leads0_25.png" % (figpath,expdir,varname,classname,runstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%%

#%%