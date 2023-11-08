#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Test Metrics Experiements (Acc. by Class/Predictor and Relevance Maps)

Copied Manuscript_Figures.ipynb

Created on Wed Nov  8 11:09:59 2023

@author: gliu
"""


# Import Packages

import sys
import numpy as np
import os
import time
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob
import copy
import xarray as xr


#%% User Edits
darkmode        = False

# Other paths
datpath          = "../../results_manuscript/"
figpath          = "../../results_presentation/"
#datpath          = "../../results_presentation/"  

# Custom Module Packages
amvpath          = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/"
pytorch_lrp_path = "/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/"


#%% Import custom packages and other modules

# Import general utilities from amv module
sys.path.append(amvpath)
import proc,viz

# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

# LRP Methods
sys.path.append(pytorch_lrp_path)
from innvestigator import InnvestigateModel


#%% Load some additional variables
bbox         = pparams.bbox
class_colors = pparams.class_colors

# Set up plotting parameters
if darkmode is True:
    plt.style.use('dark_background')
    transparent= True
    dfcol      = "w"
    dfcol_r    = "k"
else:
    plt.style.use('default')
    transparent= False
    dfcol      = "k"
    dfcol_r    = "w"

# -----------------------------------------------------------------------------
#%% Part 1: Compare Test Accuracy by Class
# -----------------------------------------------------------------------------


# User Edits
# ----------------------------------

# Labels and Experiments to compare
#expdirs         = ("FNN4_128_Singlevar_PaperRun","FNN4_128_Singlevar_PaperRun_detrended")
#expdirs_long    = ("Forced","Unforced")

expdirs         = ("FNN4_128_Singlevar_PaperRun","FNN6_128_PaperRun")
expdirs_long    = ("FNN4","FNN6")

comparename     = "FNN4_v_FNN6"

# Other Settings
chance_baseline = (0.33,)*3

# Variables and other settings
varnames        = ('SST', 'SSS', 'SLP', 'SSH')
varcolors       = ("r","violet","gold","dodgerblue",)
varmarker       = ("o","d","x","^")
classes_new     = ("NASST+","Neutral","NASST-")
leads           = np.arange(0,26,1)

#%%
## Load Results
# ----------------------------------

# Load the FNN Accuracy By Predictor
# ----------------------------------
alloutputs = []
for expdir in expdirs:
    output_byvar = []
    for v in varnames:
        fn              = "%s%s/Metrics/Test_Metrics/Test_Metrics_CESM1_%s_evensample0_accuracy_predictions.npz" % (datpath,expdir,v)
        npz             = np.load(fn,allow_pickle=True)
        expdict         = proc.npz_to_dict(npz)
        output_byvar.append(expdict)
    alloutputs.append(output_byvar)

# Load the Persistence Baseline
# ----------------------------------
persaccclass = []
persacctotal = []
persleads   = []
for detrend in [False,True]:
    fn_baseline      = "%s/Baselines/persistence_baseline_CESM1_NAT_detrend%i_quantile0_nsamplesNone_repeat1.npz" % (datpath,detrend,)
    ldp              = np.load(fn_baseline)
    pers_class_acc   = ldp['acc_by_class']
    pers_total_acc   = ldp['total_acc']
    pers_leads       = ldp['leads']
    persaccclass.append(pers_class_acc)
    persacctotal.append(pers_total_acc)
    persleads.append(pers_leads)

#%% Make Plot (same Subplot)
# ----------------------------------

# Set Color Mode
# darkmode = False
# if darkmode == True:
#     plt.style.use('dark_background')
#     dfcol = "w"
# else:
#     plt.style.use('default')
#     dfcol = "k"

# Get some needed information variables
nvars        = len(varnames)

# Toggles and ticks
plotclasses  = [0,2]     # Just plot positive/negative
expnums      = [0,1]     # Which Experiments to Plot
detrends     = [0,1]     # Whether or not it was detrended
leadticks    = np.arange(0,26,5)
legend_sp    = 2         # Subplot where legend is included
ytks         = np.arange(0,1.2,.2)

# Error Bars
plotstderr   = True  # If True, plot standard error (95%), otherwise plot 1-stdev
alpha        = 0.25  # Alpha of error bars

# Initialize figures
fig,axs =  plt.subplots(1,2,constrained_layout=True,figsize=(16,5.5))
it = 0
for iplot,ex in enumerate(expnums):
    
    # Get the axes row
    #axs_row = axs[iplot,:]
    if iplot == 0:
        ls='dotted'
    elif iplot == 1:
        ls='solid'
    
    # Unpack the data
    totalacc = np.array([alloutputs[ex][v]['total_acc'] for v in range(nvars)])
    classacc = np.array([alloutputs[ex][v]['class_acc'] for v in range(nvars)])
    ypred    = np.array([alloutputs[ex][v]['predictions'] for v in range(nvars)])
    ylabs    = np.array([alloutputs[ex][v]['targets'] for v in range(nvars)])
    
    # Indicate detrending
    exp_dt = detrends[ex]
    
    for rowid,c in enumerate(plotclasses):
        
        ax = axs[rowid]
        
        # Initialize plot
        if ex == 0:
            viz.label_sp(it,ax=ax,fig=fig,fontsize=pparams.fsz_splbl,
                         alpha=0.2,x=0.02)
        
        # Set Ticks/limits
        ax.set_xlim([0,24])
        ax.set_xticks(leadticks,fontsize=pparams.fsz_ticks)
        ax.set_ylim([0,1])
        ax.set_yticks(ytks,fontsize=pparams.fsz_ticks)
        ax.set_yticklabels((ytks*100).astype(int),)
        ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                            spinecolor="darkgray",tickcolor="dimgray",
                            ticklabelcolor="k",fontsize=pparams.fsz_ticks)
        
        # Add Class Labels
        if iplot == 0:
            ax.set_title("%s" %(classes_new[c]),fontsize=pparams.fsz_title,)
        
        # Loop for each predictor
        for i in range(nvars):
            
            # Compute Mean and 2*stderr (or sigma)
            mu       = classacc[i,:,:,c].mean(0)
            if plotstderr:
                sigma = 2*classacc[i,:,:,c].std(0) / np.sqrt(classacc.shape[1])
            else:
                sigma = np.array(plotacc).std(0)
            
            # Plot mean and bounds
            ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],alpha=1.0,lw=2.5,label=varnames[i] + "(%s)" % expdirs_long[ex],zorder=3,ls=ls)
            ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors[i],zorder=1)
        
        # Plot the persistence and chance baselines
        ax.plot(leads,persaccclass[exp_dt][:,c],color=dfcol,label="Persistence",ls="dashed")
        ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dotted")
        
        # Additional Labeling (y-axis and experiment)
        if ex == 0 and rowid == 0:
            ax.set_ylabel("Prediction Accuracy (%)",fontsize=pparams.fsz_axlbl,) # Label Y-axis for first column
            ax.text(-0.14, 0.55,expdirs_long[ex], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=pparams.fsz_title)
        
        # Label x-axis and set legend
        if (ex == 1):
            ax.set_xlabel("Prediction Lead (Years)",fontsize=pparams.fsz_axlbl,) # Label Y-axis for first column
        if it == legend_sp:
            ax.legend(ncol=3,fontsize=pparams.fsz_legend)
        it += 1

plt.savefig("%sPredictor_Intercomparison_byclass_stderr%i_%s_sameplot.png"% (figpath,plotstderr,comparename),
            dpi=200,bbox_inches="tight",transparent=False)
print(figpath)