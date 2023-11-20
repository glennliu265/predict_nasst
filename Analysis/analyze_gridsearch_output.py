#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze the Hyperparameter Gridsearch Output for FNNs

Copied section from train_gridsearch

Created on Wed Nov  1 17:00:39 2023

@author: gliu
"""

import numpy as np
import itertools

import sys
import numpy as np
import os
import time
import tqdm
import os
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset

import matplotlib.pyplot as plt
import glob

# <Start copy from train_NN_CESM1.py>  ========================================
#%% Load custom packages and setup parameters

machine = 'Astraeus' # Indicate machine (see module packages section in pparams)

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
from amv import proc,viz

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some output paths
figpath = pparams.figpath
proc.makedir(figpath)

# Set some looping parameters and toggles
varnames            = ['SSH',]# Only Supportss 1!! ["SST","SSS","SLP","NHFLX",]       # Names of predictor variables
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


splabels_class = ['A) NASST+','B) Neutral','C) NASST-']
splabels_class2 = ['A) NASST+','B) NASST-']
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

# -------------------------------------------------------------
# %% I. Check # of Layers, Dropout (wrong), # Units
# -------------------------------------------------------------

# Set some variables needed
varname    = varnames[0]
predictors = data[[0],...] # Get selected predictor

# For FNN4, just test the number of layers and units
nlayers  = [2,4,6,8,10]      # Number of Layers
nunits   = [64,128,256] # Number of Units
dropouts = [False] # Useing Dropout Layer 

# Set up testing dictionary
test_param_names  = ["nlayers","nunits","dropout"]
test_param_values = [nlayers,nunits,dropouts]
test_params       = dict(zip(test_param_names,test_param_values))

# Get some measurements
nparams = len(test_param_names)
nvalues = [len(p) for p in test_param_values]
ntotal  = np.prod(nvalues)

# Make Parameter combinations -------------------------------------------------
param_combinations = list(itertools.product(*test_param_values))
ncombos            = len(param_combinations)
print(ntotal == len(param_combinations))

#  -------------------------
#%% Load the FNN Metrics
# --------------------------

combo_names = []

nruns = len(runids)
nleads = len(leads)
nclasses = len(pparams.classes)
accs_all = np.full((ncombos,nruns,nleads,nclasses),np.nan)

for nc in range(ncombos): # Loop for each combination -------------------------
    pcomb           = param_combinations[nc]
    ct              = time.time()
    
    # Copy dictionaries to use for this particular combo
    combo_expdict   = eparams_original.copy()
    combo_paramdict = nnparams_original.copy()

    # Get the Experiment string and replace into the dictionaray
    expstr = ""
    for p in range(nparams):
        
        # Make the experiment string
        name = test_param_names[p]
        expstr += "%s%s_" % (name,pcomb[p])
        
        # Copy parameters into dictionary ----------------------------
        if name in combo_expdict.keys(): # Check experiment dictionary
            print("Found <%s> in eparams_original;\t replacing with value: %s" % (name,pcomb[p]))
            combo_expdict[name] = pcomb[p]
        elif name in combo_paramdict.keys(): # Check parameter dictionary
            print("Found <%s> in nnparams_original;\t replacing with value: %s" % (name,pcomb[p]))
            combo_paramdict[name] = pcomb[p]
        # ------------------------------------------------------------
    
    # Make the experiment string and prepare the folder
    expstr = expstr[:-1]
    print(expstr)
    outdir = "%s%s/ParamTesting/%s/Metrics/" % (pparams.datpath,expdir,expstr)
    combo_names.append(expstr)
    
    # Retrieve the metrics file
    flist = glob.glob("%s*%s*ALL.npz" % (outdir,varname))
    nruns = len(flist)
    flist.sort()
    print("Found %i files" % (nruns))
    
    # Read the files
    
    for f in range(nruns):
        ld  = np.load(flist[f])
        acc = ld['acc_by_class']
        accs_all[nc,f,:,:] = acc.copy()

#%% Quick Spaghetti Plot of Architecture Differences

iclass = 0

#idori_dropout = param_combinations.index((4, 128, True))
idori_nodrop  = param_combinations.index((4, 128, False))
idori = [idori_nodrop]

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

for nc in range(ncombos):
    
    pcomb = param_combinations[nc]
    
    if pcomb[-1] == True:
        ls = 'solid'
    elif pcomb[-1] == False:
        ls = 'dashed'
    
    plot_acc = accs_all[nc,:,:,iclass].mean(0)
    
    if nc in idori:
        
        ax.plot(leads,plot_acc,label=combo_names[nc],ls=ls,color="k")
    else:
        ax.plot(leads,plot_acc,label=combo_names[nc],ls=ls)
ax.legend()

# -------------------------------------------------------------
# %% II. Compare # of Layers and Units per Layer
# -------------------------------------------------------------
#plt.style.use("")
iclass        = 2

#idori_dropout = param_combinations.index((4, 128, True))
idori_nodrop  = param_combinations.index((4, 128, False))
idori         = [idori_nodrop]


fsz_axlbl    = 20
leadticks    = np.arange(0,26,5)


for iclass in range(3):

    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
        
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(13,4))
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=pparams.fsz_ticks)
    
    for nc in range(ncombos):
        
        pcomb = param_combinations[nc]
        
        if pcomb[-1] == True:
            continue
        
        if pcomb[0] == 2:
            ls = 'dotted'
            c  = "red"
        elif pcomb[0] == 4:
            ls = 'dashed'
            c  = "goldenrod"
        elif pcomb[0] == 6:
            ls = 'solid'
            c  = "darkblue"
        elif pcomb[0] == 8:
            ls = 'dashdot'
            c  = 'violet'
        elif pcomb[0] == 10:
            ls = 'solid'
            c  = "gold"
        
        if pcomb[1] == 64:
            #alpha = 0.2
            marker = "v"
        elif pcomb[1] == 128:
            #alpha = 0.6
            marker = 'd'
        elif pcomb[1] == 256:
            #alpha = 1
            marker = '^'
        alpha = 1
        plot_acc = accs_all[nc,:,:,iclass].mean(0)
        
        if nc in idori:
            
            ax.plot(leads,plot_acc,label=combo_names[nc],ls=ls,color="k",alpha=alpha,marker=marker)
        else:
            ax.plot(leads,plot_acc,label=combo_names[nc],ls=ls,alpha=alpha,marker=marker,color=c)
    ax.set_xlabel("Leadtime (Years)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    
    ax.legend(ncol=4)
    savename = "%sFNN4_ParamTesting_%s_class%i.png" % (figpath,varname,iclass)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%% Examine Effect of Layers and Units, Redo all in 1 plot

#plt.style.use("")
iclass        = 2

#idori_dropout = param_combinations.index((4, 128, True))
idori_nodrop  = param_combinations.index((4, 128, False))
idori         = [idori_nodrop]

fsz_axlbl    = 20
fsz_splbl    = 22
fsz_tklbl    = 18
leadticks    = np.arange(0,26,5)

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(16.5,12))

for iclass in range(3):

    ax = axs[iclass]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=fsz_tklbl)
    
        
    viz.label_sp(splabels_class[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=fsz_splbl,
             alpha=0.2,x=0.)
    
    for nc in range(ncombos):
        
        pcomb = param_combinations[nc]
        
        if pcomb[-1] == True:
            continue
        
        if pcomb[0] == 2:
            ls = 'dotted'
            c  = "red"
        elif pcomb[0] == 4:
            ls = 'dashed'
            c  = "goldenrod"
        elif pcomb[0] == 6:
            ls = 'solid'
            c  = "darkblue"
        elif pcomb[0] == 8:
            ls = 'dashdot'
            c  = 'violet'
        elif pcomb[0] == 10:
            ls = 'solid'
            c  = "gold"
        
        if pcomb[1] == 64:
            #alpha = 0.2
            marker = "v"
        elif pcomb[1] == 128:
            #alpha = 0.6
            marker = 'd'
        elif pcomb[1] == 256:
            #alpha = 1
            marker = '^'
        alpha = 1
        plot_acc = accs_all[nc,:,:,iclass].mean(0)
        lbl = combo_names[nc].replace("_dropoutFalse","")
        if nc in idori:
            
            ax.plot(leads,plot_acc,label=lbl,ls=ls,color="k",alpha=alpha,marker=marker)
        else:
            ax.plot(leads,plot_acc,label=lbl,ls=ls,alpha=alpha,marker=marker,color=c)
    
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Accuracy",fontsize=fsz_axlbl)
    if iclass == 1:
        ax.legend(ncol=5,fontsize=14,loc=[.025,.6])
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=fsz_axlbl)
#plt.suptitle("Predictor: %s" % (varname))
savename = "%sFNN4_ParamTesting_%s_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Fig. S1 : FNN4 Hyperparams, but just NASST+ and NASST- (Figure S1, Revision 01)

class_choose = [0,2]

#idori_dropout = param_combinations.index((4, 128, True))
idori_nodrop  = param_combinations.index((4, 128, False))
idori         = [idori_nodrop]

fsz_axlbl    = 20
fsz_splbl    = 28
fsz_tklbl    = 18
leadticks    = np.arange(0,26,5)

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(16.5,10))

for a in range(2):
    iclass = class_choose[a]
    ax = axs[a]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=fsz_tklbl)
    
        
    viz.label_sp(splabels_class2[a],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=fsz_splbl,
             alpha=0.2,x=0.)
    
    for nc in range(ncombos):
        
        pcomb = param_combinations[nc]
        
        if pcomb[-1] == True:
            continue
        
        if pcomb[0] == 2:
            ls = 'dotted'
            c  = "red"
        elif pcomb[0] == 4:
            ls = 'dashed'
            c  = "goldenrod"
        elif pcomb[0] == 6:
            ls = 'solid'
            c  = "darkblue"
        elif pcomb[0] == 8:
            ls = 'dashdot'
            c  = 'violet'
        elif pcomb[0] == 10:
            ls = 'solid'
            c  = "gold"
        
        if pcomb[1] == 64:
            #alpha = 0.2
            marker = "v"
        elif pcomb[1] == 128:
            #alpha = 0.6
            marker = 'd'
        elif pcomb[1] == 256:
            #alpha = 1
            marker = '^'
        alpha = 1
        plot_acc = accs_all[nc,:,:,iclass].mean(0)
        lbl = combo_names[nc].replace("_dropoutFalse","")
        if nc in idori:
            
            ax.plot(leads,plot_acc,label=lbl,ls=ls,color="k",alpha=alpha,marker=marker)
        else:
            ax.plot(leads,plot_acc,label=lbl,ls=ls,alpha=alpha,marker=marker,color=c)
    
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Validation Accuracy",fontsize=fsz_axlbl)
    if a == 0:
        ax.legend(ncol=5,fontsize=14,loc=[.035,.65],framealpha=0.3)
    elif a == 1:
        ax.set_xlabel("Leadtime (Years)",fontsize=fsz_axlbl)
#plt.suptitle("Predictor: %s" % (varname))
savename = "%sFNN4_ParamTesting_%s_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')



# -------------------------------------------------------------
# %% III. Compare # of Layers
# -------------------------------------------------------------
#  Metrics for these specific Experiments

nlayers_all      = np.arange(2,18,2)

load_old = False

nexps            = len(nlayers_all)
accs_all_nlayers = np.full((nexps,nruns,nleads,nclasses),np.nan)
combo_names      = []


for nn in range(len(nlayers_all)):
    
    nlayers = nlayers_all[nn]
    
    if nlayers < 11 and load_old:
        expstr = "nlayers%i_nunits128_dropoutFalse" % (nlayers)
        expstr_out = "nlayers%i_nunits128" % (nlayers)
    else:
        expstr = "nlayers%i_nunits128" % (nlayers)
        expstr_out = expstr
    
    outdir = "%s%s/ParamTesting/%s/Metrics/" % (pparams.datpath,expdir,expstr)

    combo_names.append(expstr_out)
    
    # Retrieve the metrics file
    flist = glob.glob("%s*%s*ALL.npz" % (outdir,varname))
    nruns = len(flist)
    flist.sort()
    print("Found %i files" % (nruns))
    
    # Read the files
    for f in range(nruns):
        ld  = np.load(flist[f])
        acc = ld['acc_by_class']
        accs_all[nn,f,:,:] = acc.copy()
        
        # Add the actual predictions
        if (f == 0) and (nn==0): # Preallocate
            nmodels = len(flist)
            nleads,nsamples = ld['yvalpred'].shape
            ypreds          = np.zeros((nexps,nmodels,nleads,nsamples))
            ylabs           = ypreds.copy()
        ypreds[nn,f,:,:] = ld['yvalpred'].copy()
        ylabs[nn,f,:,:]  = ld['yvallabels'].copy()

#%% Make the Plot (Comparing Number of Layers)

fsz_axlbl    = 20
leadticks    = np.arange(0,26,5)
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,10))

for iclass in range(3):

    ax = axs[iclass]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=14)
    
        
    viz.label_sp(pparams.classes[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=18,
             alpha=0.2,x=0.)
    
    for nn in range(nexps):
        
        nlayers = nlayers_all[nn]
        #pcomb = param_combinations[nc]
        
        if pcomb[-1] == True:
            continue
        
        if nlayers == 2:
            ls = 'dotted'
            c  = "red"
            marker="v"
        elif nlayers == 4:
            ls = 'dashed'
            c  = "goldenrod"
            marker="d"
        elif nlayers == 6:
            ls = 'solid'
            c  = "darkblue"
            marker="x"
        elif nlayers == 8:
            ls = 'dashdot'
            c  = 'violet'
            marker="+"
        elif nlayers == 10:
            ls = 'solid'
            c  = "gold"
            marker="."
        elif nlayers == 12:
            ls = "dotted"
            c  = "magenta"
            marker="h"
        elif nlayers == 14:
            ls = "dashed"
            c = "cyan"
            marker="s"
        elif nlayers == 16:
            ls = "solid"
            c = 'limegreen'
            marker="^"
        
        
        alpha = 1
        plot_acc = accs_all[nn,:,:,iclass].mean(0)
        lbl = combo_names[nn].replace("_dropoutFalse","")
        if nc in idori:
            
            ax.plot(leads,plot_acc,label=lbl,ls=ls,color="k",alpha=alpha,marker=marker)
        else:
            ax.plot(leads,plot_acc,label=lbl,ls=ls,alpha=alpha,marker=marker,color=c)
    
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Accuracy",fontsize=16)
    if iclass == 0:
        ax.legend(ncol=3,fontsize=13.5)
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=16)
#plt.suptitle("Predictor: %s" % (varname))
savename = "%sFNN4_LayersTesting_%s_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -------------------------------------------------------------
# %% IV. Looking at # of Predictions + Confusion Matrix
# -------------------------------------------------------------
nclasses = len(pparams.classes) 
percpreds  = np.zeros((nexps,nmodels,nleads,nclasses))

for ex in range(nexps):
    for mm in range(nmodels):
        for l in range(nleads):
            
            preds = ypreds[ex,mm,l,:]
            targs = ylabs[ex,mm,l,:]
            nsamp = len(preds)
            
            
            
            for iclass in range(nclasses):
                percpreds[ex,mm,l,iclass] = (preds==iclass).sum() / nsamp
#%% Plot this percentage


fsz_axlbl    = 20
leadticks    = np.arange(0,26,5)
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,10))

for iclass in range(3):

    ax = axs[iclass]
    if iclass in [0,2]:
        xlm = [0,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=14)
        
    viz.label_sp(pparams.classes[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=18,
             alpha=0.2,x=0.)
    
    for nn in range(nexps):
        
        nlayers = nlayers_all[nn]
        #pcomb = param_combinations[nc]
        
        if pcomb[-1] == True:
            continue
        
        if nlayers == 2:
            ls = 'dotted'
            c  = "red"
            marker="v"
        elif nlayers == 4:
            ls = 'dashed'
            c  = "goldenrod"
            marker="d"
        elif nlayers == 6:
            ls = 'solid'
            c  = "darkblue"
            marker="x"
        elif nlayers == 8:
            ls = 'dashdot'
            c  = 'violet'
            marker="+"
        elif nlayers == 10:
            ls = 'solid'
            c  = "gold"
            marker="."
        elif nlayers == 12:
            ls = "dotted"
            c  = "magenta"
            marker="h"
        elif nlayers == 14:
            ls = "dashed"
            c = "cyan"
            marker="s"
        elif nlayers == 16:
            ls = "solid"
            c = 'limegreen'
            marker="^"
        
        
        alpha = 1
        plot_acc = percpreds[nn,:,:,iclass].mean(0)#accs_all[nn,:,:,iclass].mean(0)
        lbl = combo_names[nn].replace("_dropoutFalse","")

        ax.plot(leads,plot_acc,label=lbl,ls=ls,alpha=alpha,marker=marker,color=c)
    
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Percentage of Predictions",fontsize=16)
    if iclass == 0:
        ax.legend(ncol=3,fontsize=13.5)
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=16)
#plt.suptitle("Predictor: %s" % (varname))
savename = "%sFNN4_Perc_Pred_%s_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -----------------------------------------------------------------------------
# %% V. CNN Results
# -------------------------------------------------------------

expdir2 = "CNN2_PaperRun"

# Glob All Sets
cnnpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CNN2_PaperRun/ParamTesting/"
cnnexps  = [os.path.basename(x) for x in glob.glob(cnnpath + "nlayers*")]
cnnexps.sort()

# Explicit Sets 
# Set 1: Test # of Layers
cnnexp_layer = [
    "nlayers1_filtersize2_stride2",
    "nlayers2_filtersize2_stride2",
    "nlayers3_filtersize2_stride2",]

cnnexp_stride = [
    "nlayers1_filtersize2_stride1",
    "nlayers1_filtersize2_stride2",
    "nlayers1_filtersize2_stride3",
    "nlayers1_filtersize2_stride4",
    "nlayers2_filtersize2_stride1",
    "nlayers2_filtersize2_stride2",
    "nlayers2_filtersize2_stride3",]

cnnexp_filter = [
    "nlayers1_filtersize2_stride2",
    "nlayers1_filtersize3_stride2",
    "nlayers1_filtersize4_stride2",
    "nlayers2_filtersize2_stride2",
    "nlayers2_filtersize3_stride2",
    "nlayers2_filtersize4_stride2",
    ]

cnnexp_list = {
    'nlayers' : cnnexp_layer,
    'stride'  : cnnexp_stride,
    'filter'  : cnnexp_filter,
    }

#  -------------------------
#%% Load the CNN Metrics
# --------------------------

ncombos  = len(cnnexps)
nruns    = len(runids)
nleads   = len(leads)
nclasses = len(pparams.classes)
cnn_accs_all = np.full((ncombos,nruns,nleads,nclasses),np.nan)
cnn_combo_names = []
for nc in range(ncombos): # Loop for each combination -------------------------
    ct              = time.time()
    # Make the experiment string and prepare the folder
    expstr = cnnexps[nc]
    print(expstr)
    outdir = "%s%s/Metrics/" % (cnnpath,expstr)
    cnn_combo_names.append(expstr)
    
    # Retrieve the metrics file
    flist = glob.glob("%s*%s*ALL.npz" % (outdir,varname))
    nruns = len(flist)
    flist.sort()
    print("Found %i files" % (nruns))
    
    # Read the files
    
    for f in range(nruns):
        ld  = np.load(flist[f])
        acc = ld['acc_by_class']
        cnn_accs_all[nc,f,:,:] = acc.copy()

#%% Plot CNN Hyerparameter Tests (All Classes)

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(18,10))

for iclass in range(3):

    ax = axs[iclass]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=14)
    
        
    viz.label_sp(splabels_class[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=20,
             alpha=0.2,x=0.)
    
    for nc in range(ncombos):
        
        alpha = 1
        plot_acc = cnn_accs_all[nc,:,:,iclass].mean(0)
        lbl      = cnn_combo_names[nc].replace("_dropoutFalse","")
        
        
        if "nlayers1" in lbl:
            #c="red"
            #marker = "x"
            ls = "dotted"
        elif "nlayers2" in lbl:
            #c='darkblue'
            #marker="."
            ls = "dashed"
        elif "nlayers3" in lbl:
            #c='violet'
            #marker="d"
            ls = "solid"
            
        if "filtersize2" in lbl:
            c="red"
            #ls = "dotted"
        elif "filtersize3" in lbl:
            c='darkblue'
            #ls = "dashed"
        elif "filtersize4" in lbl:
            c='violet'
           # ls = "solid"
            
        
        if "stride1" in lbl:
            marker="v"
            #c = "red"
        elif "stride2" in lbl:
            marker="d"
            #c = "gold"
        elif "stride3" in lbl:
            marker="x"
            #c = "blue"
        elif "stride4" in lbl:
            marker="^"
            #c = "magenta"
            
        # if ("stride1" in lbl) and ("filtersize2" in lbl) and ("nlayers2" in lbl):
        #     print(nc)
        #     c = 'k'
        
        ax.plot(leads,plot_acc,label=lbl,ls=ls,marker=marker,lw=2.5,c=c,markersize=10,alpha=0.8)
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Accuracy",fontsize=16)
    if iclass == 0:
        ax.legend(ncol=5,fontsize=12)
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=16)

savename = "%sCNN2_ParamTesting_%s_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Fig S2. Redo but just focus on NASST+ and NASST- (Figure S2 of Revision 01!!)

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(18,10))

class_choose=[0,2]
for a in range(2):
    iclass = class_choose[a]
    
    ax = axs[a]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=18)
    
        
    viz.label_sp(splabels_class2[a],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=25,
             alpha=0.2,x=0.)
    
    for nc in range(ncombos):
        
        alpha = 1
        plot_acc = cnn_accs_all[nc,:,:,iclass].mean(0)
        lbl      = cnn_combo_names[nc].replace("_dropoutFalse","")
        
        
        if "nlayers1" in lbl:
            #c="red"
            #marker = "x"
            ls = "dotted"
        elif "nlayers2" in lbl:
            #c='darkblue'
            #marker="."
            ls = "dashed"
        elif "nlayers3" in lbl:
            #c='violet'
            #marker="d"
            ls = "solid"
            
        if "filtersize2" in lbl:
            c="red"
            #ls = "dotted"
        elif "filtersize3" in lbl:
            c='darkblue'
            #ls = "dashed"
        elif "filtersize4" in lbl:
            c='violet'
           # ls = "solid"
            
        
        if "stride1" in lbl:
            marker="v"
            #c = "red"
        elif "stride2" in lbl:
            marker="d"
            #c = "gold"
        elif "stride3" in lbl:
            marker="x"
            #c = "blue"
        elif "stride4" in lbl:
            marker="^"
            #c = "magenta"
            
        # if ("stride1" in lbl) and ("filtersize2" in lbl) and ("nlayers2" in lbl):
        #     print(nc)
        #     c = 'k'
        
        ax.plot(leads,plot_acc,label=lbl,ls=ls,marker=marker,lw=2.5,c=c,markersize=10,alpha=0.8)
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Validation Accuracy",fontsize=22)
    if a == 0:
        ax.legend(ncol=5,fontsize=12.5,loc=(.005,.65),framealpha=0.5)
    elif a == 1:
        ax.set_xlabel("Leadtime (Years)",fontsize=22)

savename = "%sCNN2_ParamTesting_%s_classPosNeg.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Make plots comparing accuracy for certain classes

testparam = "nlayers" # nlayers, stride, filter
inexps    = cnnexp_list[testparam]

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,10))
for iclass in range(3):

    ax = axs[iclass]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=14)
    
        
    viz.label_sp(pparams.classes[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=18,
             alpha=0.2,x=0.)
    
    for nc in range(ncombos):
        
        alpha = 1
        plot_acc = cnn_accs_all[nc,:,:,iclass].mean(0)
        lbl      = cnn_combo_names[nc].replace("_dropoutFalse","")
        
        if lbl not in inexps:
            print("%s not found in %s" % (lbl,inexps)) 
            continue
        
        if testparam == "nlayers":
            if "nlayers1" in lbl:
                marker = "x"
                c      = "red"
                ls = "dotted"
            elif "nlayers2" in lbl:
                marker="."
                c      = "blue"
            elif "nlayers3" in lbl:
                marker="d"
                c      = "magenta"
            ls  = 'solid'
            ncols = 3
        elif testparam == "stride":
            
            if "nlayers1" in lbl:
                #c      = "red"
                ls      = 'solid'
            elif "nlayers2" in lbl:
                #c      = "blue"
                ls      = 'dotted'
                
            if "stride1" in lbl:
                c = "red"
            elif "stride2" in lbl:
                c = "goldenrod"
            elif "stride3" in lbl:
                c = "blue"
            elif "stride4" in lbl:
                c = "magenta"
            ncols = 2
        
        elif testparam == "filter":
            
            if "nlayers1" in lbl:
                #c      = "red"
                ls      = 'solid'
            elif "nlayers2" in lbl:
                #c      = "blue"
                ls      = 'dotted'
            
            
            
            if "filtersize2" in lbl:
                c = 'red'
            elif "filtersize3" in lbl:
                c = 'blue'
            elif "filtersize4" in lbl:
                c = 'magenta'
            ncols = 2
        
        
        # if "nlayers1" in lbl:
        #     marker = "x"
        #     ls = "dotted"
        # elif "nlayers2" in lbl:
        #     marker="."
        # elif "nlayers3" in lbl:
        #     marker="d"
            
        # if "filtersize2" in lbl:
        #     ls = "dotted"
        # elif "filtersize3" in lbl:
        #     ls = "dashed"
        # elif "filtersize4" in lbl:
        #     ls = "solid"
            
        
        # if "stride1" in lbl:
        #     c = "red"
        # elif "stride2" in lbl:
        #     c = "gold"
        # elif "stride3" in lbl:
        #     c = "blue"
        # elif "stride4" in lbl:
        #     c = "magenta"
        
        # if nc in idori:
        #     ax.plot(leads,plot_acc,label=lbl,ls=ls,color="k",alpha=alpha,marker=marker)
        # else:
        #ax.plot(leads,plot_acc,label=lbl,ls=ls,alpha=alpha,marker=marker,color=c)
        ax.plot(leads,plot_acc,label=lbl,ls=ls,marker=marker,lw=3,c=c)
    
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Accuracy",fontsize=16)
    if iclass == 0:
        ax.legend(ncol=ncols,fontsize=13.5)
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=16)

savename = "%sCNN2_ParamTesting_%s_%s_classALL.png" % (figpath,testparam,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# -----------------------------------------------------------------------
#%% VI. Compare Effect of Dropout
# -----------------------------------------------------------------------

# Set Experiments and Load
expdir = "FNN4_128_SingleVar_PaperRun"

exps_dropout1  = [
    "nlayers2_nunits128",
    "nlayers4_nunits128",
    "nlayers6_nunits128",
                ]

exps_dropout0 = [
    "nlayers2_nunits128_dropoutFalse",
    "nlayers4_nunits128_dropoutFalse",
    "nlayers6_nunits128_dropoutFalse",]

exps = np.hstack([exps_dropout0,exps_dropout1])


expnames = [
    "2 Layers (0.5 Dropout)",
    "4 Layers (0.5 Dropout)",
    "6 Layers (0.5 Dropout)",
    "2 Layers (No Dropout)",
    "4 Layers (No Dropout)",
    "6 Layers (No Dropout)",
    ]

ncombos  = len(exps)
nruns    = len(runids)
nleads   = len(leads)
nclasses = len(pparams.classes)
exp_accs_all = np.full((ncombos,nruns,nleads,nclasses),np.nan)
for nc in range(ncombos): # Loop for each combination -------------------------
    ct              = time.time()
    # Make the experiment string and prepare the folder
    expstr = cnnexps[nc]
    print(expstr)
    outdir = "%s%s/Metrics/" % (cnnpath,expstr)
    
    # Retrieve the metrics file
    flist = glob.glob("%s*%s*ALL.npz" % (outdir,varname))
    nruns = len(flist)
    flist.sort()
    print("Found %i files" % (nruns))
    
    # Read the files
    
    for f in range(nruns):
        ld  = np.load(flist[f])
        acc = ld['acc_by_class']
        exp_accs_all[nc,f,:,:] = acc.copy()

#%% Visualize Dropout Differences by # of Layers

inexps    = exp_accs_all

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,10))

ncols = 2

for iclass in range(3):

    ax = axs[iclass]
    if iclass in [0,2]:
        xlm = [0.4,1]
    else:
        xlm = [0,1]
    
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=14)
    
        
    viz.label_sp(pparams.classes[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=18,
             alpha=0.2,x=0.)
    
    for nc in range(ncombos):
        
        alpha = 1
        plot_acc = exp_accs_all[nc,:,:,iclass].mean(0)
        lbl      = expnames[nc]
        
        if "2" in lbl:
            c='orange'
        elif "4" in lbl:
            c='darkblue'
        elif "6" in lbl:
            c='violet'
        
        if "No" in lbl:
            ls = 'dotted'
            marker = "x"
        else:
            ls = 'solid'
            marker = "o"
        
        # if lbl not in inexps:
        #     print("%s not found in %s" % (lbl,inexps)) 
        #     continue
        
        ax.plot(leads,plot_acc,label=lbl,ls=ls,marker=marker,lw=3,c=c)
    
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Accuracy",fontsize=16)
    if iclass == 0:
        ax.legend(ncol=ncols,fontsize=13.5)
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=16)

savename = "%sCNN2_ParamTesting_%s_CNNDropout_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize Dropout Differences by # of Layers



inexps     = exp_accs_all

# Dropout minus No Dropout [nlayers x lead x class]
inexp_diff = inexps[:3,:,:,:].mean(1) - inexps[3:,:,:,:].mean(1)

# Take Maximum Inter-Network Standard Deviation between Dropout and No Dropout Scenario
inexp_std  = np.array([inexps[:3,:,:,:].std(1),inexps[3:,:,:,:].std(1)]).max(0) 

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,10))

ncols = 2

for iclass in range(3):

    ax = axs[iclass]
    
    ax.axhline([0],ls='solid',lw=0.5,color="k")
    xlm =  [-.25,.25]
    ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                    spinecolor="darkgray",tickcolor="dimgray",
                    ticklabelcolor="k",fontsize=14)
    
    
    viz.label_sp(pparams.classes[iclass],labelstyle="%s",usenumber=True,
                 ax=ax,fig=fig,fontsize=18,
             alpha=0.2,x=0.)
    
    for nc in range(3):
        
        alpha = 1
        plot_acc    = inexp_diff[nc,:,iclass]#exp_accs_all[nc,:,:,iclass].mean(0)
        plot_stderr = 2* inexp_std[nc,:,iclass] / np.sqrt(nmodels)
        lbl      = expnames[nc][:8]
        
        
        if "2" in lbl:
            c='orange'
        elif "4" in lbl:
            c='darkblue'
        elif "6" in lbl:
            c='violet'
        
        if "No" in lbl:
            ls = 'dotted'
            marker = "x"
        else:
            ls = 'solid'
            marker = "o"
        
        # if lbl not in inexps:
        #     print("%s not found in %s" % (lbl,inexps)) 
        #     continue
        
        ax.plot(leads,plot_acc,label=lbl,ls=ls,marker=marker,lw=3,c=c)
        
        ax.plot(leads,plot_stderr,color=c,ls='dashed',lw=1,)
        ax.plot(leads,-plot_stderr,color=c,ls='dashed',lw=1,)
        #ax.fill_between(leads,-plot_stderr,plot_stderr,color=c,ls='dashed',lw=0.75,alpha=.1)
        
    
    #ax.set_title("Predictor: %s, Class: %s" % (varname,pparams.classes[iclass]))
    ax.set_ylim([xlm[0],xlm[-1]])
    ax.set_xlim([0,25])
    
    #ax.axhline([.6],label="Original Acc. (PaperRun)",color="k",ls='dashed')
    ax.set_ylabel("Accuracy Difference \n (Dropout - No Dropout)",fontsize=16)
    if iclass == 0:
        ax.legend(ncol=ncols,fontsize=13.5)
    elif iclass == 2:
        ax.set_xlabel("Leadtime (Years)",fontsize=16)

savename = "%sCNN2_ParamTesting_%s_CNNDropout_classALL.png" % (figpath,varname)
plt.savefig(savename,dpi=150,bbox_inches='tight')