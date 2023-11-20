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

# FNN 4 vs FNN 6
expdirs         = ("FNN4_128_Singlevar_PaperRun","FNN6_128_PaperRun")
expdirs_long    = ("FNN4","FNN6")
comparename     = "FNN4_v_FNN6"

# Ice vs No Ice Mask
# expdirs         = ("FNN4_128_Singlevar_PaperRun","FNN4_128_SingleVar_PaperRun_NoIceMask")
# expdirs_long    = ("Mask","No Mask")
# comparename     = "IceMask"

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
exp_dt       = 0 # Detrend option
fsz_ticks    = 18
fsz_axlbl    = 22
fsz_legend   = 14

# Toggles and ticks
plotclasses  = [0,2]     # Just plot positive/negative
expnums      = [0,1]     # Which Experiments to Plot
detrends     = [0,1]     # Whether or not it was detrended
leadticks    = np.arange(0,26,5)
legend_sp    = 2         # Subplot where legend is included
ytks         = np.arange(0,1.2,.2)

# Error Bars
plotstderr   = True  # If True, plot standard error (95%), otherwise plot 1-stdev
alpha        = 0.1  # Alpha of error bars

# Initialize figures
fig,axs =  plt.subplots(1,2,constrained_layout=True,figsize=(18,5.5))
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
    plotleads    = np.array(alloutputs[ex][0]['leads'])
    
    # Indicate detrending
    exp_dt = detrends[ex]
    
    for rowid,c in enumerate(plotclasses):
        
        ax = axs[rowid]
        
        # Initialize plot
        if ex == 0:
            viz.label_sp(it,ax=ax,fig=fig,fontsize=fsz_axlbl,
                         alpha=0.2,x=0.02)
        
        # Set Ticks/limits
        ax.set_xlim([0,24])
        ax.set_xticks(leadticks,fontsize=pparams.fsz_ticks)
        ax.set_ylim([0,1])
        ax.set_yticks(ytks,fontsize=pparams.fsz_ticks)
        ax.set_yticklabels((ytks*100).astype(int),)
        ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                            spinecolor="darkgray",tickcolor="dimgray",
                            ticklabelcolor="k",fontsize=fsz_ticks)
        
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
                sigma = np.classacc[i,:,:,c].std(0)
            
            # Plot mean and bounds
            ax.plot(plotleads,mu,color=varcolors[i],marker=varmarker[i],alpha=1.0,lw=2.5,label=varnames[i] + " (%s)" % expdirs_long[ex],zorder=3,ls=ls)
            ax.fill_between(plotleads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors[i],zorder=9)
        
        # Plot the persistence and chance baselines
        if iplot==1:
            ax.plot(leads,persaccclass[exp_dt][:,c],color=dfcol,label="Persistence",ls="dashed")
            ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dotted")
        
        # Additional Labeling (y-axis and experiment)
        if ex == 0 and rowid == 0:
            ax.set_ylabel("Prediction Accuracy (%)",fontsize=fsz_axlbl,) # Label Y-axis for first column
            # ax.text(-0.14, 0.55,expdirs_long[ex], va='bottom', ha='center',rotation='vertical',
            #         rotation_mode='anchor',transform=ax.transAxes,fontsize=pparams.fsz_title)
        
        # Label x-axis and set legend
        if (ex == 1):
            ax.set_xlabel("Prediction Lead (Years)",fontsize=fsz_axlbl,) # Label Y-axis for first column
        if it == legend_sp:
            ax.legend(ncol=3,fontsize=fsz_legend,loc=(.09,.72),framealpha=0.4)
        it += 1

plt.savefig("%sPredictor_Intercomparison_byclass_stderr%i_%s_sameplot.png"% (figpath,plotstderr,comparename),
            dpi=200,bbox_inches="tight",transparent=False)
print(figpath)


# ----------------------------------------------------------------------------
#%% Plot the Relevance Composites

# Figure 3: User Edits
# --------------------------

# Data and variable settings
varnames       = ("SST","SSH","SSS","SLP") 
varnames_plot  = ("SST","SSH","SSS","SLP")
expdir         = "FNN4_128_SingleVar_PaperRun_erule_exp1"
eparams        = train_cesm_params.train_params_all[expdir]
classes        = pparams.classes
datpath        = "../../results_manuscript/"
metrics_dir    = "%s%s/Metrics/Test_Metrics/" % (datpath,expdir)

leads          = np.arange(0,26,5)
print(metrics_dir)

# Compositing options
nmodels        = 100 # Specify manually how many networks to load for the analysis
topN           = 50 # Top N networks to include in composite

# Plotting Options
proj           = ccrs.PlateCarree()

# Toggles
debug          = False


#%% Load the Relevance and Predictor Composites
# -----------------------------
nvars       = len(varnames)
nleads      = len(leads)
#metrics_dir = "%s%s/Metrics/Test_Metrics/" % (datpath,expdir)
pcomps   = []
rcomps   = []
ds_all   = []
acc_dict = []
for v in range(nvars):
    # Load the composites
    varname = varnames[v]
    ncname = "%sTest_Metrics_CESM1_%s_evensample0_relevance_maps.nc" % (metrics_dir,varname)
    ds     = xr.open_dataset(ncname)
    rcomps.append(ds['relevance_composites'].values)
    pcomps.append(ds['predictor_composites'].values)
    
    # Load the accuracies
    ldname  = "%sTest_Metrics_CESM1_%s_evensample0_accuracy_predictions.npz" % (metrics_dir,varname)
    npz     = np.load(ldname,allow_pickle=True)
    expdict = proc.npz_to_dict(npz)
    acc_dict.append(expdict)

nleads,nruns,nclasses,nlat,nlon=rcomps[v].shape
lon = ds.lon.values
lat = ds.lat.values


# Composite top N performing networks for each class
# --------------------------------------------------
class_accs  = [acc_dict[v]['class_acc'] for v in range(nvars)]
rcomps_topN = np.zeros((nvars,nleads,nclasses,nlat,nlon))
pcomps_topN = np.zeros((nvars,nleads,nclasses,nlat,nlon))

for v in range(nvars):
    for l in tqdm.tqdm(range(nleads)):
        for c in range(nclasses):
            # Get ranking of models by test accuracy
            acc_list = class_accs[v][:,l,c] # [runs]
            id_hi2lo  = np.argsort(acc_list)[::-1] # Reverse to get largest value first
            id_topN   = id_hi2lo[:topN]
            
            # Make composite 
            rcomp_in  = rcomps[v][l,id_topN,c,:,:] # [runs x lat x lon]
            rcomps_topN[v,l,c,:,:] = rcomp_in.mean(0) # Mean along run dimension
            
            # Make predictor composite
            pcomp_in  = pcomps[v][l,id_topN,c,:,:]
            pcomps_topN[v,l,c,:,:] = pcomp_in.mean(0)
            
#%% Make Fig. 3
# -----------

# Select which class you would like to plot
c               = 2  # Class Index, where 0=NASST+, 1=Neutral, 2=NASST-

# Set darkmode
darkmode        = False
# if darkmode:
#     plt.style.use('dark_background')
#     dfcol = "w"
#     transparent      = True
# else:
#     plt.style.use('default')
#     dfcol = "k"
#     transparent      = False

# Indicate which variables to plot
plotvars         = pparams.varnames[:4]
plotorder        = [0,3,2,1] # Indices based on pparams.varnames

# Select which leadtimes to plot, as well as the bounding box
plot_bbox        = [-80,0,0,65]
leadsplot        = [25,20,10,5,0]

# Additional Options
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
cmin             = 1
clvl             = np.arange(-2.1,2.1,0.3)
no_sp_label      = False
cmap             ='cmo.balance'

# Font Sizes
fsz_title        = 32
fsz_axlbl        = 32
fsz_ticks        = 22
fsz_contourlbl   = 18

ia = 0 # Axes Index
fig,axs = plt.subplots(4,5,figsize=(24,16),
                       subplot_kw={'projection':proj},constrained_layout=True)
# Loop for variable
for v,varname in enumerate(plotvars):

    iv = plotorder[v]
    # Loop for leadtime
    for l,lead in enumerate(leadsplot):

        # Get lead index
        id_lead    = list(leads).index(lead)

        if debug:
            print("Lead %02i, idx=%i" % (lead,id_lead))

        # Axis Formatting
        ax = axs[v,l]
        blabel = [0,0,0,0]

        if v == 0:
            ax.set_title("Lead %02i Years" % (leads[id_lead]),fontsize=fsz_title)
        if l == 0:
            blabel[0] = 1
            ax.text(-0.18, 0.55, varnames_plot[iv], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
        if v == (len(plotvars)-1):
            blabel[-1]=1

        ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color='gray',fontsize=fsz_ticks,ignore_error=True)
        if no_sp_label is False:
            ax = viz.label_sp(ia,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)


        # --------- Composite the Relevances and variables --------
        plotrel = rcomps_topN[iv,id_lead,c,:,:]
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
        plotvar = pcomps_topN[iv,id_lead,c,:,:]

        # Boost SSS values by 1.5
        if varnames_plot[iv] == "SSS":
            plotrel = plotrel*2
        #plotvar = plotvar/np.max(np.abs(plotvar))

        # Set Land Points to Zero
        plotrel[plotrel==0] = np.nan
        plotvar[plotrel==0] = np.nan

        # Do the plotting
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmin,vmax=cmax,cmap=cmap)
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=1)
        ax.clabel(cl,clvl[::2],fontsize=fsz_contourlbl)
        ia += 1

        # Finish Leadtime Loop (Column)
    # Finish Variable Loop (Row)

# Make Colorbar
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
cb.ax.tick_params(labelsize=fsz_ticks)

# Make overall title
if not no_sp_label:
    plt.suptitle("Mean LRP Maps for Predicting %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],topN,),
                fontsize=fsz_title)
    
# Save Figure
savename = "%sPredictorComparison_%s_LRP_%s_top%02i_normalize%i_abs%i_Draft2.png" % (figpath,expdir,classes[c],topN,normalize_sample,absval)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)



