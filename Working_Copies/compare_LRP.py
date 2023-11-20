#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare LRP Outputs from Test Metrics

- Examine LRP outputs from [compute_test_metrics.py]
- Check effect of using different number of outputs



Copied setions from "Manuscript Figures.ipynb"

Created on Fri Nov  3 13:49:45 2023

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


#%%
# Import custom packages and other modules

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
figpath      = pparams.figpath

#%% Set up plotting parameters
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

#%% Figure 3: User Edits
# --------------------------

# Data and variable settings
varnames       = ("SST","SSH","SSS","SLP")#)#"SSS","SLP") 
varnames_plot  = ("SST","SSH","SSS","SLP")#"SSS","SLP")
expdir         = "FNN4_128_SingleVar_PaperRun"
eparams        = train_cesm_params.train_params_all[expdir]
classes        = pparams.classes
datpath        = "../../results_manuscript/"
metrics_dir    = "%s%s/Metrics/Test_Metrics/" % (datpath,expdir)
print(metrics_dir)

# Compositing options
nmodels        = 100 # Specify manually how many networks to load for the analysis
topN           = 50 # Top N networks to include in composite

# Plotting Options
proj           = ccrs.PlateCarree()

# Toggles
debug          = False

#%%# Load the Relevance and Predictor Composites
# -----------------------------

leads           = np.arange(0,26,1)
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


#%% Composite top N performing networks for each class
# --------------------------------------------------
class_accs  = [acc_dict[v]['class_acc'] for v in range(nvars)]
rcomps_topN = np.zeros((nvars,nleads,nclasses,nlat,nlon))

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


#%%
# Make Fig. 3
# -----------

# Select which class you would like to plot
c               = 0 # Class Index, where 0=NASST+, 1=Neutral, 2=NASST-

# Set darkmode
darkmode = False
# if darkmode:
#     plt.style.use('dark_background')
#     dfcol = "w"
#     transparent      = True
# else:
#     plt.style.use('default')
#     dfcol = "k"
#     transparent      = False

# Indicate which variables to plot (CHANGE THINGS HERE!)
plotvars         = pparams.varnames[:4]#['SST',"SSH"]#pparams.varnames[:4]
plotorder        = [0,3,2,1]#[0,1]#[0,3,2,1] # Indices based on pparams.varnames

# Select which leadtimes to plot, as well as the bounding box
plot_bbox        = [-80,0,0,60]
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

ia = 0
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
        plotvar = pcomps[iv][id_lead,c,:,:]

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
savename = "%sPredictorComparison_LRP_%s_%s_top%02i_normalize%i_Draft2.png" % (figpath,expdir,classes[c],topN,normalize_sample)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
# plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)

# ------------
#%% Visualize Relevance for a specific leadtime and predictor
# ------------

id_lead  = 0
iv       = 3
c        = 0

fig,ax = plt.subplots(1,1,figsize=(12,10),
                       subplot_kw={'projection':proj},constrained_layout=True)
blabel=[1,0,0,1]
ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color='gray',fontsize=fsz_ticks,ignore_error=True)



# --------- Composite the Relevances and variables --------
plotrel = rcomps_topN[iv,id_lead,c,:,:]
if normalize_sample == 2:
    plotrel = plotrel/np.max(np.abs(plotrel))
plotvar = pcomps[iv][id_lead,c,:,:]

# Boost SSS values by 1.5
if varnames_plot[iv] == "SSS":
    plotrel = plotrel*2
#plotvar = plotvar/np.max(np.abs(plotvar))

# Set Land Points to Zero
plotrel[plotrel==0] = np.nan
plotvar[plotrel==0] = np.nan

# Do the plotting
pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap=cmap)
cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=1)
ax.clabel(cl,clvl[::2],fontsize=fsz_contourlbl)

# Set Title
ax.set_title("%s | %s | Lead: %02i yr" % (varnames_plot[iv],pparams.classes[c],leads[id_lead]),fontsize=35)

# Make Colorbar
cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.035,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=24)
cb.ax.tick_params(labelsize=fsz_ticks)

savename = "%s%s_Relevance_Composite_topN%02i_%s_class%s_lead%02i.png" % (figpath,expdir,topN,varnames_plot[iv],c,leads[id_lead])
print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Check Differences if you composite a different number of outputs
# --------------------------------------------------

#% Calculate composites of top N performing networks for each class
class_accs      = [acc_dict[v]['class_acc'] for v in range(nvars)]
topNs           = [10,25,50,75,100]
rcomps_topN_all = []
for TN in range(len(topNs)):
    topN = topNs[TN]
    
    rcomps_topN = np.zeros((nvars,nleads,nclasses,nlat,nlon))
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
    rcomps_topN_all.append(rcomps_topN)

#%% Visualize the relevance

fsz_ticks = 14
fsz_title = 16

ivar   = 1
ilead  = 25
iclass = 0
cmax   = 1

topN_name = "%sRel_%s_%s_lead%02i_class%i.png" % (figpath,expdir,varnames[ivar],leads[ilead],iclass,)


fig,axs = plt.subplots(1,5,figsize=(12,16),
                       subplot_kw={'projection':proj},constrained_layout=True)

for a in range(5):
    ax = axs[a]
    ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color='gray',fontsize=fsz_ticks,ignore_error=True)
    ax.set_title("Top %i" % (topNs[a]),fontsize=fsz_title)
    
    plotrel = rcomps_topN_all[a][ivar,ilead,iclass,:,:]
    plotrel = plotrel/np.max(np.abs(plotrel))
    # Set Land Points to Zero
    plotrel[plotrel==0] = np.nan
    
    pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap=cmap)

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.008,pad=0.01)
cb.set_label("Normalized Relevance")
plt.savefig(topN_name,dpi=150,bbox_inches="tight")
#%% Visualize relevance difference

fsz_ticks = 14
fsz_title = 14


ivar      = 1
ilead     = 25
iclass    = 0


itopN_ref = topNs.index(50)


subtract_after_norm = True
cmax                 = 0.1


# Make the name
topN_name = "%sRelDiff_%s_%s_lead%02i_class%i_topNref50_afternorm%i.png" % (figpath,expdir,varnames[ivar],leads[ilead],iclass,
                                                           subtract_after_norm,)

ref_map             = rcomps_topN_all[itopN_ref][ivar,ilead,iclass,:,:]

fig,axs = plt.subplots(1,5,figsize=(12,16),
                       subplot_kw={'projection':proj},constrained_layout=True)

for a in range(5):
    ax = axs[a]
    ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color='gray',fontsize=fsz_ticks,ignore_error=True)
    
    plotrel = rcomps_topN_all[a][ivar,ilead,iclass,:,:]
    if not subtract_after_norm:
        plotrel = plotrel - ref_map
    plotrel = plotrel/np.max(np.abs(plotrel))
    if subtract_after_norm:
        ref_map = ref_map / np.max(np.abs(ref_map))
        plotrel = plotrel - ref_map
        
    # Set Land Points to Zero
    if a != itopN_ref:
        plotrel[plotrel==0] = np.nan
        
    title_val = "%.3f $\pm$ %.3f" % (np.nanmean(np.abs(plotrel)),np.nanstd(np.abs(plotrel)))
    ax.set_title("Top %i\n$\mu$=%s" % (topNs[a],title_val),fontsize=fsz_title)
    
    pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap=cmap)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.008,pad=0.01)
cb.set_label("Relevance Difference\nRelative to Top %i composite" % (50))
plt.savefig(topN_name,dpi=150,bbox_inches="tight")

#%%
