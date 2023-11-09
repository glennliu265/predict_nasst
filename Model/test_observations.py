#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test selected networks on reanalysis or observation based dataset

- Works with gridded obs/reanalysis dataset preprocessed in 
- 
    Copied upper section from test_predictor_uncertainty

Created on Tue Apr  4 11:20:44 2023

@author: gliu
"""

import numpy as np
import sys
import glob
import importlib
import copy
import xarray as xr

import torch
from torch import nn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm
import time
import os

from torch.utils.data import DataLoader, TensorDataset,Dataset
#%% Load some functions

#% Load custom packages and setup parameters
# Import general utilities from amv module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc,viz


# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am
import pamv_visualizer as pviz

# Import LRP package
lrp_path = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/ml_demo/Pytorch-LRP-master/"
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

#%% User Edits

# Shared Information
varname            = "SST" # Testing variable
detrend            = True
leads              = np.arange(0,26,1)
region_name        = "NAT"
nsamples           = "ALL"
shuffle_trainsplit = False

# Set Manual Threshold
manual_threshold   = None#0.37 # Set to None to automatically detect

# CESM1-trained model information
expdir             = "FNN4_128_SingleVar_PaperRun_detrended"
modelname          = "FNN4_128"
nmodels            = 100 # Specify manually how much to do in the analysis
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens                = 0#eparams['ens']
runids             = np.arange(0,nmodels)

# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes
bbox               = pparams.bbox

#eparams['shuffle_trainsplit'] = False # Turn off shuffling

# Reanalysis dataset information
dataset_name       = "HadISST"
regrid             = "CESM1"


# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1

# Other toggles
debug              = False
checkgpu           = True
darkmode           = False


if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False


#%% Load the datasets

# Check to detrended
if "detrend" in expdir:
    detrend = True
else:
    detrend = False
    

# Load reanalysis datasets [channel x ensemble x year x lat x lon]
re_data,re_lat,re_lon=dl.load_data_reanalysis(dataset_name,varname,bbox,
                        detrend=detrend,regrid=regrid,return_latlon=True)

# Load the target dataset
re_target = dl.load_target_reanalysis(dataset_name,region_name,detrend=detrend)
re_target = re_target[None,:] # ens x year

# Do further preprocessing and get dimensions sizes
re_data[np.isnan(re_data)]     = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = re_data.shape
inputsize                      = nchannels*nlat*nlon

#%% Load regular data... (as a comparison for debugging, can remove later)

# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
#data,lat,lon   = dl.load_data_cesm([varname,],eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
#data                           = data[:,0:ens,...]      # Limit to Ens
#data[np.isnan(data)]           = 0                      # NaN Points to Zero

#%% Make the classes from reanalysis data

# Set exact threshold value
if manual_threshold is None:
    std1         = re_target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
else:
    std1         = manual_threshold
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']
    
#thresholds_in  = [-.36,.36]

# Classify AMV Events
target_class = am.make_classes(re_target.flatten()[:,None],thresholds_in,
                               exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(re_target.shape)

# Get necessary dimension sizes/values
nclasses     = len(eparams['thresholds'])+1
nlead        = len(leads)

# Get class count for later...
_,class_count=am.count_samples(None,target_class[:,25:])

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""     

# ----------------------------------------------------
# %% Retrieve a consistent sample if the option is set
# ----------------------------------------------------


if shuffle_trainsplit is False:
    print("Pre-selecting indices for consistency")
    output_sample = am.consistent_sample(re_data,target_class,leads,nsamples,leadmax=leads.max(),
                          nens=1,ntime=ntime,
                          shuffle_class=eparams['shuffle_class'],debug=False)
    
    target_indices,target_refids,predictor_indices,predictor_refids = output_sample
else:
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



#%% Load model weights 

# Get the model weights
modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)

# Get list of metric files
search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
flist  = glob.glob(search)
flist  = [f for f in flist if "of" not in f]
flist.sort()

print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))
#%% 



# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

# Print Message

# ------------------------
# 04. Loop by predictor...
# ------------------------
vt                    = time.time()
predictors            = re_data[[0],...] # Get selected predictor
total_acc_all         = np.zeros((nmodels,nlead))
class_acc_all         = np.zeros((nmodels,nlead,3)) # 

relevances_all        = []
predictor_all         = []
sampled_idx           = []

if shuffle_trainsplit:
    y_actual_all      = []
else:
    nsample_total     = len(target_indices)
    y_predicted_all   = np.zeros((nmodels,nlead,nsample_total))
    y_actual_all      = np.zeros((nlead,nsample_total))

# --------------------
# 05. Loop by runid...
# --------------------
for nr,runid in tqdm(enumerate(runids)):
    rt = time.time()
    
    # Preallocate Evaluation Metrics...
    # -----------------------
    # 07. Loop by Leadtime...
    # -----------------------
    outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,dataset_name)
    
    predictor_lead = []
    relevances_lead  = []
    for l,lead in enumerate(leads):
        
        if target_indices is None:
            # --------------------------
            # 08. Apply lead/lag to data
            # --------------------------
            # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
            X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # ----------------------
            # 09. Select samples
            # ----------------------
            if shuffle_trainsplit is False:
                if eparams['nsamples'] is None: # Default: nsamples = smallest class
                    threscount = np.zeros(nclasses)
                    for t in range(nclasses):
                        threscount[t] = len(np.where(y_class==t)[0])
                    eparams['nsamples'] = int(np.min(threscount))
                    print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
                y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
            else:
                print("Select the sample samples")
                shuffidx = sampled_idx[l-1] # This variable didnt exist before so I just added it... might remove this section later.
                y_class  = y_class[shuffidx,...]
                X        = X[shuffidx,...]
                am.count_samples(eparams['nsamples'],y_class)
            shuffidx = shuffidx.astype(int)
        else:
            #print("Using preselected indices")
            pred_indices = predictor_indices[l]
            nchan        = predictors.shape[0]
            y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
            X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
            X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
            shuffidx     = target_indices    
        
        # ----------------------
        # Flatten inputs for FNN
        # ----------------------
        if "FNN" in eparams['netname']:
            ndat,nchannels,nlat,nlon = X.shape
            inputsize                = nchannels*nlat*nlon
            outsize                  = nclasses
            X_in                     = X.reshape(ndat,inputsize)
        
        # -----------------------------
        # Place data into a data loader
        # -----------------------------
        # Convert to Tensors
        X_torch = torch.from_numpy(X_in.astype(np.float32))
        y_torch = torch.from_numpy(y_class.astype(np.compat.long))
        
        # Put into pytorch dataloaders
        test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
        
        
        #
        # Rebuild the model
        #
        # Get the models (now by leadtime)
        modweights = modweights_lead[l][nr]
        modlist    = modlist_lead[l][nr]
        
        # Rebuild the model
        pmodel = am.recreate_model(modelname,nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
        
        # Load the weights
        pmodel.load_state_dict(modweights)
        pmodel.eval()
        
        # ------------------------------------------------------
        # Test the model separately to get accuracy by class
        # ------------------------------------------------------
        y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                       checkgpu=checkgpu,debug=False)
        
        lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=False,verbose=False)
        
        
        
        total_acc_all[nr,l]   = lead_acc
        class_acc_all[nr,l,:] = class_acc
        y_predicted_all[nr,l,:]   = y_predicted
        y_actual_all[l,:] = y_actual
        
        #
        # Perform LRP
        #
        nsamples_lead = len(shuffidx)
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                          method=innmethod,
                                          beta=innbeta)
        model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
        model_prediction = model_prediction.detach().numpy().copy()
        sample_relevances = sample_relevances.detach().numpy().copy()
        if "FNN" in eparams['netname']:
            predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
            sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
        predictor_lead.append(predictor_test)
        relevances_lead.append(sample_relevances)
        
                
        
        # Clear some memory
        del pmodel
        torch.cuda.empty_cache()  # Save some memory
        
        #print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
        # End Lead Loop >>>
    predictor_all.append(predictor_lead)
    relevances_all.append(relevances_lead)
    #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
    # End Runid Loop >>>
#print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
# End Predictor Loop >>>

#print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))


#%% Perform LRP

#%% Prepare to do some visualization

# Load baselines
persleads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline(dataset_name,
                                                                        return_npfile=False,region="NAT",quantile=False,
                                                                        detrend=False,limit_samples=True,nsamples=None,repeat_calc=1)

# Load results from CESM1
#%%

fig,ax = plt.subplots(1,1)
for nr in range(nmodels):
    ax.plot(leads,total_acc_all[nr,:],alpha=0.1,color="g")
    
ax.plot(leads,total_acc_all.mean(0),color="green",label="CESM1-trained NN (SST)")
ax.plot(persleads,pers_total_acc,color="k",ls="dashed",label="Persistence Baseline")
ax.axhline([.33],color="gray",ls="dashed",lw=0.75,label="Random Chance Baseline")

ax.legend()
ax.grid(True,ls="dotted")
ax.set_xticks(persleads[::3])
ax.set_xlim([0,24])
ax.set_yticks(np.arange(0,1.25,0.25))
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.set_title("Total Accuracy (HadISST Testing, %i samples per class)" % (nsample_total/3))
# 
figname = "%sReanalysis_Test_%s_Total_Acc.png" % (figpath,dataset_name)
plt.savefig(figname,dpi=150)

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Visualize Accuracy by Class (SI_Draft02)
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
# Note: Took toggles from viz_acc_by_predictor on 2023.07.12

plotconf   = True
add_conf   = True
plotconf   = False #0.95
fill_alpha = 0.20
plotmax    = False # Set to True to plot maximum
maxmod     = 100 # Maximum networks to plot
mks        = 5 # Marker Size
xtks_lead  = np.arange(0,26,5)

fsz            = 14
fszt           = 12
fszb           = 16
fsz_axlbl      = 18

fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(18,4))
ii = 0
for c in range(3):
    ax = axs[c]
    
    # Calculate mean and stdev
    mu      = class_acc_all.mean(0)[...,c]
    sigma   = class_acc_all.std(0)[...,c]
    
    # Sort accuracy
    sortacc = np.sort(class_acc_all[...,c],0) # Sort along runs dimension
    idpct   = sortacc.shape[0] * plotconf
    lobnd   = np.floor(idpct).astype(int)
    hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
    
    if add_conf is False: # Just plot all models
        for nr in range(nmodels):
            ax.plot(leads,class_acc_all[nr,:,c],alpha=0.1,color=class_colors[c])
    else:
        if plotconf: # Plot confidence based on sorted accuracies
            ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=fill_alpha,color=class_colors[c],zorder=1,label="")
        else: # Just plot 1stdev
            ax.fill_between(leads,mu-sigma,mu+sigma,alpha=fill_alpha,color=class_colors[c],zorder=1)
    
    ax.plot(leads,mu,color=class_colors[c],label="CESM1-trained NN (SST)")
    ax.plot(persleads,pers_class_acc[:,c],color="k",ls="solid",label="Persistence Baseline")
    ax.axhline([.33],color="gray",ls="dashed",lw=2,label="Random Chance Baseline")
    if c == 0:
        ax.legend(fontsize=fsz)
    if c == 1:
        ax.set_ylabel("Accuracy")
    if c == 2:
        ax.set_xlabel("Prediction Lead (Years)")
    
    ax = pviz.format_acc_plot(leads,ax)
    ax.grid(True,ls="dotted")
    ax.set_xticks(xtks_lead,fontsize=fszt)
    ax.set_xlim([0,25])
    
    ax.set_yticks(np.arange(0,1.25,0.25))
    ax.set_ylim([-.1,1.1])
    ax.set_title("%s (N=%i)" % (classes[c],class_count[c]),fontsize=fszb)
    ax = viz.label_sp(ii,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_axlbl,x=0.01)
    ii+=1
figname = "%sReanalysis_Test_%s_Class_Acc_%s.png" % (figpath,dataset_name,expdir)
plt.savefig(figname,dpi=150)

#%% Visualize the HadiSST NASST Index

idx_by_class,count_by_class = am.count_samples(None,target_class)

class_str = "Class Count: AMV+ (%i) | Neutral (%i) | AMV- (%i)" % tuple(count_by_class)
timeaxis = np.arange(0,re_target.shape[1]) + 1870
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

ax.plot(timeaxis,re_target.squeeze(),color="k",lw=2.5)
ax.grid(True,ls="dashed")
ax.minorticks_on()

for th in thresholds_in:
    ax.axhline([th],color="k",ls="dashed")
ax.axhline([0],color="k",ls="solid",lw=0.5)
ax.set_xlim([timeaxis[0],timeaxis[-1]])
ax.set_title("HadISST NASST Index (1870-2022) \n%s" % (class_str))
plt.savefig("%sHadISST_NASST.png" %(figpath),dpi=150,bbox_inches='tight')

#%% Get correct indices for each class


# y_predicted_all = [runs,lead,sample]
# y_actual_all    = [lead,sample]

correct_mask = []
for l in range(len(leads)):
    lead = leads[l]
    y_preds   = y_predicted_all[:,l,:] # [runs lead sample]
    i_correct = (y_preds == y_actual_all[l,:][None,:]) # Which Predictions are correct
    correct_mask_lead = []
    for c in range(3):
        i_class = (y_actual_all[l,:] == c)
        correct_mask_lead.append(i_correct*i_class)
    correct_mask.append(correct_mask_lead)

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Visualize relevance maps for HadiSST (SI_Draft02)
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

relevances_all = np.array(relevances_all)
predictor_all = np.array(predictor_all)
nruns,nleads,nsamples_lead,nlat,nlon = relevances_all.shape

# Options
plotleads         = [25,20,10,5,0]
normalize_sample  = 2
hide_class_labels = False


plot_bbox        = [-80,0,0,60]

cmax  = 1
clvl = np.arange(-2.2,2.2,0.2)

fsz_title        = 20
fsz_axlbl        = 18
fsz_ticks        = 16

fig,axs  = plt.subplots(3,len(plotleads),constrained_layout=True,figsize=(18,10),
                        subplot_kw={'projection':ccrs.PlateCarree()})

ii = 0
for c in range(3):
    for l in range(len(plotleads)):
        
        ax = axs.flatten()[ii]
        lead  = plotleads[l]
        ilead = list(leads).index(lead) 
        
        # Axis Formatting
        blabel = [0,0,0,0]
        if c == 0:
            ax.set_title("%s-Year Lead" % (plotleads[l]),fontsize=fsz_title)
        if l == 0:
            blabel[0] = 1
            ax.text(-0.15, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
        ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
        ax = viz.label_sp(ii,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
            
        # Get correct predictions
        cmask = correct_mask[l][c].flatten()
        relevances_in = relevances_all[:,ilead,:,:,:]
        newshape      = (np.prod(relevances_in.shape[:2]),) + (nlat,nlon)
        # Apprently using cmask[:,...] brocasts, while cmask[:,None,None] doesn't
        relevances_sel = relevances_in.reshape(newshape)[cmask[:,...]] # [Samples x Lat x Lon]
        
        predictor_in   = predictor_all[:,ilead,:,:,:]
        predictor_sel = predictor_in.reshape(newshape)[cmask[:,...]] # [Samples x Lat x Lon]
        if normalize_sample == 1:
            relevances_sel = relevances_sel / np.abs(relevances_sel.max(0))[None,...]
        
        
        # Plot the results
        plotrel = relevances_sel.mean(0)
        plotvar = predictor_sel.mean(0)
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
            
        # Set Land Points to Zero
        plotrel[plotrel==0] = np.nan
        plotvar[plotrel==0] = np.nan
        
            
        # Do the plotting
        pcm=ax.pcolormesh(re_lon,re_lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        cl = ax.contour(re_lon,re_lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        ax.clabel(cl,clvl[::2])
        
            
        ii+=1
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
cb.ax.tick_params(labelsize=fsz_ticks)


savename = "%sHadISSTClassComposites_LRP_%s_normalize%i_Outline.png" % (figpath,expdir,normalize_sample)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)

#%% Make a scatterplot of the event distribution and 

imodel = 6
ilead  = 25
msize  = 100
timeaxis = np.arange(0,re_target.shape[1]) + 1870

for imodel in range(50):
    # Select the model
    y_predicted_in = y_predicted_all[imodel,ilead,:]
    y_actual_in    = y_actual_all[ilead,:]
    re_target_in   = re_target[:,leads[ilead]:].squeeze()
    id_correct     = (y_predicted_in == y_actual_in)
    
    
    timeaxis_in = np.arange(leads[ilead],re_target.shape[1]) + 1870
    
    
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Plot the amv classes
    for c in range(3):
        
        # Get the id for the class
        id_class = (y_actual_in == c)
        
        id_right = id_class * id_correct
        id_wrong = id_class * ~id_correct
        
        # Plot the correct ones
        ax.scatter(timeaxis_in[id_right],re_target_in[id_right],s=msize,marker="o",color=class_colors[c],facecolors="None")
        ax.scatter(timeaxis_in[id_wrong],re_target_in[id_wrong],s=msize,marker="x",color=class_colors[c])
        
    
    # Plot the actual AMV Index
    #ax.plot(timeaxis,re_target.squeeze(),color="k",lw=0.75,zorder=-9)
    ax.grid(True,ls="dashed")
    ax.minorticks_on()
    
    # Plot the Thresholds
    for th in thresholds_in:
        ax.axhline([th],color="k",ls="dashed")
    ax.axhline([0],color="k",ls="solid",lw=0.5)
    ax.set_xlim([timeaxis[0],timeaxis[-1]])
    
    class_str = "Class Acc: AMV+ (%.2f), Neutral (%.2f), AMV- (%.2f)" % (class_acc_all[imodel,ilead,0],
                                                                         class_acc_all[imodel,ilead,1],
                                                                         class_acc_all[imodel,ilead,2])
    ax.set_title("HadISST NASST Index and Prediction Results (1870-2022) \nNetwork #%i, Lead = %i years \n %s" % (imodel+1,leads[ilead],class_str))
    plt.savefig("%sHadISST_NASST_lead%02i_imodel%03i.png" %(figpath,leads[ilead],imodel,),dpi=150,bbox_inches='tight')




#%% Function version of above
def plot_scatter_predictions(imodel,ilead,y_predicted_all,y_actual_all,re_target,class_acc_all,msize=100,
                             figsize=(12,4),class_colors=('salmon', 'gray', 'cornflowerblue')):
    
    
    # Select the model
    y_predicted_in = y_predicted_all[imodel,ilead,:]
    y_actual_in    = y_actual_all[ilead,:]
    re_target_in   = re_target[:,leads[ilead]:].squeeze()
    id_correct     = (y_predicted_in == y_actual_in)
    
    timeaxis_in = np.arange(leads[ilead],re_target.shape[1]) + 1870
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Plot the amv classes
    for c in range(3):
        
        # Get the id for the class
        id_class = (y_actual_in == c)
        
        id_right = id_class * id_correct
        id_wrong = id_class * ~id_correct
        
        # Plot the correct ones
        ax.scatter(timeaxis_in[id_right],re_target_in[id_right],s=msize,marker="o",color=class_colors[c],facecolors="None")
        ax.scatter(timeaxis_in[id_wrong],re_target_in[id_wrong],s=msize,marker="x",color=class_colors[c])
        
    
    # Plot the actual AMV Index
    #ax.plot(timeaxis,re_target.squeeze(),color="k",lw=0.75,zorder=-9)
    ax.grid(True,ls="dashed")
    ax.minorticks_on()
    
    # Plot the Thresholds
    for th in thresholds_in:
        ax.axhline([th],color="k",ls="dashed")
    ax.axhline([0],color="k",ls="solid",lw=0.5)
    ax.set_xlim([timeaxis[0],timeaxis[-1]])
    
    class_str = "Class Acc: AMV+ (%.2f), Neutral (%.2f), AMV- (%.2f)" % (class_acc_all[imodel,ilead,0],
                                                                         class_acc_all[imodel,ilead,1],
                                                                         class_acc_all[imodel,ilead,2])
    return fig,ax

    
#%% MAKE A PLOT OF ABOVE, BUT WITH THE BEST performing model

ilead   = -1
id_best = total_acc_all[:,ilead].argmax()


fig,ax = plot_scatter_predictions(id_best,ilead,y_predicted_all,y_actual_all,re_target,class_acc_all,msize=100,
                             figsize=(12,4))

ax.set_ylim([-1.5,1.5])
ax.set_xlim([1890,2025])
ax.set_title("HadISST NASST Index and Prediction Results (1870-2022) \nNetwork #%i, Lead = %i years \n %s" % (id_best+1,leads[ilead],class_str))
plt.savefig("%sHadISST_NASST_lead%02i_imodel%03i.png" %(figpath,leads[ilead],imodel,),dpi=150,bbox_inches='tight')



#%% Make a histogram


# Visualize prediction count by year

# Select the model
#y_predicted_in = y_predicted_all[imodel,ilead,:]
#y_actual_in    = y_actual_all[ilead,:]
#re_target_in   = re_target[:,leads[ilead]:].squeeze()
#id_correct     = (y_predicted_in == y_actual_in)

count_by_year = np.zeros((ntime-leads[-1],nclasses))
timeaxis_in   = np.arange(leads[ilead],re_target.shape[1])

# Assumes leads are not shuffled
for y in range(ntime-leads[ilead]):
    y_pred_year = y_predicted_all[...,y]
    
    for c in range(3):
        
        count_by_year[y,c] = (y_pred_year == c).sum()
        
        

    

#%% General Settings

darkmode=False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    dfcol_r = "k"
else:
    plt.style.use('default')
    dfcol = "k"
    dfcol_r = "w"
    
# Copied from below
selected_leads      = leads.copy() #[0,6,12,18,24]
nleads_sel          = len(selected_leads)

    
#%% I was up to here. Make the barplotfor Draft 1


fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

for c in range(3):
    label = classes[c]
    #label = "%s (Test Acc = %.2f" % (classes[c],class_acc[c]*100)+"%)"
    
    totalnum  = count_by_year.sum(1)[0]
    plotcount = count_by_year[:,c]
    plotcount = plotcount/totalnum
    
    plotcountbot = count_by_year[:,:c].sum(1)
    plotcountbot = plotcountbot/totalnum
    
    
    
    ax.bar(timeaxis_in+1870,plotcount,bottom=plotcountbot,
           label=label,color=class_colors[c],alpha=0.75,edgecolor=dfcol_r)

ax.set_ylabel("Frequency of Predicted Class")
ax.set_xlabel("Year")
ax.legend()
ax.minorticks_on()
ax.grid(True,ls="dotted")
ax.set_xlim([1880,2025])
ax.set_ylim([0,1])
#ax.set_ylim([0,450])

ax2 = ax.twinx()
ax2.plot(timeaxis,re_target.squeeze(),color=dfcol,label="HadISST NASST Index")
ax2.set_ylabel("NASST Index ($\degree C$)")
ax2.set_ylim([-1.3,1.3])
for th in thresholds_in:
    ax2.axhline([th],color=dfcol,ls="dashed")
ax2.axhline([0],color=dfcol,ls="solid",lw=0.5)
plt.savefig("%sHadISST_Prediction_Count_AllLeads.png"%figpath,dpi=150,bbox_inches="tight",transparent=True)


#%%


#%% Try the above, but get prediction count for selected leadtimes
# Q : Is there a systematic shift towards the selected leadtimes?
selected_leads      = leads.copy() #[0,6,12,18,24]
nleads_sel          = len(selected_leads)

count_by_year_leads = np.zeros((ntime-leads[-1],nclasses,nleads_sel))

# Assumes leads are not shuffled
for y in range(ntime-leads[ilead]):
    
    for ll in range(nleads_sel):
        sel_lead_index = list(leads).index(selected_leads[ll])
        y_pred_year = y_predicted_all[...,sel_lead_index,y]
    
        for c in range(3):
            
            count_by_year_leads[y,c,ll] = (y_pred_year == c).sum()


#%% 
fig,axs       = plt.subplots(3,1,constrained_layout=True,figsize=(16,8))



lead_colors = ["lightsteelblue","cornflowerblue","royalblue","mediumblue","midnightblue","gray","black","red","orange"]
for c in range(3):
    ax = axs[c]
    
    for ll in range(nleads_sel):
        ax.plot(timeaxis_in+1870,count_by_year_leads[:,c,ll],label="%02i-yr Lead" % selected_leads[ll],lw=1.5,c=lead_colors[ll])
        
    if c == 0:
        ax.legend()
    ax.set_title(classes[c])
    
    ax.set_xlabel("Year")
    ax.minorticks_on()
    ax.grid(True,ls="dashed")
    
    # label = "%s (Test Acc = %.2f" % (classes[c],class_acc[c]*100)+"%)"
    # ax.bar(timeaxis_in+1870,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
    #        label=label,color=class_colors[c],alpha=0.75,edgecolor="k")
    
    ax.set_ylabel("Predicted Class Count")

plt.savefig("%sHadISST_Class_Prediction_Frequency_byYear.png"%(figpath),dpi=150,bbox_inches="tight")


#%% Remake barplot. but for separately the selected leadtimes

def make_count_barplot(count_by_year,lead,re_target,leadmax=24,classes=['AMV+', 'Neutral', 'AMV-'],
                       class_colors=('salmon', 'gray', 'cornflowerblue')
                       ):
    
    timeaxis      = np.arange(0,len(re_target.squeeze()))
    timeaxis_in   = np.arange(leadmax,re_target.shape[1])
    
    maxcount = count_by_year.sum(-1)
    #print(maxcount)
    fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

    for c in range(3):
        label = classes[c]
        plotcount = count_by_year/maxcount[:,None]
        ax.bar(timeaxis_in+1870,plotcount[:,c],bottom=plotcount[:,:c].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor=dfcol_r)
    
    ax.set_ylabel("Frequency of Predicted Class")
    ax.set_xlabel("Year")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True,ls="dotted")
    ax.set_xlim([1880,2025])
    ax.set_ylim([0,1])
    
    ax2 = ax.twinx()
    ax2.plot(timeaxis+1870,re_target.squeeze(),color=dfcol,label="HadISST NASST Index")
    ax2.set_ylabel("NASST Index ($\degree C$)")
    ax2.set_ylim([-1.3,1.3])
    for th in thresholds_in:
        ax2.axhline([th],color=dfcol,ls="dashed")
    ax2.axhline([0],color=dfcol,ls="solid",lw=0.5)
    axs = [ax,ax2]
    return fig,axs

lagcorr = []
for ll in range(nleads_sel):
    lead = selected_leads[ll]
    lagcorr.append(np.corrcoef(re_target[0,:(ntime-lead)],re_target[0,lead:])[0,1])

for ll in range(nleads_sel):
    lead = selected_leads[ll]
    ilead = list(leads).index(lead)
    
    # Plot histogram
    fig,axs = make_count_barplot(count_by_year_leads[:,:,ll],lead,re_target,leadmax=25)
    axs[0].set_title("Histogram for Lead %i Years" % (lead))
    plt.savefig("%sHadISST_Prediction_Count_Lead%02i.png"% (figpath,lead),dpi=150,bbox_inches="tight",transparent=True)
    
    # Plot lagocorr with moving bar
    # fig,ax = plt.subplots(1,1)
    # ax.set_ylim([0,1])
    # ax.plot(selected_leads,lagcorr,marker="x")
    # ax.axvline(lead,color="k",ls='dashed')
    # ax.set_xticks(selected_leads)
    # ax.set_title("Lag Correlation of NASST Index")
    # plt.savefig("%sHadISST_Lag_Correlation_lead%02i.png" % (figpath,lead),dpi=150)
    # plt.show()

# <0><0><0><0><0><0> <0><0><0><0><0><0> <0><0><0><0><0><0> <0><0><0><0><0><0>
#%% Group Barplot by years (interannual, decadal, multidecadal)
# Predict AMV Draft 03

# count_by_year_leads # [year x class x lead]
interann_count      = count_by_year_leads[:,:,np.arange(1,10)].sum(2) # year x class
decadal_count       = count_by_year_leads[:,:,np.arange(10,20)].sum(2)
multidecadal_count  = count_by_year_leads[:,:,np.arange(20,26)].sum(2)


counts_in           = [interann_count,decadal_count,multidecadal_count]
count_labels        = ["Interannual (1-9 years)","Decadal (10-19 years)", "Multidecadal (20-26 years)"]


# Plot Settings
leadmax = 25
fsz_axlbl = 20
fsz_ticks = 14


fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,10))

for ii in range(3):
    
    
    ax = axs[ii]

    # (Copied from function above)
    # Set up timeaxis
    timeaxis      = np.arange(0,len(re_target.squeeze()))
    timeaxis_in   = np.arange(leadmax,re_target.shape[1])
    
    # Plot the counts
    plotcount = counts_in[ii]
    maxcount  = plotcount.sum(-1)
    
    for c in range(3):
        label = classes[c]
        plotcount = plotcount
        print(plotcount[:,c][0])
        print(plotcount[:,:c].sum(1)[0])
        ax.bar(timeaxis_in+1870,plotcount[:,c]/maxcount*100,bottom=plotcount[:,:c].sum(1)/maxcount*100,
               label=label,color=class_colors[c],alpha=0.75,edgecolor=dfcol_r)
    
    # Label and set ticks
    if ii == 0:
        ax.legend(loc='lower right')
    if ii == 1:
        ax.set_ylabel("Frequency of Predicted Class (%)",fontsize=fsz_axlbl)
    ax.minorticks_on()
    ax.grid(True,ls="dotted")
    ax.set_xlim([1890,2025])
    ax.set_ylim([0,110])
    if ii == 2:
        ax.set_xlabel("Year",fontsize=fsz_axlbl)

    
    # Plot NASST Index on Separate Axis
    ax2 = ax.twinx()
    ax2.plot(timeaxis+1870,re_target.squeeze(),color=dfcol,label="HadISST NASST Index")
    if ii == 1:
        ax2.set_ylabel("NASST Index ($\degree C$)",fontsize=fsz_axlbl)
    ax2.set_ylim([-1.3,1.3])
    for th in thresholds_in:
        ax2.axhline([th],color=dfcol,ls="dashed")
    ax2.axhline([0],color=dfcol,ls="solid",lw=0.5)
    ax = viz.label_sp(ii,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_axlbl,labelstyle="%s) "+count_labels[ii])
    #axs = [ax,ax2]
    
    # Final adjustment of font sizes
    ax.tick_params(labelsize=fsz_ticks)
    ax2.tick_params(labelsize=fsz_ticks)


figname = "%sHadISST_Prediction_Count_Lead_TimeSplit_%s.png"% (figpath,expdir)
plt.savefig(figname,dpi=150,bbox_inches="tight",transparent=False)

                                     




#%% WIP BELOW -----------------------------------------------------------------





#%%

ax.set_title(title)


plot_mode = 0

for plot_mode in range(2):
    ax = axs[plot_mode]
    ax = format_axis(ax,x=timeaxis)
    if plot_mode == 0:
        title = "Actual Class"
    elif plot_mode == 1:
        title = "Predicted Class"
    testc = np.arange(0,3)
    for c in range(3):
        label = "%s (Test Acc = %.2f" % (class_names[c],class_acc[c]*100)+"%)"
        if debug:
            print("For c %i, sum of prior values is %s" % (c,testc[:c]))
        ax.bar(timeaxis,count_by_year[:,c,plot_mode],bottom=count_by_year[:,:c,plot_mode].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor="k")
    ax.set_title(title)
    ax.set_ylim([0,10])
    if plot_mode == 0:
        ax.legend()
plt.suptitle("AMV Class Distribution by Year (%s) \n %s" % (modelname,exp_titlestr))
if savefig:
    plt.savefig("%sClass_Distr_byYear_%s_lead%02i_nepochs%02i.png" % (figpath,varnames[v],lead,epoch_axis[-1]),dpi=150)


#%%

