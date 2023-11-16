#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:42:42 2023

Compute Test Metrics + Test_LRP_Sensitivity

 - Test Accuracy
 - Loss by Epoch (Test)
 - Test Sensitivity to LRP Parameters (by Sample)
 
- For a given experiment and variable, compute the test metrics
- Save to an output file...

 Copied from test_cesm_witheld on Tue Jun 13 11:20AM

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

#%% Looping for each model, obtain the relevances for all samples and networks...

innmethod = 'e-rule'
inneps    = 1e-6
innexp    = 1

predictor_all  = []
relevances_all = []
ypred_all = []
ylab_all  = []
for nr in tqdm(range(nmodels)):
    
    runid = runids[nr]
    
    # =====================
    # II. Rebuild the model
    # =====================
    # Get the models (now by leadtime)
    modweights = modweights_lead[l][nr]
    modlist    = modlist_lead[l][nr]
    
    # Rebuild the model
    pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
    
    # Load the weights
    pmodel.load_state_dict(modweights)
    pmodel.eval()
            
    # =======================================================
    # III. Test the model separately to get accuracy by class
    # =======================================================
    y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                   checkgpu=checkgpu,debug=False)
    
    # =======================================================
    # III. Test the model separately to get accuracy by class
    # =======================================================
    y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                   checkgpu=checkgpu,debug=False)
    lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
    
    
    # ===========================
    # IV. Perform LRP
    # ===========================
    nsamples_lead = len(y_actual)
    inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                      method=innmethod,
                                      beta=innbeta,
                                      epsilon=inneps)
    model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
    model_prediction                    = model_prediction.detach().numpy().copy()
    sample_relevances                   = sample_relevances.detach().numpy().copy()
    if "FNN" in eparams['netname']:
        predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
        sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
    else: # Assume CNN
        predictor_test    = X_torch.detach().numpy().copy().squeeze() # [test samples x lat x lon]
        sample_relevances = sample_relevances.squeeze()
    
    
    # Save Variables
    if nr == 0:
        predictor_all.append(predictor_test) # Predictors are the same across model runs
    relevances_all.append(sample_relevances)
    ypred_all.append(y_predicted)
    ylab_all.append(y_actual)
    
    del pmodel

predictor_all          = np.array(predictor_all)
relevances_all         = np.array(relevances_all)
predictions_model      = np.array(ypred_all)
targets_all            = np.array(ylab_all)


#%% Calculate Event based Statistics
# ----------------------------------
nmodels = targets_all.shape[0]
nevents = X_torch.shape[0]
correct = targets_all == predictions_model

acc_bymodel = correct.sum(1) / nevents
acc_byevent = correct.sum(0) / nmodels

fig,axs = plt.subplots(2,1,constrained_layout=True)
ax = axs[0]
ax.bar(np.arange(nmodels),acc_bymodel)
ax.set_title("Test Accuracy by Model\n Range: [%.2f (%i) to %.2f (%i)]" % (acc_bymodel.min(),
                                                                        acc_bymodel.argmin(),
                                                                        acc_bymodel.max(),
                                                                        acc_bymodel.argmax()
                                                                        ))

ax = axs[1]
ax.bar(np.arange(nevents),acc_byevent)
ax.set_title("Test Accuracy By Event\n Range: [%.2f (%i) to %.2f (%i)]" % (acc_byevent.min(),
                                                                        acc_byevent.argmin(),
                                                                        acc_byevent.max(),
                                                                        acc_byevent.argmax()
                                                                        ))
plt.suptitle("Prediction %s, Lead %02i" % (varnames[v],leads[l]))


#%% Look at events that are all correct or all wrong
id_allcorrect = (acc_byevent == 1)
id_allwrong   = (acc_byevent == 0)
print("%i events out of %i were predicted CORRECTLY by all %i Networks" % (id_allcorrect.sum(),nevents,nmodels))
print("%i events out of %i were predicted WRONG by all %i Networks" % (id_allwrong.sum(),nevents,nmodels))


icorr_sample = []
iwrong_sample   = []
for iclass in range(3):
    # Retrieve [actual indices][indices corresponding to class]
    icorr  = np.where(id_allcorrect)[0][np.where(targets_all[1,id_allcorrect]==iclass)[0]]
    iwrong = np.where(id_allwrong)[0][np.where(targets_all[1,id_allwrong]==iclass)[0]]
    icorr_sample.append(icorr)
    iwrong_sample.append(iwrong)
    
    print("%i (all correct) were of class %i \t: %s" % ((targets_all[1,id_allcorrect]==iclass).sum(),iclass,icorr))
    print("%i (all wrong) were of class %i \t: %s" % ((targets_all[1,id_allwrong]==iclass).sum(),iclass,iwrong))

#%% Lets examine LRP Sensitivity for a given event

isample    = icorr_sample[0][0] # Select a sample index
iclass     = int(targets_all[0,isample])
imodel     = 1 # Select a model index



# Load Predictor and Model
X_in       = X_torch[[isample],:]
modweights = modweights_lead[l][imodel]
pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
pmodel.load_state_dict(modweights)
pmodel.eval()

def lrp_wrap(pmodel,X_in,innexp,innmethod,innbeta,inneps):
    # Perform LRP
    inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                              method=innmethod,
                                              beta=innbeta,
                                              epsilon=inneps)
    model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_in)
    model_prediction                    = model_prediction.detach().numpy().copy()
    sample_relevances                   = sample_relevances.detach().numpy().copy()
    
    # Assume FNN
    predictor_test    = X_in.detach().numpy().copy().reshape(1,nlat,nlon) # [1,lat,lon] 
    sample_relevances = sample_relevances.reshape(1,nlat,nlon) # [1,lat,lon] 
    return predictor_test,sample_relevances

# ------------------------------------------
#%% Test (1): e-rule and b-rule, vary innexp
# -----------------------------------------
innexps    = [1,2,3,4]
innmethods = ['e-rule','b-rule']
inneps     = 1e-2
innbeta    = 0.1

relevances_exptest = np.zeros((2,4,nlat,nlon)) # [Method, Exp, Lat, Lon]

for imeth in range(2):
    innmethod = innmethods[imeth]
    for iexp in range(4):
        innexp = innexps[iexp]
        predictor_test,sample_relevance = lrp_wrap(pmodel,X_in,innexp,innmethod,innbeta,inneps)
        relevances_exptest[imeth,iexp,:,:] = sample_relevance
        
# Make a quick landmask
landmask = predictor_test[0,...].copy()
landmask[landmask==0.] = np.nan
landmask[~np.isnan(landmask)] = 1
plt.pcolormesh(landmask)


#%% Sample plot
lon       = load_dict['lon']
lat       = load_dict['lat']
classes   = pparams.classes
fsz_title = 16
fsz_ticks = 14
vlms      = []#None
norm      = True

predlvls        = np.arange(-2,2.5,0.5)
innmethod_fancy = [r"$\epsilon$-rule",r"$\beta$-rule"]
fig,axs         = plt.subplots(2,4,figsize=(18,9),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ia = 0

for imeth in range(2):
    innmethod = innmethods[imeth]
    for iexp in range(4):
        innexp = innexps[iexp]
        
        # Axis Selection and Labeling
        ax     = axs[imeth,iexp]
        blabel = [0,0,0,0]
        if imeth == 1:
            blabel[-1] = 1
        else:
            ax.set_title("p = %.2f" % innexp,fontsize=fsz_title)
        if iexp == 0:
            blabel[0] = 0
            ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
        ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
        
        # Plotting
        plotrel = relevances_exptest[imeth,iexp,:,:] * landmask
        plotvar = predictor_test.squeeze() * landmask
        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
        ax.clabel(cl,fontsize=fsz_ticks)
        if norm is False:
            pcm     = ax.pcolormesh(lon,lat,plotrel)
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
        else:
            normfactor = np.nanmax(np.abs(plotrel))
            plotrel = plotrel / normfactor
            pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
            label = "max=%.0e" % normfactor
            viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
        ia += 1
        
if norm:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_title)

title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model: %03i" % (varnames[v],leads[l],isample,classes[iclass],imodel)
fnstr    = "%s_lead%02i_sample%03i_class%s_model%03i_norm%i" % (varnames[v],leads[l],isample,iclass,imodel,norm)
title_l2 = r"LRP Params: $\epsilon$=%.2e, $\beta$=%.2f" % (inneps,innbeta)
plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)

savename = "%s%s.png" % (figpath,fnstr)
print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Test(1) Try for a bunch of different models

make_figure_bymodel = False
saverel = True

relevances_exptest_all = []

for imodel in tqdm(range(100)):
    
    # Chose the model ---------------------------------------------------------
    modweights = modweights_lead[l][imodel]
    pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
    pmodel.load_state_dict(modweights)
    pmodel.eval()
    
    # Calculate Relevances for different parameters (copied from above) -------
    relevances_exptest = np.zeros((2,4,nlat,nlon)) # [Method, Exp, Lat, Lon]
    for imeth in range(2):
        innmethod = innmethods[imeth]
        for iexp in range(4):
            innexp = innexps[iexp]
            predictor_test,sample_relevance = lrp_wrap(pmodel,X_in,innexp,innmethod,innbeta,inneps)
            relevances_exptest[imeth,iexp,:,:] = sample_relevance
    relevances_exptest_all.append(relevances_exptest)
    
    # Copy Relevance Figures --------------------------------------------------
    if make_figure_bymodel:
        fig,axs         = plt.subplots(2,4,figsize=(18,9),
                               subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
        ia = 0
        for imeth in range(2):
            innmethod = innmethods[imeth]
            for iexp in range(4):
                innexp = innexps[iexp]
                
                # Axis Selection and Labeling
                ax     = axs[imeth,iexp]
                blabel = [0,0,0,0]
                if imeth == 1:
                    blabel[-1] = 1
                else:
                    ax.set_title("p = %.2f" % innexp,fontsize=fsz_title)
                if iexp == 0:
                    blabel[0] = 0
                    ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                            rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
                ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
                
                # Plotting
                plotrel = relevances_exptest[imeth,iexp,:,:] * landmask
                plotvar = predictor_test.squeeze() * landmask
                cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
                ax.clabel(cl,fontsize=fsz_ticks)
                if norm is False:
                    pcm     = ax.pcolormesh(lon,lat,plotrel)
                    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
                else:
                    normfactor = np.nanmax(np.abs(plotrel))
                    plotrel = plotrel / normfactor
                    pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
                    label = "max=%.0e" % normfactor
                    viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
                ia += 1
    
        if norm:
            cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
            cb.set_label("Normalized Relevance",fontsize=fsz_title)
    
        title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model: %03i (%.2f)" % (varnames[v],leads[l],isample,classes[iclass],imodel,acc_bymodel[imodel])
        fnstr    = "%s_lead%02i_sample%03i_class%s_model%03i_norm%i" % (varnames[v],leads[l],isample,iclass,imodel,norm)
        title_l2 = r"LRP Params: $\epsilon$=%.2e, $\beta$=%.2f" % (inneps,innbeta)
        plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)
    
        savename = "%s%s.png" % (figpath,fnstr)
        #print(savename)
        plt.savefig(savename,dpi=150,bbox_inches='tight')

relevances_exptest_all =np.array(relevances_exptest_all) # [Network, Method, Exp, Lat, Lon]

if saverel:
    savename = "%sRelevances_ExpTest_%s_%s_sample%02i_class%i_allmodels.npz" % (figpath,varnames[v],leads[l],isample,iclass)
    np.savez(savename,
             **{'relevances':relevances_exptest_all,
                'innexps' : innexps,
                'innmethods': innmethods,
                'lat':lat,
                'lon':lon,
                'runids':runids,
                'inneps':inneps,
                'innbeta':innbeta,
                 },allow_pickle=True
             )

#%% Plot the composite relevance for that sample across all models 

fig,axs         = plt.subplots(2,4,figsize=(18,9),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
ia = 0
for imeth in range(2):
    innmethod = innmethods[imeth]
    for iexp in range(4):
        innexp = innexps[iexp]
        
        # Axis Selection and Labeling
        ax     = axs[imeth,iexp]
        blabel = [0,0,0,0]
        if imeth == 1:
            blabel[-1] = 1
        else:
            ax.set_title("p = %.2f" % innexp,fontsize=fsz_title)
        if iexp == 0:
            blabel[0] = 0
            ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
        ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
        
        # Plotting
        plotrel = relevances_exptest_all.mean(0)[imeth,iexp,:,:] * landmask
        plotvar = predictor_test.squeeze() * landmask
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
            viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
        ia += 1

if norm:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_title)

title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model Composite" % (varnames[v],leads[l],isample,classes[iclass])
fnstr    = "%s_lead%02i_sample%03i_class%s_modelALL_norm%i" % (varnames[v],leads[l],isample,iclass,norm)
title_l2 = r"LRP Params: $\epsilon$=%.2e, $\beta$=%.2f" % (inneps,innbeta)
plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)

savename = "%s%s.png" % (figpath,fnstr)
#print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Test (2) : Epsilon

innexp     = 1
innmethods = ['e-rule','b-rule']
innepss    = [1e-1,1e-2,1e-4,1e-8]
innbeta    = 0.1

saverel = True
make_figure_bymodel=True

relevances_epstest_all = []
for imodel in tqdm(range(100)):
    
    # Chose the model ---------------------------------------------------------
    modweights = modweights_lead[l][imodel]
    pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
    pmodel.load_state_dict(modweights)
    pmodel.eval()
    
    # Calculate Relevances for different parameters (copied from above) -------
    relevances_exptest = np.zeros((2,4,nlat,nlon)) # [Method, Epsilon, Lat, Lon]
    for imeth in range(2):
        innmethod = innmethods[imeth]
        for iexp in range(4):
            inneps = innepss[iexp]
            predictor_test,sample_relevance = lrp_wrap(pmodel,X_in,innexp,innmethod,innbeta,inneps)
            relevances_exptest[imeth,iexp,:,:] = sample_relevance
    relevances_epstest_all.append(relevances_exptest)
    
    # Copy Relevance Figures --------------------------------------------------
    if make_figure_bymodel:
        fig,axs         = plt.subplots(2,4,figsize=(18,9),
                               subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
        ia = 0
        for imeth in range(2):
            innmethod = innmethods[imeth]
            for iexp in range(4):
                inneps = innepss[iexp]
                
                # Axis Selection and Labeling
                ax     = axs[imeth,iexp]
                blabel = [0,0,0,0]
                if imeth == 1:
                    blabel[-1] = 1
                else:
                    ax.set_title("$\epsilon$ = %.0e" % inneps,fontsize=fsz_title)
                if iexp == 0:
                    blabel[0] = 0
                    ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                            rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
                ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
                
                # Plotting
                plotrel = relevances_exptest[imeth,iexp,:,:] * landmask
                plotvar = predictor_test.squeeze() * landmask
                cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
                ax.clabel(cl,fontsize=fsz_ticks)
                if norm is False:
                    pcm     = ax.pcolormesh(lon,lat,plotrel)
                    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
                else:
                    normfactor = np.nanmax(np.abs(plotrel))
                    plotrel = plotrel / normfactor
                    pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
                    label = "max=%.0e" % normfactor
                    viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
                ia += 1
    
        if norm:
            cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
            cb.set_label("Normalized Relevance",fontsize=fsz_title)
    
        title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model: %03i (%.2f)" % (varnames[v],leads[l],isample,classes[iclass],imodel,acc_bymodel[imodel])
        fnstr    = "EpsTest_%s_lead%02i_sample%03i_class%s_model%03i_norm%i" % (varnames[v],leads[l],isample,iclass,imodel,norm)
        title_l2 = r"LRP Params: $Exponent$=%.2f, $\beta$=%.2f" % (innexp,innbeta)
        plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)
        
        savename = "%s%s.png" % (figpath,fnstr)
        #print(savename)
        plt.savefig(savename,dpi=150,bbox_inches='tight')

relevances_epstest_all =np.array(relevances_epstest_all) # [Network, Method, Exp, Lat, Lon]

if saverel:
    savename = "%sRelevances_EspTest_%s_%s_sample%02i_class%i_allmodels.npz" % (figpath,varnames[v],leads[l],isample,iclass)
    np.savez(savename,
             **{'relevances':relevances_epstest_all,
                'innepss' : innepss,
                'innmethods': innmethods,
                'lat':lat,
                'lon':lon,
                'runids':runids,
                'innexp':innexp,
                'innbeta':innbeta,
                 },allow_pickle=True
             )
    
#%% Make a composite plot for epsilon

fig,axs         = plt.subplots(2,4,figsize=(18,9),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
ia = 0
for imeth in range(2):
    innmethod = innmethods[imeth]
    for iexp in range(4):
        inneps = innepss[iexp]
        
        # Axis Selection and Labeling
        ax     = axs[imeth,iexp]
        blabel = [0,0,0,0]
        if imeth == 1:
            blabel[-1] = 1
        else:
            ax.set_title("$\epsilon$ = %.0e" % inneps,fontsize=fsz_title)
        if iexp == 0:
            blabel[0] = 0
            ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
        ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
        
        # Plotting
        plotrel = relevances_epstest_all.mean(0)[imeth,iexp,:,:] * landmask
        plotvar = predictor_test.squeeze() * landmask
        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
        ax.clabel(cl,fontsize=fsz_ticks)
        if norm is False:
            pcm     = ax.pcolormesh(lon,lat,plotrel)
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
        else:
            normfactor = np.nanmax(np.abs(plotrel))
            plotrel = plotrel / normfactor
            pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
            label = "max=%.0e" % normfactor
            viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
        ia += 1

if norm:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_title)

title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model ALL (%.2f)" % (varnames[v],leads[l],isample,classes[iclass],acc_bymodel.mean())
fnstr    = "EpsTest_%s_lead%02i_sample%03i_class%s_modelALL_norm%i" % (varnames[v],leads[l],isample,iclass,norm)
title_l2 = r"LRP Params: $Exponent$=%.2f, $\beta$=%.2f" % (innexp,innbeta)
plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)

savename = "%s%s.png" % (figpath,fnstr)
#print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Test (3) : Beta


innexp     = 1
innmethods = ['e-rule','b-rule']
inneps     = 1e-6

innbetas   = [0.1,0.5,0.75,1]

saverel    = True
make_figure_bymodel=True

relevances_betatest_all = []
for imodel in tqdm(range(100)):
    
    # Chose the model ---------------------------------------------------------
    modweights = modweights_lead[l][imodel]
    pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
    pmodel.load_state_dict(modweights)
    pmodel.eval()
    
    # Calculate Relevances for different parameters (copied from above) -------
    relevances_exptest = np.zeros((2,4,nlat,nlon)) # [Method, Epsilon, Lat, Lon]
    for imeth in range(2):
        innmethod = innmethods[imeth]
        for iexp in range(4):
            innbeta = innbetas[iexp]
            predictor_test,sample_relevance = lrp_wrap(pmodel,X_in,innexp,innmethod,innbeta,inneps)
            relevances_exptest[imeth,iexp,:,:] = sample_relevance
    relevances_betatest_all.append(relevances_exptest)
    
    # Copy Relevance Figures --------------------------------------------------
    if make_figure_bymodel:
        fig,axs         = plt.subplots(2,4,figsize=(18,9),
                               subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
        ia = 0
        for imeth in range(2):
            innmethod = innmethods[imeth]
            for iexp in range(4):
                innbeta = innbetas[iexp]
                
                # Axis Selection and Labeling
                ax     = axs[imeth,iexp]
                blabel = [0,0,0,0]
                if imeth == 1:
                    blabel[-1] = 1
                else:
                    ax.set_title(r"$\beta$ = %.0e" % innbeta,fontsize=fsz_title)
                if iexp == 0:
                    blabel[0] = 0
                    ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                            rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
                ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
                
                # Plotting
                plotrel = relevances_exptest[imeth,iexp,:,:] * landmask
                plotvar = predictor_test.squeeze() * landmask
                cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
                ax.clabel(cl,fontsize=fsz_ticks)
                if norm is False:
                    pcm     = ax.pcolormesh(lon,lat,plotrel)
                    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
                else:
                    normfactor = np.nanmax(np.abs(plotrel))
                    plotrel = plotrel / normfactor
                    pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
                    label = "max=%.0e" % normfactor
                    viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
                ia += 1
    
        if norm:
            cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
            cb.set_label("Normalized Relevance",fontsize=fsz_title)
    
        title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model: %03i (%.2f)" % (varnames[v],leads[l],isample,classes[iclass],imodel,acc_bymodel[imodel])
        fnstr    = "BetaTest_%s_lead%02i_sample%03i_class%s_model%03i_norm%i" % (varnames[v],leads[l],isample,iclass,imodel,norm)
        title_l2 = r"LRP Params: $Exponent$=%.2f, $\epsilon$=%.0e" % (innexp,inneps)
        plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)
        
        savename = "%s%s.png" % (figpath,fnstr)
        #print(savename)
        plt.savefig(savename,dpi=150,bbox_inches='tight')

relevances_betatest_all =np.array(relevances_epstest_all) # [Network, Method, Exp, Lat, Lon]

if saverel:
    savename = "%sRelevances_BetaTest_%s_%s_sample%02i_class%i_allmodels.npz" % (figpath,varnames[v],leads[l],isample,iclass)
    np.savez(savename,
             **{'relevances':relevances_betatest_all,
                'inneps' : inneps,
                'innmethods': innmethods,
                'lat':lat,
                'lon':lon,
                'runids':runids,
                'innexp':innexp,
                'innbetas':innbetas,
                 },allow_pickle=True
             )

#%% Make a beta plot

fig,axs         = plt.subplots(2,4,figsize=(18,9),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ia = 0
for imeth in range(2):
    innmethod = innmethods[imeth]
    for iexp in range(4):
        innbeta = innbetas[iexp]
        
        # Axis Selection and Labeling
        ax     = axs[imeth,iexp]
        blabel = [0,0,0,0]
        if imeth == 1:
            blabel[-1] = 1
        else:
            ax.set_title(r"$\beta$ = %.0e" % innbeta,fontsize=fsz_title)
        if iexp == 0:
            blabel[0] = 0
            ax.text(-0.05, 0.55, innmethod_fancy[imeth], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_title)
        ax     = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
        
        # Plotting
        plotrel = relevances_betatest_all.mean(0)[imeth,iexp,:,:] * landmask
        plotvar = predictor_test.squeeze() * landmask
        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=predlvls)
        ax.clabel(cl,fontsize=fsz_ticks)
        if norm is False:
            pcm     = ax.pcolormesh(lon,lat,plotrel)
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
        else:
            normfactor = np.nanmax(np.abs(plotrel))
            plotrel = plotrel / normfactor
            pcm     = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="cmo.balance")
            label = "max=%.0e" % normfactor
            viz.label_sp(ia,ax=ax,fig=fig,alpha=0.5,fontsize=fsz_title,labelstyle="%s) "+label,)
        ia += 1

if norm:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_title)

title_l1 = "Predictor %s, Leadtime: %02i, Sample ID: %i (%s), Model ALL (Mean Acc. %.2f)" % (varnames[v],leads[l],isample,classes[iclass],acc_bymodel.mean())
fnstr    = "BetaTest_%s_lead%02i_sample%03i_class%s_modelALL_norm%i" % (varnames[v],leads[l],isample,iclass,norm)
title_l2 = r"LRP Params: $Exponent$=%.2f, $\epsilon$=%.0e" % (innexp,inneps)
plt.suptitle("%s\n%s"%(title_l1,title_l2),fontsize=24)

savename = "%s%s.png" % (figpath,fnstr)
#print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Look at captum explanations 

import captum
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule


isample    = icorr_sample[0][0] # Select a sample index
iclass     = int(targets_all[0,isample])
imodel     = 1 # Select a model index




# Load Predictor and Model
X_in       = X_torch[[isample],:]
modweights = modweights_lead[l][imodel]
pmodel     = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
pmodel.load_state_dict(modweights)
pmodel.eval()


lrp                 = captum.attr.LRP(pmodel)
relevance_captum    = lrp.attribute(X_in,target=iclass)
relevance_captum    = relevance_captum.reshape(1,nlat,nlon).squeeze().detach().numpy()
predictor_samp      = X_in.reshape(1,nlat,nlon).squeeze().detach().numpy()
fig,ax         = plt.subplots(1,1,figsize=(10,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax= viz.add_coast_grid(ax,bbox,fill_color='gray')
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("Captum LRP-z")

#%% Try Integrated Gradients (https://captum.ai/tutorials/TorchVision_Interpret)

ig = captum.attr.IntegratedGradients(pmodel)
relevance_ig = ig.attribute(X_in,target=iclass)
relevance_ig    = relevance_ig.reshape(1,nlat,nlon).squeeze().detach().numpy()


norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax= viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_ig * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("Integrated Gradients")

#%% Try IG + Noise Tunnel
noise_tunnel    = captum.attr.NoiseTunnel(ig)
relevance_ig_nt = noise_tunnel.attribute(X_in,nt_samples=10,nt_type='smoothgrad_sq',target=iclass,
                                         )
relevance_ig_nt    = relevance_ig_nt.reshape(1,nlat,nlon).squeeze().detach().numpy()

norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax= viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_ig_nt * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("Integrated Gradients + Noise Tunnel")


#%% Try Occlusion

occ_stride = 3
occ_window = 5
occlusion        = captum.attr.Occlusion(pmodel)

relevance_occ = occlusion.attribute(X_in,
                                       strides = (occ_stride,),
                                       target=iclass,
                                       sliding_window_shapes=(occ_window,),
                                       baselines=0)
relevance_occ    = relevance_occ.reshape(1,nlat,nlon).squeeze().detach().numpy()

norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax      = viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_occ * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("Occlusion (Stride = %i, Window = %i)" % (occ_stride,occ_window))

#%% Try Gradient Shap

gradshap           = captum.attr.GradientShap(pmodel)
rand_img_dist      = torch.cat([X_in * 0, X_in * 1])

relevance_gradshap = gradshap.attribute(X_in,n_samples=50,stdevs=0.0001,baselines=rand_img_dist,target=iclass)
relevance_gradshap    = relevance_gradshap.reshape(1,nlat,nlon).squeeze().detach().numpy()

norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax      = viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_gradshap * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("GradientShap")



#%% Composite LRP

#from torchvision import models

#model = models.vgg16(pretrained=True)
#model.eval()
#layers = list(model._modules["features"]) + list(model._modules["classifier"])


layers = list([pmodel._modules[key] for key in pmodel._modules.keys()]) #+ list(pmodel._modules["classifier"])

num_layers = len(layers)
for idx_layer in range(1,num_layers):
    if idx_layer <= 5:
        setattr(layers[idx_layer], "rule", GammaRule())
    elif 6 <= idx_layer <= 8:
        setattr(layers[idx_layer], "rule", EpsilonRule())
    elif idx_layer >8:
        setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0))

lrp                 = captum.attr.LRP(pmodel)
relevance_lrpcomp    = lrp.attribute(X_in,target=iclass)
relevance_lrpcomp    = relevance_lrpcomp.reshape(1,nlat,nlon).squeeze().detach().numpy()

norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax      = viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_lrpcomp * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("LRP Composite")

#%%


rule_sel = EpsilonRule(epsilon=30)#GammaRule(gamma=1)
layers   = list([pmodel._modules[key] for key in pmodel._modules.keys()]) #+ list(pmodel._modules["classifier"])

num_layers = len(layers)
for idx_layer in range(1,num_layers):
    setattr(layers[idx_layer], "rule", rule_sel)


lrp                 = captum.attr.LRP(pmodel)
relevance_lrpcomp    = lrp.attribute(X_in,target=iclass)
relevance_lrpcomp    = relevance_lrpcomp.reshape(1,nlat,nlon).squeeze().detach().numpy()

norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax      = viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_lrpcomp * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("LRP 1-Rule")

#%% Deep Lift

dl           = captum.attr.DeepLift(pmodel)
relevance_dl = dl.attribute(X_in,target=iclass)
relevance_dl    = relevance_dl.reshape(1,nlat,nlon).squeeze().detach().numpy()


norm=True
fig,ax         = plt.subplots(1,1,figsize=(12,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

ax      = viz.add_coast_grid(ax,bbox,fill_color='gray')
plotvar = relevance_dl * landmask
if norm:
    plotvar = plotvar/np.nanmax(np.abs(plotvar))
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="cmo.balance")
else:
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="cmo.balance")
cl  = ax.contour(lon,lat,predictor_samp,colors="k",linewidths=0.75)
cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
ax.set_title("DeepLift")

#%% Repeat and compare with model composites







#def make_relevance_plot(relevances_exptest,predictor_test):
    



#%%



# def calc_lrp(innmethod,inneps,innbeta,X_in,modweights_lead):
    
#     runids,modweights_lead,modlist_lead,eparams,nn_param_dict,nlat,nlon,test_loader = inputs
    
#     nmodels  = len(runids)
#     nclasses = 3
#     checkgpu = True
    
#     predictor_all  = []
#     relevances_all = []
#     ypred_all      = []
#     ylab_all       = []
#     for nr in tqdm(range(nmodels)):
        
#         runid = runids[nr]
        
#         # =====================
#         # II. Rebuild the model
#         # =====================
#         # Get the models (now by leadtime)
#         modweights = modweights_lead[l][nr]
#         modlist    = modlist_lead[l][nr]
        
#         # Rebuild the model
#         pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
        
#         # Load the weights
#         pmodel.load_state_dict(modweights)
#         pmodel.eval()
                
#         # =======================================================
#         # III. Test the model separately to get accuracy by class
#         # =======================================================
#         y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
#                                                        checkgpu=checkgpu,debug=False)
        
#         # =======================================================
#         # III. Test the model separately to get accuracy by class
#         # =======================================================
#         y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
#                                                        checkgpu=checkgpu,debug=False)
#         lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
        
        
#         # ===========================
#         # IV. Perform LRP
#         # ===========================
#         nsamples_lead = len(y_actual)
#         inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
#                                           method=innmethod,
#                                           beta=innbeta,
#                                           epsilon=inneps)
#         model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
#         model_prediction                    = model_prediction.detach().numpy().copy()
#         sample_relevances                   = sample_relevances.detach().numpy().copy()
#         if "FNN" in eparams['netname']:
#             predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
#             sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
#         else: # Assume CNN
#             predictor_test    = X_torch.detach().numpy().copy().squeeze() # [test samples x lat x lon]
#             sample_relevances = sample_relevances.squeeze()
        
    
#         # Save Variables
#         predictor_all.append(predictor_test) # Predictors are the same across model runs
#         relevances_all.append(sample_relevances)
#         ypred_all.append(y_predicted)
#         ylab_all.append(y_actual)
    
#         del pmodel

#     predictor_all  = np.array(predictor_all)
#     relevances_all = np.array(relevances_all)
#     predictions_all      = np.array(ypred_all)
#     targets_all       = np.array(ylab_all)
    
#     outdict = {
#         'input' : np.array(predictor_all),
#         'rel'   : np.array(relevances_all),
#         'pred'  : np.array(ypred_all),
#         'targ'  : np.array(ylab_all)
#         }
#     return outdict







# Note, can design several tests from here, etc
#%% Make composites of correct/incorrect predictions

# Preallocate composites
nmodels,nsamples,nlat,nlon = relevances_all.shape

relevance_composites    = np.zeros((2,3,nmodels,nlat,nlon)) # [Correct/Incorrect, Class, Model, Lat, Lon]
#relevance_variances     = relevance_composites.copy()
predictor_composites    = relevance_composites.copy()
n_byclass = np.zeros((2,3,nmodels,))

for nr in tqdm(range(nmodels)):
    for c in range(3):
            
        # Get correct indices
        class_indices                   = np.where(targets_all[nr,:] == c)[0] # Sample indices of a particular class
        correct_ids                     = np.where(targets_all[nr,class_indices] == predictions_model[nr,class_indices])
        incorrect_ids                   = np.where(targets_all[nr,class_indices] != predictions_model[nr,class_indices])
        
        ids_choose = [incorrect_ids,correct_ids,] # 0=incorrect, 1 = correct
        
        for correct in range(2):
            id_in                           = ids_choose[correct]
            correct_pred_id                 = class_indices[id_in] # Correct predictions to composite over
            ncorrect                        = len(id_in)
            n_byclass[correct,c,nr]         = ncorrect
            if ncorrect == 0:
                continue # Set NaN to model without any results
            
            # Make Composite
            correct_relevances               =  relevances_all[nr,correct_pred_id,...]
            relevance_composites[correct,c,nr,:,:] =  correct_relevances.mean(0)
            #relevance_variances[correct,c,nr,:,:]  =  correct_relevances.var(0)
            #relevance_range[correct,c,nr,:,:]      =  correct_relevances.max(0) - correct_relevances.min(0)
            
            # Make Corresponding predictor composites
            correct_predictors               = predictor_all[0,correct_pred_id,...]
            predictor_composites[correct,c,nr,:,:]    = correct_predictors.mean(0)
            #predictor_variances[correct,c,nr,:,:]     = correct_predictors.var(0)


#%% Examine Differences Between Each Case

lon = load_dict['lon']
lat = load_dict['lat']
labels = ["Incorrect","Correct","Correct - Incorrect"]



iclass=1

fig,axs = plt.subplots(1,3,figsize=(12,4.75),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

plotrels = []
plotvars = []
for a in range(3):
    
    ax = axs[a]
    blabel = [0,0,0,1]
    if a == 0:
        blabel[0] = 1
    ax = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
    
    if a < 2:
        plotrel = relevance_composites[a,iclass,:,:,:].mean(0)
        plotvar = predictor_composites[a,iclass,:,:,:].mean(0)
        plotrel = plotrel / np.nanmax(np.abs(plotrel))
        plotrels.append(plotrel)
        clms = [-1,1]
        cmap = 'cmo.balance'
        
    else:
        plotrel = plotrels[1] - plotrels[0]
        clms = [-.25,.25]
        cmap = "seismic"#'seismic'
    plotrel[plotrel==0] = np.nan
    pcm = ax.pcolormesh(lon,lat,plotrel,vmin=clms[0],vmax=clms[1],cmap=cmap)
    if a <2:
        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
    
    if a == 1:
        cb = fig.colorbar(pcm,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
        cb.set_label("Normalized Relevance",fontsize=14)
    if a == 2:
        cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.035,pad=0.015)
        cb.set_label("Difference",fontsize=14)

    ax.set_title(labels[a],fontsize=18)
plt.suptitle("%s | Predictor %s, Lead %i" % (pparams.classes[iclass],varnames[v],leads[l]),fontsize=20)
plt.savefig("%sRelevance_Incorrect_Comparison_%s_%s_class%i_lead%02i.png" % (figpath,expdir,varname,iclass,leads[l]),dpi=150,bbox_inches='tight')

#%% Test LRP Parameter Results (make this into a wrapper)


innmethod = 'e-rule'
inneps    = 1e-6
innbeta   = 0.1

inputs    = [runids,modweights_lead,modlist_lead,eparams,nn_param_dict,nlat,nlon,test_loader] #My lazy way...

def calc_lrp(innmethod,inneps,innbeta,inputs):
    
    runids,modweights_lead,modlist_lead,eparams,nn_param_dict,nlat,nlon,test_loader = inputs
    
    nmodels  = len(runids)
    nclasses = 3
    checkgpu = True
    
    predictor_all  = []
    relevances_all = []
    ypred_all      = []
    ylab_all       = []
    for nr in tqdm(range(nmodels)):
        
        runid = runids[nr]
        
        # =====================
        # II. Rebuild the model
        # =====================
        # Get the models (now by leadtime)
        modweights = modweights_lead[l][nr]
        modlist    = modlist_lead[l][nr]
        
        # Rebuild the model
        pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
        
        # Load the weights
        pmodel.load_state_dict(modweights)
        pmodel.eval()
                
        # =======================================================
        # III. Test the model separately to get accuracy by class
        # =======================================================
        y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                       checkgpu=checkgpu,debug=False)
        
        # =======================================================
        # III. Test the model separately to get accuracy by class
        # =======================================================
        y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                       checkgpu=checkgpu,debug=False)
        lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
        
        
        # ===========================
        # IV. Perform LRP
        # ===========================
        nsamples_lead = len(y_actual)
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                          method=innmethod,
                                          beta=innbeta,
                                          epsilon=inneps)
        model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
        model_prediction                    = model_prediction.detach().numpy().copy()
        sample_relevances                   = sample_relevances.detach().numpy().copy()
        if "FNN" in eparams['netname']:
            predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
            sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
        else: # Assume CNN
            predictor_test    = X_torch.detach().numpy().copy().squeeze() # [test samples x lat x lon]
            sample_relevances = sample_relevances.squeeze()
        
    
        # Save Variables
        predictor_all.append(predictor_test) # Predictors are the same across model runs
        relevances_all.append(sample_relevances)
        ypred_all.append(y_predicted)
        ylab_all.append(y_actual)
    
        del pmodel

    predictor_all  = np.array(predictor_all)
    relevances_all = np.array(relevances_all)
    predictions_all      = np.array(ypred_all)
    targets_all       = np.array(ylab_all)
    
    outdict = {
        'input' : np.array(predictor_all),
        'rel'   : np.array(relevances_all),
        'pred'  : np.array(ypred_all),
        'targ'  : np.array(ylab_all)
        }
    return outdict

#%% Compare Beta and Epsilon Rule

inputs    = [runids,modweights_lead,modlist_lead,eparams,nn_param_dict,nlat,nlon,test_loader] #My lazy way...

innmethod = 'b-rule'

innbeta   = 0.1
dict_beta = calc_lrp(innmethod,inneps,innbeta,inputs)

inneps    = 0
innmethod = 'e-rule'
dict_eps  = calc_lrp(innmethod,inneps,innbeta,inputs)

#%% Plot Comparison of Epsilon and Beta Rules

relevance_composites = np.array([dict_beta['rel'],dict_eps['rel']]).squeeze()
predictor_composites = np.array([dict_beta['input'],dict_eps['input']]).squeeze()

labels = ["Beta-Rule","Z-Rule","Beta - Z"]

iclass = 0

fig,axs = plt.subplots(1,3,figsize=(12,4.75),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

plotrels = []
plotvars = []
for a in range(3):
    
    ax = axs[a]
    blabel = [0,0,0,1]
    if a == 0:
        blabel[0] = 1
    ax = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
    
    if a < 2:
        plotrel = relevance_composites[a,iclass,:,:,:].mean(0)
        plotvar = predictor_composites[a,iclass,:,:,:].mean(0)
        plotrel = plotrel / np.nanmax(np.abs(plotrel))
        plotrels.append(plotrel)
        clms = [-1,1]
        cmap = 'cmo.balance'
        
    else:
        plotrel = plotrels[1] - plotrels[0]
        clms = [-.25,.25]
        cmap = "seismic"#'seismic'
    plotrel[plotrel==0] = np.nan
    pcm = ax.pcolormesh(lon,lat,plotrel,vmin=clms[0],vmax=clms[1],cmap=cmap)
    if a <2:
        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
    
    if a == 1:
        cb = fig.colorbar(pcm,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
        cb.set_label("Normalized Relevance",fontsize=14)
    if a == 2:
        cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.035,pad=0.015)
        cb.set_label("Difference",fontsize=14)

    ax.set_title(labels[a],fontsize=18)
plt.suptitle("%s | Predictor %s, Lead %i" % (pparams.classes[iclass],varnames[v],leads[l]),fontsize=20)
plt.savefig("%sRelevance_LRPMethod_Comparison_%s_%s_class%i_lead%02i_epsi%.2f.png" % (figpath,expdir,varname,iclass,leads[l],inneps),dpi=150,bbox_inches='tight')


#%% Look at individual samples

inetwork = 33
samples  = [0,34,351,566,609]

for inetwork in range(nmodels):

    fig,axs = plt.subplots(2,5,figsize=(18,6.5),
                           subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
    
    for row in range(2):
        for samp in range(len(samples)):
            ax    = axs[row,samp]
            isamp = samples[samp]
            
            blabel = [0,0,0,0]
            if row == 1:
                blabel[-1] = 1
                
                
            if samp == 0:
                blabel[0] = 1
                ax.text(-0.14, 0.55,labels[row], va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes,fontsize=pparams.fsz_title)
                plotrel = relevance_composites[row,inetwork,:,:,:].mean(0)
                plotvar = predictor_composites[row,inetwork,:,:,:].mean(0)
                ax.set_title("Mean")
            else:
                plotrel = relevance_composites[row,inetwork,isamp,:,:]
                plotvar = predictor_composites[row,inetwork,isamp,:,:]
                ax.set_title(isamp)
            
            ax      = viz.add_coast_grid(ax,bbox,blabels=blabel,fill_color='gray')
            plotrel = plotrel / np.nanmax(np.abs(plotrel))
            plotrel[plotrel==0] = np.nan
            pcm = ax.pcolormesh(lon,lat,plotrel,vmin=clms[0],vmax=clms[1],cmap=cmap)
            cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
            
    plt.suptitle("Predictor %s | Lead %02i | Network %02i" % (varnames[v],leads[l],inetwork),fontsize=32)
    savename = "%sRelevance_Beta_v_Epsilon_RandomSamp_Network%03i.png" % (figpath,inetwork)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    print(savename)


#%% Choose a particular sample





#%%SCRAP BELOW------------------------------------------------------------------




#%%


    
    
    


# #%% Select a Leadtime and Variable

# v = 0
# l = 0
# lead = leads[l]

# vt      = time.time()
# varname = varnames[v]


# # ================================
# #% 1. Load model weights + Metrics
# # ================================
# # Get the model weights [lead][run]
# modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
# nmodels = len(modweights_lead[0])


# # Get list of metric files
# search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
# flist  = glob.glob(search)
# flist  = [f for f in flist if "of" not in f]
# flist.sort()
# print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))

# # ======================================
# #% 2. Retrieve predictor and preallocate
# # ======================================
# lt = time.time()
# predictors            = data[[v],...] # Get selected predictor

# # Preallocate
# total_acc_all         = np.zeros((nmodels,nlead))
# class_acc_all         = np.zeros((nmodels,nlead,3)) # 

# # Relevances
# relevances_all        = [] # [nlead][nmodel][sample x lat x lon]
# predictor_all         = [] # [nlead][sample x lat x lon]

# # Predictions
# predictions_all       = [] # [nlead][nmodel][sample]
# targets_all           = [] # [nlead][sample]


# # ===================================
# # I. Data Prep
# # ===================================

# # IA. Apply lead/lag to data
# # --------------------------
# # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
# X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)

# # ----------------------
# # IB. Select samples
# # ----------------------
# _,class_count = am.count_samples(None,y_class)
# if even_sample:
#     eparams['nsamples'] = int(np.min(class_count))
#     print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
#     y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])

# # ----------------------
# # IC. Flatten inputs for FNN
# # ----------------------
# if "FNN" in eparams['netname']:
#     ndat,nchannels,nlat,nlon = X.shape
#     inputsize                = nchannels*nlat*nlon
#     outsize                  = nclasses
#     X_in                     = X.reshape(ndat,inputsize)
# else:
#     X_in = X

# # -----------------------------
# # ID. Place data into a data loader
# # -----------------------------
# # Convert to Tensors
# X_torch = torch.from_numpy(X_in.astype(np.float32))
# y_torch = torch.from_numpy(y_class.astype(np.compat.long))

# # Put into pytorch dataloaders
# test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])



# # ---------------
# # Preallocate
# predictor_lead   = []
# relevances_lead  = []

# predictions_lead = []
# targets_lead     = []
        
        
        
        
# #%% 

# """

# General Procedure

#  1. Load data and subset to test set
#  2. Looping by variable...
#      3. Load the model weights and metrics
#      4. 
     
# """

# nvars = len(varnames)
# for v in range(nvars):
#     vt      = time.time()
#     varname = varnames[v]
    
#     # ================================
#     #% 1. Load model weights + Metrics
#     # ================================
#     # Get the model weights [lead][run]
#     modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
#     nmodels = len(modweights_lead[0])
    
#     # Get list of metric files
#     search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
#     flist  = glob.glob(search)
#     flist  = [f for f in flist if "of" not in f]
#     flist.sort()
#     print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))
    
#     # ======================================
#     #% 2. Retrieve predictor and preallocate
#     # ======================================
#     lt = time.time()
#     predictors            = data[[v],...] # Get selected predictor
    
#     # Preallocate
#     total_acc_all         = np.zeros((nmodels,nlead))
#     class_acc_all         = np.zeros((nmodels,nlead,3)) # 
    
#     # Relevances
#     relevances_all        = [] # [nlead][nmodel][sample x lat x lon]
#     predictor_all         = [] # [nlead][sample x lat x lon]
    
#     # Predictions
#     predictions_all       = [] # [nlead][nmodel][sample]
#     targets_all           = [] # [nlead][sample]
    
#     # ==============
#     #%% Loop by lead
#     # ==============
#     # Note: Since the testing sample is the same withheld set for the experiment, we can use leadtime as the outer loop.
    
#     # -----------------------
#     # Loop by Leadtime...
#     # -----------------------
#     outname = "/Test_Metrics_%s_%s_evensample%i.npz" % (dataset_name,varname,even_sample)
#     if standardize_input:
#         outname = proc.addstrtoext(outname,"_standardizeinput")
        
#     for l,lead in enumerate(leads):
        

        
#         # ===================================
#         # I. Data Prep
#         # ===================================
        
#         # IA. Apply lead/lag to data
#         # --------------------------
#         # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
#         X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)
        
#         # ----------------------
#         # IB. Select samples
#         # ----------------------
#         _,class_count = am.count_samples(None,y_class)
#         if even_sample:
#             eparams['nsamples'] = int(np.min(class_count))
#             print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
#             y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
        
#         # ----------------------
#         # IC. Flatten inputs for FNN
#         # ----------------------
#         if "FNN" in eparams['netname']:
#             ndat,nchannels,nlat,nlon = X.shape
#             inputsize                = nchannels*nlat*nlon
#             outsize                  = nclasses
#             X_in                     = X.reshape(ndat,inputsize)
#         else:
#             X_in = X
        
#         # -----------------------------
#         # ID. Place data into a data loader
#         # -----------------------------
#         # Convert to Tensors
#         X_torch = torch.from_numpy(X_in.astype(np.float32))
#         y_torch = torch.from_numpy(y_class.astype(np.compat.long))
        
#         # Put into pytorch dataloaders
#         test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
        
#         # Preallocate
#         predictor_lead   = []
#         relevances_lead  = []
        
#         predictions_lead = []
#         targets_lead     = []
        
#         # --------------------
#         # 05. Loop by runid...
#         # --------------------
#         for nr,runid in tqdm(enumerate(runids)):
#             rt = time.time()
            
#             # =====================
#             # II. Rebuild the model
#             # =====================
#             # Get the models (now by leadtime)
#             modweights = modweights_lead[l][nr]
#             modlist    = modlist_lead[l][nr]
            
#             # Rebuild the model
#             pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
            
#             # Load the weights
#             pmodel.load_state_dict(modweights)
#             pmodel.eval()
            
#             # =======================================================
#             # III. Test the model separately to get accuracy by class
#             # =======================================================
#             y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
#                                                            checkgpu=checkgpu,debug=False)
#             lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
            
#             # Save variables
#             total_acc_all[nr,l]   = lead_acc
#             class_acc_all[nr,l,:] = class_acc
#             predictions_lead.append(y_predicted)
            
#             if nr == 0:
#                 targets_all.append(y_actual)
            
#             if calc_lrp:
#                 # ===========================
#                 # IV. Perform LRP
#                 # ===========================
#                 nsamples_lead = len(y_actual)
#                 inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
#                                                   method=innmethod,
#                                                   beta=innbeta)
#                 model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
#                 model_prediction                    = model_prediction.detach().numpy().copy()
#                 sample_relevances                   = sample_relevances.detach().numpy().copy()
#                 if "FNN" in eparams['netname']:
#                     predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
#                     sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
#                 else: # Assume CNN
#                     predictor_test    = X_torch.detach().numpy().copy().squeeze() # [test samples x lat x lon]
#                     sample_relevances = sample_relevances.squeeze()
                    
                
#                 # Save Variables
#                 if nr == 0:
#                     predictor_all.append(predictor_test) # Predictors are the same across model runs
#                 relevances_lead.append(sample_relevances)
                
#                 # Clear some memory
#                 del pmodel
#                 torch.cuda.empty_cache()  # Save some memory
                
#                 #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
#                 # End Lead Loop >>>
#         relevances_all.append(relevances_lead)
#         predictions_all.append(predictions_lead)
#         print("\nCompleted training for %s lead %i of %i in %.2fs" % (varname,lead,leads[-1],time.time()-lt))
    
    
#     # =============================================================================================================================
#     #%% Composite the relevances (can look at single events later, it might actually be easier to write a separate script for that)
#     # =============================================================================================================================
#     if calc_lrp:
#         # the purpose here is to get some quick, aggregate metrics
        
#         # Need to add option to cull models in visualization script...
#         st_rel_comp          = time.time()
        
#         relevance_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan # [lead x model x class x lat x lon]
#         relevance_variances  = relevance_composites.copy()                    # [lead x model x class x lat x lon]
#         relevance_range      = relevance_composites.copy()                    # [lead x model x class x lat x lon]
#         predictor_composites = np.zeros((nlead,3,nlat,nlon)) * np.nan         # [lead x class x lat x lon]
#         predictor_variances  = predictor_composites.copy()                    # [lead x class x lat x lon]
#         ncorrect_byclass     = np.zeros((nlead,nmodels,3))                # [lead x model x class
        
#         for l in range(nlead):
            
#             for nr in tqdm(range(nmodels)):
                
#                 predictions_model = predictions_all[l][nr] # [sample]
#                 relevances_model  = relevances_all[l][nr]  # [sample x lat x lon]
                
#                 for c in range(3):
                    
#                     # Get correct indices
#                     class_indices                   = np.where(targets_all[l] == c)[0] # Sample indices of a particular class
#                     correct_ids                     = np.where(targets_all[l][class_indices] == predictions_model[class_indices])
#                     correct_pred_id                 = class_indices[correct_ids] # Correct predictions to composite over
#                     ncorrect                        = len(correct_pred_id)
#                     ncorrect_byclass[l,nr,c]        = ncorrect
                    
#                     if ncorrect == 0:
#                         continue # Set NaN to model without any results
#                     # Make Composite
#                     correct_relevances               =  relevances_model[correct_pred_id,...]
#                     relevance_composites[l,nr,c,:,:] =  correct_relevances.mean(0)
#                     relevance_variances[l,nr,c,:,:]  =  correct_relevances.var(0)
#                     relevance_range[l,nr,c,:,:]      =  correct_relevances.max(0) - correct_relevances.min(0)
                    
#                     # Make Corresponding predictor composites
#                     correct_predictors               = predictor_all[l][correct_pred_id,...]
#                     predictor_composites[l,c,:,:]    = correct_predictors.mean(0)
#                     predictor_variances[l,c,:,:]     = correct_predictors.var(0)
#         print("Saved Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
            
#         # ===================================================
#         #%% Save Relevance Output
#         # ===================================================
#         #Save as relevance output as a dataset
        
#         st_rel = time.time()
        
#         lat = load_dict['lat']
#         lon = load_dict['lon']
        
#         # Save variables
#         save_vars      = [relevance_composites,relevance_variances,relevance_range,predictor_composites,predictor_variances,ncorrect_byclass]
#         save_vars_name = ['relevance_composites','relevance_variances','relevance_range','predictor_composites','predictor_variances',
#                           'ncorrect_byclass']
        
#         # Make Coords
#         coords_relevances = {"lead":leads,"runid":runids,"class":pparams.classes,"lat":lat,"lon":lon}
#         coords_preds      = {"lead":leads,"class":pparams.classes,"lat":lat,"lon":lon}
#         coords_counts     = {"lead":leads,"runid":runids,"class":pparams.classes}
        
#         # Convert to dataarray and make encoding dictionaries
#         ds_all    = []
#         encodings = {}
#         for sv in range(len(save_vars)):
            
#             svname = save_vars_name[sv]
#             if "relevance" in svname:
#                 coord_in = coords_relevances
#             elif "predictor" in svname:
#                 coord_in = coords_preds
#             elif "ncorrect" in svname:
#                 coord_in = coords_counts
            
#             da = xr.DataArray(save_vars[sv],dims=coord_in,coords=coord_in,name=svname)
#             encodings[svname] = {'zlib':True}
#             ds_all.append(da)
            
#         # Merge into dataset
#         ds_all = xr.merge(ds_all)
        
#         # Save Relevance data
#         outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (datpath,expdir,dataset_name,varname,even_sample)
#         if standardize_input:
#             outname = proc.addstrtoext(outname,"_standardizeinput")
#         ds_all.to_netcdf(outname,encoding=encodings)
#         print("Saved Relevances to %s in %.2fs" % (outname,time.time()-st_rel))
        
#         if save_all_relevances:
#             st_acc = time.time()
#             print("Saving all relevances!")
#             save_vars      = [relevances_all,predictor_all,]
#             save_vars_name = ["relevances","predictors",]
#             for sv in range(len(save_vars)):
#                 outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_%s.npy" % (datpath,expdir,dataset_name,varname,even_sample,save_vars_name[sv])
#                 np.save(outname,save_vars[sv],allow_pickle=True)
#                 print("Saved %s to %s in %.2fs" % (save_vars_name[sv],outname,time.time()-st_acc))

#     # ===================================================
#     #%% Save accuracy and prediction data
#     # ===================================================
#     st_acc = time.time()
    
#     save_vars         = [total_acc_all,class_acc_all,predictions_all,targets_all,ens_test,leads,runids]
#     save_vars_name    = ["total_acc","class_acc","predictions","targets","ensemble","leads","runids"]
#     metrics_dict      = dict(zip(save_vars_name,save_vars))
#     outname           = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_accuracy_predictions.npz" % (datpath,expdir,dataset_name,varname,even_sample)
#     if standardize_input:
#         outname = proc.addstrtoext(outname,"_standardizeinput")
#     np.savez(outname,**metrics_dict,allow_pickle=True)
#     print("Saved Accuracy and Predictions to %s in %.2fs" % (outname,time.time()-st_acc))
    
#     print("Completed calculating metrics for %s in %.2fs" % (varname,time.time()-vt))





