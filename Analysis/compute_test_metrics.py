#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Compute Test Metrics

 - Test Accuracy
 - Loss by Epoch (Test)
 - 


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
from amv import proc

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

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN10_128_PaperRun_Detrended"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters
outdir_cust         = None#"e-rule_exp1/" #Set to None or a custom directory within Metrics

# Processing Options
even_sample         = False
standardize_input   = False # Set to True to standardize variance at each point
calc_lrp            = True # Set to True to calculate relevance composites

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
varnames            = ["SSS","SLP","SSH","SST"]       # Names of predictor variables
leads               = np.arange(0,26,1)#np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1
innepsilon     = 1e-2

# Other toggles
save_all_relevances = False                # True to save all relevances (~33G per file...)
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = False                 # Set verbose outputs

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
        

#%% 

"""

General Procedure

 1. Load data and subset to test set
 2. Looping by variable...
     3. Load the model weights and metrics
     4. 
     
"""

nvars = len(varnames)
for v in range(nvars):
    vt      = time.time()
    varname = varnames[v]
    
    # ================================
    #% 1. Load model weights + Metrics
    # ================================
    # Get the model weights [lead][run]
    modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
    nmodels = len(modweights_lead[0])
    
    # Get list of metric files
    search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))
    
    # ======================================
    #% 2. Retrieve predictor and preallocate
    # ======================================
    lt = time.time()
    predictors            = data[[v],...] # Get selected predictor
    
    # Preallocate
    total_acc_all         = np.zeros((nmodels,nlead))
    class_acc_all         = np.zeros((nmodels,nlead,3)) # 
    
    # Relevances
    relevances_all        = [] # [nlead][nmodel][sample x lat x lon]
    predictor_all         = [] # [nlead][sample x lat x lon]
    
    # Predictions
    predictions_all       = [] # [nlead][nmodel][sample]
    targets_all           = [] # [nlead][sample]
    
    # ==============
    #%% Loop by lead
    # ==============
    # Note: Since the testing sample is the same withheld set for the experiment, we can use leadtime as the outer loop.
    
    # -----------------------
    # Loop by Leadtime...
    # -----------------------
    outname = "/Test_Metrics_%s_%s_evensample%i.npz" % (dataset_name,varname,even_sample)
    if standardize_input:
        outname = proc.addstrtoext(outname,"_standardizeinput")
        
    for l,lead in enumerate(leads):
        

        
        # ===================================
        # I. Data Prep
        # ===================================
        
        # IA. Apply lead/lag to data
        # --------------------------
        # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
        X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)
        
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
        
        # Preallocate
        predictor_lead   = []
        relevances_lead  = []
        
        predictions_lead = []
        targets_lead     = []
        
        # --------------------
        # 05. Loop by runid...
        # --------------------
        for nr,runid in tqdm(enumerate(runids)):
            rt = time.time()
            
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
            lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
            
            # Save variables
            total_acc_all[nr,l]   = lead_acc
            class_acc_all[nr,l,:] = class_acc
            predictions_lead.append(y_predicted)
            if nr == 0:
                targets_all.append(y_actual)
            
            if calc_lrp:
                # ===========================
                # IV. Perform LRP
                # ===========================
                nsamples_lead = len(y_actual)
                inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                                  method=innmethod,
                                                  beta=innbeta,
                                                  epsilon=innepsilon)
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
                relevances_lead.append(sample_relevances)
                
                # Clear some memory
                del pmodel
                torch.cuda.empty_cache()  # Save some memory
                
                #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
                # End Lead Loop >>>
        relevances_all.append(relevances_lead)
        predictions_all.append(predictions_lead)
        print("\nCompleted training for %s lead %i of %i in %.2fs" % (varname,lead,leads[-1],time.time()-lt))
    
    
    # =============================================================================================================================
    #%% Composite the relevances (can look at single events later, it might actually be easier to write a separate script for that)
    # =============================================================================================================================
    if calc_lrp:
        # the purpose here is to get some quick, aggregate metrics
        
        # Need to add option to cull models in visualization script...
        st_rel_comp          = time.time()
        
        relevance_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan # [lead x model x class x lat x lon]
        relevance_variances  = relevance_composites.copy()                    # [lead x model x class x lat x lon]
        relevance_range      = relevance_composites.copy()                    # [lead x model x class x lat x lon]
        predictor_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan         # [lead x class x lat x lon]
        predictor_variances  = predictor_composites.copy()                    # [lead x class x lat x lon]
        ncorrect_byclass     = np.zeros((nlead,nmodels,3))                # [lead x model x class
        
        # Preallocate incorrect arrays
        incorrect_relevance_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan
        incorrect_predictor_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan
        nincorrect_byclass   = np.zeros((nlead,nmodels,3))
        
        for l in range(nlead):
            
            for nr in tqdm(range(nmodels)):
                
                predictions_model = predictions_all[l][nr] # [sample]
                relevances_model  = relevances_all[l][nr]  # [sample x lat x lon]
                
                for c in range(3):
                    
                    # Get correct indices
                    class_indices                   = np.where(targets_all[l] == c)[0] # Sample indices of a particular class
                    correct_ids                     = np.where(targets_all[l][class_indices] == predictions_model[class_indices])
                    correct_pred_id                 = class_indices[correct_ids] # Correct predictions to composite over
                    ncorrect                        = len(correct_pred_id)
                    ncorrect_byclass[l,nr,c]        = ncorrect
                    
                    # Get Incorrect data
                    incorrect_ids                   = np.where(targets_all[l][class_indices] != predictions_model[class_indices])
                    incorrect_pred_id               = class_indices[incorrect_ids]
                    nincorrect_byclass[l,nr,c]      = len(incorrect_pred_id)
                    
                    if ncorrect == 0:
                        continue # Set NaN to model without any results
                    
                    # Make Composite
                    correct_relevances               =  relevances_model[correct_pred_id,...]
                    relevance_composites[l,nr,c,:,:] =  correct_relevances.mean(0)
                    relevance_variances[l,nr,c,:,:]  =  correct_relevances.var(0)
                    relevance_range[l,nr,c,:,:]      =  correct_relevances.max(0) - correct_relevances.min(0)
                    
                    # Make Corresponding predictor composites
                    correct_predictors               = predictor_all[l][correct_pred_id,...]
                    predictor_composites[l,nr,c,:,:]    = correct_predictors.mean(0)
                    predictor_variances[l,nr,c,:,:]     = correct_predictors.var(0)
                    
                    # Make Incorrect Composites
                    incorrect_relevances                       =  relevances_model[incorrect_pred_id,...]
                    incorrect_relevance_composites[l,nr,c,:,:] =  incorrect_relevances.mean(0)
                    incorrect_predictors                       =  predictor_all[l][incorrect_pred_id,...]
                    incorrect_predictor_composites[l,nr,c,:,:] =  incorrect_predictors.mean(0)
                    
        print("Saved Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
            
        # ===================================================
        #%% Save Relevance Output
        # ===================================================
        #Save as relevance output as a dataset
        
        st_rel = time.time()
        
        lat = load_dict['lat']
        lon = load_dict['lon']
        
        # Save variables
        save_vars      = [relevance_composites,relevance_variances,relevance_range,predictor_composites,predictor_variances,ncorrect_byclass,
                          incorrect_relevance_composites,incorrect_predictor_composites,nincorrect_byclass]
        save_vars_name = ['relevance_composites','relevance_variances','relevance_range','predictor_composites','predictor_variances',
                          'ncorrect_byclass','incorrect_relevance_composites','incorrect_predictor_composites','nincorrect_byclass']
        
        # Make Coords
        coords_relevances = {"lead":leads,"runid":runids,"class":pparams.classes,"lat":lat,"lon":lon}
        coords_preds      = {"lead":leads,"class":pparams.classes,"lat":lat,"lon":lon}
        coords_counts     = {"lead":leads,"runid":runids,"class":pparams.classes}
        
        # Convert to dataarray and make encoding dictionaries
        ds_all    = []
        encodings = {}
        for sv in range(len(save_vars)):
            
            svname = save_vars_name[sv]
            if "relevance" in svname:
                coord_in = coords_relevances
            elif "predictor" in svname:
                coord_in = coords_relevances
            elif "ncorrect" in svname:
                coord_in = coords_counts
            
            da = xr.DataArray(save_vars[sv],dims=coord_in,coords=coord_in,name=svname)
            encodings[svname] = {'zlib':True}
            ds_all.append(da)
        
        # Merge into dataset
        ds_all = xr.merge(ds_all)
        
        # Save Relevance data
        if outdir_cust is not None:
            outdir_new = "%s%s/Metrics/%s" % (datpath,expdir,outdir_cust)
            proc.makedir(outdir_new)
            outname    = "%sTest_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (outdir_new,dataset_name,varname,even_sample)
        else:
            outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (datpath,expdir,dataset_name,varname,even_sample)
        if standardize_input:
            outname = proc.addstrtoext(outname,"_standardizeinput")
        # if innmethod == 'e-rule':
        #     outname = proc.addstrtoext(outname,"_epsrule")
        ds_all.to_netcdf(outname,encoding=encodings)
        print("Saved Relevances to %s in %.2fs" % (outname,time.time()-st_rel))
        
        if save_all_relevances:
            st_acc = time.time()
            print("Saving all relevances!")
            save_vars      = [relevances_all,predictor_all,]
            save_vars_name = ["relevances","predictors",]
            for sv in range(len(save_vars)):
                outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_%s.npy" % (datpath,expdir,dataset_name,varname,even_sample,save_vars_name[sv])
                np.save(outname,save_vars[sv],allow_pickle=True)
                print("Saved %s to %s in %.2fs" % (save_vars_name[sv],outname,time.time()-st_acc))
    
    # ===================================================
    #%% Save accuracy and prediction data
    # ===================================================
    st_acc = time.time()
    
    save_vars         = [total_acc_all,class_acc_all,predictions_all,targets_all,ens_test,leads,runids]
    save_vars_name    = ["total_acc","class_acc","predictions","targets","ensemble","leads","runids"]
    metrics_dict      = dict(zip(save_vars_name,save_vars))
    if outdir_cust is not None:
        outname = "%s/Test_Metrics_%s_%s_evensample%i_accuracy_predictions.npz" % (outdir_new,dataset_name,varname,even_sample)
    else:
        outname = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_accuracy_predictions.npz" % (datpath,expdir,dataset_name,varname,even_sample)
    if standardize_input:
        outname = proc.addstrtoext(outname,"_standardizeinput")
    np.savez(outname,**metrics_dict,allow_pickle=True)
    print("Saved Accuracy and Predictions to %s in %.2fs" % (outname,time.time()-st_acc))
    
    print("Completed calculating metrics for %s in %.2fs" % (varname,time.time()-vt))





