#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Predict AMV Module (amvmod)

Module containing functions for predict_amv. Contents are below:

<><><> Data Loading  <><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    load_model_weights              : Use string glob to get model weights/file structure produced by [train_NN_CESM1.py]
    load_cmip6_data                 : Load predictor and target values processed with [prep_data_lens.py]
    
<><><> Indexing and Sampling <><><><><><><><><><><><><><><><><><><N_simple><><><><>
    
        ~~~ Sampling and Classification
    make_classes                    : Create classes for an index based on given thresholds
    select_samples                  : Sample even amounts from each class with option to shuffle data
    count_samples                   : Counts classes and returns indices for each class
    consistent_sample               : Select the same samples of a target predictor dataset based on the maximum leadtime.
    
        ~~~ Ensemble Year Indexing 
    get_ensyr                       : Get the ensemble and year for reshaped validation (or training) indices after it has already been subsetted
    make_ensyr                      : Create meshgrid or array index for [nens] x [nyr]
    select_ensyr_linearids          : Given an array of [sample x 2] where 0=ens, 1=year retrieve corresponding linear indices
    get_ensyr_linear                : Given linear indices for an ens x year array with the lead/lag appplied, retrieve corresponding linear indices for a reference lead/lag
    retrieve_ensyr_shuffid          : Given shuffled indices from [select_samples], get linear indices, ensemble, and year labels accounting for offset and leadtimes (older script)
    retrieve_lead                   : Get prediction leadtime/index from shuffled indices
    
<><><> NN Training + Wrapper Functions <><><><><><><><><><><><><><><><><><>
    
        ~~~ Predictors/Target Prep
    normalize_ds                    : Compute mean and standard deviation for a dataarray (xarray)
    compute_persistence_baseline    : Calculate the persistence baseline given the classified target indices [make_classes] output
    prepare_predictors_target       : Wrapper that 1) loads data 2) applies landice mask 3) normalizes data 4) change NaN to zero 5) get thresholds 6) standardize predictors 7) make classes 8) subset to ensemble
    apply_lead                      : Apply leadtime to data and predictor
    train_test_split                : Splits data into train/test/validation
    prep_traintest_classification   : (old?) Wrapper function that (1) applies lead, (2) makes + (3) samples classes,  (4) splits samples into subsets
    
        ~~~ NN Training
    train_NN_lead                   : Wrapper function that 1) Subsets data train/test 2) Put into dataloaders 3) build model w/[build_FNN_sample]/[transfer_model] 4) train the model [train_ResNet] 5) compute test accuracy
    train_ResNet                    : Training script for CNN or ResNet. Includes support for LR scheduler, etc
    
        ~~~ NN Testing
    test_model                      : Given model and testing set, returns the predicted + actual values and loss
    compute_class-acc               : Calculate the accuracy by each class and the total accuracy given output of [test_model]
    
        ~~~ NN Creation/Loading
    recreate_model                  : Recreate NN model for loading weights based on <modelname>, <nn_param_dict> from [predict_amv_params.py], <inputsize> and <outsize>
    transfer_model                  : Load pretrained weights and architectures for simplecnn, cnn2_lrp, and timm (no longer supported)
    build_simplecnn                 : Construct a simple CNN based on inputs
    build_simplecnn_fromdict       : Construct a simple CNN with custom parameters from a dictionary for hyperparameter testing
    build_FNN_simple                : Build a feed-forward network
    calc_layerdims                  : Compute size of first fully-connected layer after N convolutional layers
    
<><><> Analysis  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
        ~~~ General Analysis
    compute_LRP_composites          : Composites relevance for correct predictions of top N (by test acc) models and optionally normalizes output
    calc_confmat                    : Calculate the confusion matrices
    calc_confmat_loop               : Given predictions+labels, retrieves confusion matrix indices
    get_topN                        : Get indices for topN values of an array, searching along last dimension
    get_barcount                    : Get count of existing classes along an axis
    
        ~~~ Metrics Output Handling
    load_result                     : Given a metrics file, load out the results 
    load_metrics_byrun              : Load all training runs for a given experiment
    make_expdict                    : Load experiment metrics/runs into array and make into a dict
    unpack_expdict                  : Unpack variables from expdict of a metrics file
    
<><><> Network Architectures <><><><><><><><><><><><><><><><><><><><><><><>
    
    CNN2 (class)                    : CNN2 class that supports the captum interpretability package with separate pool and convolutional layers 
    
<><><> amv.proc copy <><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
        ~~~ Copies from the amv.proc module. Delete this eventually...
    find_nan                        : Remove points where there is NaN along a dimension
    eof_simple                      : Simple EOF function by Yu-Chiao
    coarsen_byavg                   : Coarsen input variable to specified resolution
    regress_2d                      : Regresses 2 2-D variables
    sel_region                      : Select points within a region
    calc_AMV                        : Compute AMV Index for detrended/anomalized SST data
    detrend_poly                    : Matrix version of polynomial detrend
    lon360to180                     : Flip longitude from degrees east to west
    area_avg                        : Compute area-weighted average
    regress2ts                      : regress variable to timeseries
    plot_AMV                        : Plot AMV time series
    plot_AMV_spatial                : Plot AMV Pattern
    deseason_lazy                   : Xarray deseason function without reading out values
    init_map                        : quickly initialize a map for plotting

@author: gliu
"""

from scipy.signal import butter,filtfilt
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset

import torch.optim as optim
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import glob
import torch
from tqdm import tqdm
import copy

#%% Import Custom DataLoader Module

import sys
sys.path.append("..")
import amv_dataloader as dl

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#%% Data Loading (Move to AMV Data Loader...)
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def load_cmip6_data(dataset_name,varname,bbox,datpath=None,detrend=0,regrid=None,
                    ystart=1850,yend=2014,lowpass=0,return_latlon=False):
    """
    Load predictor and target that has been processed with prep_data_lens.py
    
    Inputs
    ------
        dataset_name    [STR]   : Name of CMIP6 MMLE
        varname         [STR]   : Name of predictor variable
        bbox            [LIST]  : Bounding Box for cropping (West,East,South,North)
        datpath         [STR]   : Path to processed data. Default: ../../CESM_data/CMIP6_LENS/processed/
        detrend         [BOOL]  : True if the target was detrended. Default: False.
        regrid          [STR]   : Regridding setting for the data. Default: None.
        ystart          [INT]   : Start year of processed dataset. Default: 1850.
        yend            [INT]   : End year of processed dataset. Default: 2014.
        lowpass         [BOOL]  : True if the target was low-pass filtered. Default: True.
        return_latlon   [BOOL]  : True to return lat/lon. Default: False.
    
    Output
    ------
        data            [ARRAY] : Predictor [channel x ens x year x lat x lon]
        target          [ARRAY] : Target    [ens x year]
    """
    # Load data that has been processed by prep_data_lens.py
    if datpath is None:
        datpath = "../../CESM_data/CMIP6_LENS/processed/"
    
    # Load and crop data to region
    ncname  = "%s/%s_%s_NAtl_%ito%i_detrend%i_regrid%sdeg.nc" % (datpath,dataset_name,
                                                                           varname,
                                                                           ystart,yend,
                                                                           detrend,regrid)
    ds      = xr.open_dataset(ncname) # [channel x ensemble x year x lat x lon]
    ds      = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])) 
    data    = ds[varname].values[None,...]  # [channel x ensemble x year x lat x lon]
    
    # Load labels
    lblname = "%s/%s_sst_label_%ito%i_detrend%i_regrid%sdeg_lp%i.npy" % (datpath,dataset_name, #Mostly compied from NN_traiing script
                                                                         ystart,yend,
                                                                         detrend,regrid,lowpass)
    target  = np.load(lblname) # [ensemble x year]
    if return_latlon:
        lat = ds.lat.values
        lon = ds.lon.values
        return data,target,lat,lon
    return data,target
    

def load_model_weights(modpath,expdir,leads,varname):
    """
    Get list of model weights using the glob string: [modpath + expdir + *varname*.pt]
    Inputs
    ------
    modpath [STR]       : Path to where the directory model weights are saved
    expdir  [STR]       : Name of the directory where the model wheres are saved
    leads   [ARRAY]     : List of leadtimes to search for
    varname [STR]       : Name of predictor the model was trained on (used for globbing *%s*.pt)
    
    Outputs
    -------
    modweights_lead [LIST] : List of model weights by leadtime [lead][model#]
    modlist         [LIST] : List of paths to the model [lead][model#]
    """
    # Pull model list
    modlist_lead = []
    modweights_lead = []
    for lead in leads:
        # Get Model Names
        modlist = glob.glob("%s%s/Models/*%s*.pt" % (modpath,expdir,varname))
        modlist.sort()
        print("Found %i models in %s, Lead %i" % (len(modlist),expdir,lead))
        # Cull the list (only keep files with the specified leadtime)
        str1 = "_lead%i_" % (lead)   # ex. "..._lead2_..."
        str2 = "_lead%02i_" % (lead) # ex. "..._lead02_..."
        if np.any([str2 in f for f in modlist]):
            modlist = [fname for fname in modlist if str2 in fname]
        else:
            modlist = [fname for fname in modlist if str1 in fname]
        nmodels = len(modlist)
        print("\t %i models remain for lead %i" % (len(modlist),lead))
        modlist_lead.append(modlist)
        modweights = []
        for m in range(nmodels):
            mod    = torch.load(modlist[m])
            modweights.append(mod)
        modweights_lead.append(modweights)
    return modweights_lead,modlist_lead

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#%% Indexing and Sampling
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Sampling and Classification 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_classes(y,thresholds,exact_value=False,reverse=False,
                 quantiles=False):
    """
    Makes classes based on given thresholds. 

    Parameters
    ----------
    y          : ARRAY          - Labels to classify
    thresholds : ARRAY          - 1D Array of thresholds to partition the data
    exact_value: BOOL, optional - Set to True to use the exact value in thresholds (rather than scaling by standard deviation)
    reverse    : BOOL, optional - Set to True to number classes in reverse order 
                                     Ex. for thresholds=[-1,1]: returns [>1,-1<x<1,<-1] if True, reverse of this if False
    
    Returns
    -------
    y_class    : ARRAY [samples,class] Classified samples, where the second dimension contains an integer representing each threshold d
    """
    
    if quantiles is False:
        if exact_value is False: # Scale thresholds by standard deviation
            y_std = np.std(y) # Get standard deviation
            thresholds = np.array(thresholds) * y_std
    else: # Determine Thresholds from quantiles
        thresholds = np.quantile(y,thresholds,axis=0) # Replace Thresholds with quantiles
    
    nthres  = len(thresholds)
    y_class = np.zeros((y.shape[0],1))
    
    if nthres == 1: # For single threshold cases
        thres = thresholds[0]
        y_class[y<=thres] = 0
        y_class[y>thres] = 1
        
        print("Class 0 Threshold is y <= %.2f " % (thres))
        print("Class 0 Threshold is y > %.2f " % (thres))
        return y_class
    
    for t in range(nthres+1):
        if t < nthres:
            thres = thresholds[t]
        else:
            thres = thresholds[-1]
        
        if reverse: # Assign class 0 to largest values
            tassign = nthres-t
        else:
            tassign = t
        
        if t == 0: # First threshold
            y_class[y<=thres] = tassign
            print("Class %i Threshold is y <= %.2f " % (tassign,thres))
        elif t == nthres: # Last threshold
            y_class[y>thres] = tassign
            print("Class %i Threshold is y > %.2f " % (tassign,thres))
        else: # Intermediate values
            thres0 = thresholds[t-1]
            y_class[(y>thres0) * (y<=thres)] = tassign
            print("Class %i Threshold is %.2f < y <= %.2f " % (tassign,thres0,thres))
    if quantiles is True:
        return y_class,thresholds
    return y_class

def select_samples(nsamples,y_class,X,shuffle=True,verbose=True,):
    """
    Sample even amounts from each class. Shuffles data (unless shuffle=False)

    Parameters
    ----------
    nsample : INT
        Number of samples to get from each class
    y_class : ARRAY [samples x 1]
        Labels for each sample
    X : ARRAY [samples x channels x height x width]
        Input data for each sample
    shuffle : BOOL
        Set to True to shuffle the indices
    
    Returns
    -------
    
    y_class_sel : ARRAY [samples x 1]
        Subsample of labels with equal amounts for each class
    X_sel : ARRAY [samples x channels x height x width]
        Subsample of inputs with equal amounts for each class
    idx_sel : ARRAY [samples x 1]
        Indices of selected arrays
    
    """
    
    allsamples,nchannels,H,W = X.shape
    classes                  = np.unique(y_class)
    nclasses                 = len(classes)
    
    use_all_samples = False
    if nsamples == "ALL":
        print("Using all samples!")
        use_all_samples = True
    else:

        # Sort input by classes
        label_by_class  = []
        input_by_class  = []
        idx_by_class    = []
        
        y_class_sel = np.zeros([nsamples*nclasses,1])#[]
        X_sel       = np.zeros([nsamples*nclasses,nchannels,H,W])#[]
        idx_sel     = np.zeros([nsamples*nclasses]) 
        for i in range(nclasses):
            
            # Sort by Class
            inclass   = classes[i]
            idx       = (y_class==inclass).squeeze()
            sel_label = y_class[idx,:]
            sel_input = X[idx,:,:,:]
            sel_idx   = np.where(idx)[0]
            
            label_by_class.append(sel_label)
            input_by_class.append(sel_input)
            idx_by_class.append(sel_idx)
            classcount = sel_input.shape[0]
            if verbose:
                print("%i samples found for class %i" % (classcount,inclass))
            
            # Shuffle and select first N samples for that class ...
            shuffidx = np.arange(0,classcount,1)
            if shuffle:
                np.random.shuffle(shuffidx)
            else:
                if verbose:
                    print("Warning: data will not be shuffled prior to class subsetting!")
            if use_all_samples is False:
                shuffidx = shuffidx[0:nsamples] # Restrict to sample
            else:
                nsamples = classcount
            
            # Select Shuffled Indices
            y_class_sel[i*nsamples:(i+1)*nsamples,:] = sel_label[shuffidx,:]
            X_sel[i*nsamples:(i+1)*nsamples,...]     = sel_input[shuffidx,...]
            idx_sel[i*nsamples:(i+1)*nsamples]       = sel_idx[shuffidx]
    
    # Shuffle samples again before output (so they arent organized by class)
    if use_all_samples:
        total_samples = allsamples        # Use all samples, as recorded earlier
        y_class_sel   = y_class
        X_sel         = X
        idx_sel       = np.arange(0,allsamples)
    else:
        total_samples = nsamples*nclasses # Only use selected samples
    shuffidx = np.arange(0,total_samples,1)
    np.random.shuffle(shuffidx) # Shuffle classes prior to output
    
    return y_class_sel[shuffidx,...],X_sel[shuffidx,...],idx_sel[shuffidx,...]

def count_samples(nsamples,y_class):
    """
    Simplified version of select_samples that only counts the classes
    and returns the indices/counts
    """
    classes         = np.unique(y_class)
    nclasses        = len(classes)
    idx_by_class    = [] 
    count_by_class  = []
    for i in range(nclasses):
        
        # Sort by Class
        inclass   = classes[i]
        idx       = (y_class==inclass).squeeze()
        sel_idx   = np.where(idx)[0]
        
        idx_by_class.append(sel_idx)
        classcount = sel_idx.shape[0]
        count_by_class.append(classcount)
        print("%i samples found for class %i" % (classcount,inclass))
    return idx_by_class,count_by_class

def consistent_sample(data,target_class,leads,nsamples,leadmax=None,
                      nens=None,ntime=None,shuffle_class=False,debug=False):
    """
    Take consistent samples of a target.predictor dataset

    Parameters
    ----------
    data : ARR [channel x ens x time x lat x lon]
        Predictor dataset
    target_class : ARR [ens x time]
        Target (classes)
    leads : LIST [leads]
        Leadtimes
    nsamples : INT
        Number of samples to take
    leadmax : TYPE, optional
        Longest leadtime to take the sample from (defaults to maximum lead). The default is None.
    nens : INT, optional
        Number of ensemble members to use. The default is 42.
    ntime : INT, optional
        Number of timesteps. The default is 86.
    shuffle_class : BOOL, optional
        Set to true to shuffle calsses before sampling. The default is shuffle_class.
    debug : BOOL, optional
        Set to true to ouput debugging messages. The default is False.

    Returns
    -------
    target_indices : LIST [samples]
        Indices of selected samples for the target.
    target_refids : LIST
       Reference ids for target in the form of of (ens,yr). [samples x (ens,yr)].
    predictor_indices : LIST [lead][samples]
        Indices of selected samples for the predictor.
    predictor_refids : [lead][samples x (ens,yr)]
        DESCRIPTION.

    """
    nclasses = len(np.unique(target_class))
    nlead    = len(leads)
    if (nens is None) or (ntime is None):
        nchannels,nens,ntime,nlat,nlon = data.shape
    if leadmax is None:
        leadmax      = leads.max()
    X,y_class    = apply_lead(data[[0],...],target_class,leadmax,reshape=True,ens=nens,tstep=ntime)
    
    if nsamples is None: # Default: nsamples = smallest class
        threscount = np.zeros(nclasses)
        for t in range(nclasses):
            count = len(np.where(y_class==t)[0])
            print("Found %i samples for class %i" % (count,t))
            threscount[t] = count
        nsamples = int(np.min(threscount))
        print("Using %i samples, the size of the smallest class" % (nsamples))
    
    # Select samples based on the longest leadtime. 
    if nsamples == "ALL":
        shuffidx_max = np.arange(0,y_class.shape[0]).astype(int)
    else:
        y_class,X,shuffidx_max = select_samples(nsamples,y_class,X,verbose=debug,shuffle=shuffle_class)
        shuffidx_max           = shuffidx_max.astype(int) # There indices are w.r.t. the lagged data
        
    # Get [absolute] linear indices for reference lead [lead], based on applied lead [leadmax]
    target_indices,target_refids = get_ensyr_linear(leadmax,shuffidx_max,
                reflead=0,nens=nens,nyr=ntime,
                apply_lead=True,ref_lead=True,
                return_labels=True,debug=debug)
    
    # Convert to numpy array of [sample x 2] where 0 = ens, 1 = yr
    target_refids              = np.array([[a[0],a[1]] for a in target_refids],dtype='int')
    
    # Get indices for predictors
    predictor_indices = []
    predictor_refids  = []
    for l,lead in enumerate(leads):
        # Get the references 
        pref      = np.array([[a[0],a[1]-lead] for a in target_refids],dtype="int")
        if debug:
            plt.hist(pref[:,1]),plt.title("lead %i (predictor years %i to %i)"% (lead,pref[:,1].min(),pref[:,1].max())),plt.show()
            
        target_linearids = select_ensyr_linearids(pref,target_lead=0,lag=False,nens=nens,
                                                     nyr=ntime,)
        predictor_indices.append(target_linearids)
        predictor_refids.append(pref)
    if debug:  
        ii = 22
        for l in range(nlead):
            print("Lead %02i, target is (e=%02i,y=%02i, idx=%i), predictor is (e=%02i,y=%02i, idx%i)" % (leads[l],
                                                                                target_refids[ii,0],target_refids[ii,1],target_indices[ii],
                                                                                predictor_refids[l][ii,0],predictor_refids[l][ii,1],
                                                                                predictor_indices[l][ii]))
    return target_indices,target_refids,predictor_indices,predictor_refids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Ensemble Year Indexing 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_ensyr(id_val,lead,ens=40,tstep=86,percent_train=0.8,get_train=False):
    # Get ensemble and year of reshaped valdation indices (or training if get_train=True)
    # Assume target is of the order [ens  x time] (default is (40,86))
    # Assumes default 80% used for training
    id_ensyr = np.zeros((ens,tstep),dtype='object')
    for e in range(ens):
        for y in range(tstep):
            id_ensyr[e,y] = (e,y)
    reshape_id = id_ensyr[:ens,lead:].reshape(ens*(tstep-lead),1)
    nsamples = reshape_id.shape[0]
    if get_train:
        val_id = reshape_id[0:int(np.floor(percent_train*nsamples)),:]
    else:
        val_id = reshape_id[int(np.floor(percent_train*nsamples)):,:]
    return val_id[id_val]


def make_ensyr(ens=42,yr=86,meshgrid=True):
    """Make either meshgrid or index array for [nens] x [nyr]"""
    if meshgrid:
        yrs = np.tile(np.arange(0,yr)[:,None],ens).T #+ startyr # [ens x yr]
        ens = np.tile(np.arange(0,ens)[:,None],yr) #[ens x yr]
        return yrs,ens
    else:
        id_ensyr = np.zeros((ens,yr),dtype='object')
        for e in range(ens):
            for y in range(yr):
                id_ensyr[e,y] = (e,y)
        return id_ensyr # [ens x yr]

def select_ensyr_linearids(ens_yr_arr,target_lead=0,lag=False,nens=42,nyr=86,debug=False):
    """Given an array of [sample x 2] where 0=ens, 1=year, retrieve the corresponding linear
    indices. """
    
    # Get the arrays (absolute)
    yrs,ens  = make_ensyr(ens=nens,yr=nyr,meshgrid=True)  
    id_ensyr = make_ensyr(ens=nens,yr=nyr,meshgrid=False) 
    in_arrs  = [yrs,ens,id_ensyr]
    
    # Apply lead/lag for the target lead
    if lag: # Lag the data 
        apply_arr = [arr[:,:(nyr-target_lead)].flatten() for arr in in_arrs]
    else: # Lead the data
        apply_arr = [arr[:,target_lead:].flatten() for arr in in_arrs]
    target_years,target_ens,target_id = apply_arr
    
    # Find the corresponding linear indices
    target_linearids = []
    nsamples         = ens_yr_arr.shape[0]
    for NN in range(nsamples):
        sel_ens = ens_yr_arr[NN,0]
        sel_yr  = ens_yr_arr[NN,1]
        foundid = np.where((target_years ==sel_yr) * (target_ens == sel_ens))[0]
        assert len(foundid) == 1,"Found less/more than 1 id for %s: %s" % (ens_yr_arr[NN,:],foundid)
        assert np.all(np.array(target_id[foundid[0]]) == ens_yr_arr[NN,:]),"Mismatch in found indices (%s) != (%s)" % (np.array(target_id[foundid[0]]),ens_yr_arr[NN,:])
        target_linearids.append(foundid[0])
        if debug:
            print("For sample %i (ens==%i, year=%i)" % (NN,ens_yr_arr[NN,0],ens_yr_arr[NN,1]))
            print("\tFound linear id %i" % foundid[0])
            print("\tWith ens %i, yr %i" % target_id[foundid[0]])
    return target_linearids

def get_ensyr_linear(lead,linearids,
              reflead=0,nens=42,nyr=86,
              apply_lead=True,ref_lead=True,
              return_labels=False,debug=False,return_counterpart=True):
    """
    Given linear indices for a ens x year array where the lead/lag has been applied...
    Retrieve the corresponding linear indices for a reference lead/lag application
    Also optionally recover the lead and ensemble member labels
    
    Parameters
    ----------
    lead (INT)          : Lead applied to data
    linearids (LIST)    : Linear indices to find
    reflead (INT)       : Lead applied to reference (default = 0)
    nens     (INT)      : Number of ensemble members, default is 42
    nyr      (INT)      : Number of years, default is 86
    apply_lead (BOOL)   : True to apply lead, false to apply lag to data
    ref_lead  (BOOL)    : Same but for reference set
    return_labels   (BOOL) : Set to true to return ens,yr labels for a dataset

    Returns
    -------
    
    """
    
    # Get the arrays (absolute)
    yrs,ens  = make_ensyr(ens=nens,yr=nyr,meshgrid=True)
    id_ensyr = make_ensyr(ens=nens,yr=nyr,meshgrid=False)
    in_arrs  = [yrs,ens,id_ensyr]
    
    # Apply lead/lag
    if apply_lead: # Lead the data
        apply_arr = [arr[:,lead:].flatten() for arr in in_arrs]
    else: # Lag the data
        apply_arr = [arr[:,:(nyr-lead)].flatten() for arr in in_arrs]
    
    # Get the corresponding indices where lead/lag is applied
    apply_ids = [arr[linearids] for arr in apply_arr]
    yrslead,enslead,idlead = apply_ids

    # Find the corresponding indices where it is not flattened, at a specified lead
    if ref_lead: # Lead the data
        ref_arr     = [arr[:,reflead:].flatten() for arr in in_arrs]
        counterpart_arrs = [arr[:,:(nyr-reflead)].flatten() for arr in in_arrs]
    else: # Lag the data
        ref_arr          = [arr[:,:(nyr-reflead)].flatten() for arr in in_arrs]
        counterpart_arrs = [arr[:,reflead:].flatten() for arr in in_arrs]
    refyrs,refens,refids                = ref_arr
    counter_yrs,counter_ens,counter_ids = counterpart_arrs

    ref_linearids         = [] # Contains linear ids in the lead
    counterpart_linearids = [] # Countains linear ids for the counterpart (lag if lead, lead if lag...)
    for ii,ens_yr_set in enumerate(idlead):
        
        # Get the reference lead/lag
        sel_ens,sel_yr = ens_yr_set
        foundid = np.where((refyrs ==sel_yr) * (refens == sel_ens))[0]
        assert len(foundid) == 1,"Found less/more than 1 id for i=%i (%s): %s" % (linearids[ii],str(ens_yr_set),foundid)
        ref_linearids.append(foundid[0])
        if debug:
            print("For linear id %i..." % (linearids[ii]))
            print("\tApplied Lead is             : %s" % (str(ens_yr_set)))
            print("\tFound Reference Lead (l=%i) : %s" % (reflead,refids[foundid[0]]))
            print("\tReference linear id         : %i" % (foundid[0]))
        assert refids[foundid[0]] == ens_yr_set
        
        # Get the counterpart indices (and check to make sure they are ok...)
        c_ens,c_yr = counter_ids[foundid[0]]
        assert c_ens == sel_ens,"The counterpart ensemble member (%i) is not equal to the reference member (%i)" % (c_ens,sel_ens)
        assert ((sel_yr - c_yr) == reflead),"The lagged difference %i is not equal to reflead %i" % (c_yr-sel_yr,reflead)
    if return_labels:
        return ref_linearids,refids[ref_linearids]
    else:
        return ref_linearids

def retrieve_ensyr_shuffid(lead,shuffid_in,percent_train,percent_val=0,
                           offset=0,ens=42,nyr=86,debug=False):
    """
    Retrieve the linear indices, ensemble, and year labels given a set
    of shuffled indices from select_sample, accounting for offset and leadtimes.
    Generally works with shuffid output from older NN_training script
    
    Parameters
    ----------
    lead : INT. Leadtime applied in units of provided time axis.
    shuffid_in : ARRAY. Shuffled indices to subset, select, and retrieve indices from.
    percent_train : NUMERIC. % data used for training
    percent_val : NUMERIC. % data used for validation. optional, default is 0.
    offset : NUMERIC. Offset to shift train.test.val split optional, default is 0.
    ens : INT. Number of ensemble members to include optional, The default is 42.
    nyr : INT, Numer of years to include. optional, the default is 86.

    Returns
    -------
    shuffid_split : LIST of ARRAYS [train/test/val][samples]
        Shuffids partitioned into each set
    refids_linear_split : LIST of ARRAYS [train/test/val][samples]
        Corresponding linear indices to unlagged data (ens x year)
    refids_label_split : LIST of ARRAYS  [train/test/val][samples,[ens,yr]]
        Ensemble and Year arrays for each of the corresponding splits
    
    Dependencies: train_test_split, get_ensyr_linear
    """
    
    # Apply Train/Test/Validation Split, accounting for offset
    dummyX                  = shuffid_in[:,None]
    dumX,dumY,split_indices = train_test_split(dummyX,dummyX,
                                               percent_train,percent_val=percent_val,
                                               debug=debug,offset=offset,return_indices=True)
    shuffid_split           = [shuffid_in[split_id] for split_id in split_indices]

    # Get the actual ens and year (and corresponding linear id)
    refids_label_split  = []
    refids_linear_split = []
    for ii in range(len(shuffid_split)):
        shuffi = shuffid_split[ii].astype(int)
        
        ref_linearids,refids_label=get_ensyr_linear(lead,shuffi,
                      reflead=0,nens=ens,nyr=nyr,
                      apply_lead=True,ref_lead=True,
                      return_labels=True,debug=debug,return_counterpart=False)
        
        # Convert to Array
        refids_label = np.array([[rf[0],rf[1]] for rf in refids_label]) # [sample, [ens,yr]]
        
        # Append
        refids_linear_split.append(ref_linearids)
        refids_label_split.append(refids_label)
    return shuffid_split,refids_linear_split,refids_label_split

def retrieve_lead(shuffidx,lead,nens,tstep):
    """
    Get prediction leadtime/index from shuffled indices (?)
    Copied from viz_acc_by_predictor.py on 2023.01.25
    """
    orishape = [nens,tstep-lead]
    outidx   = np.unravel_index(shuffidx,orishape)
    return outidx

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#%% NN Training + Wrapper Functions
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Predictors/Target Prep
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def normalize_ds(ds):
    std,mu = np.nanstd(ds),np.nanmean(ds) 
    print("Standard deviation is %.2e, Mean is %.2e"% (std,mu))
    return std,mu

def compute_persistence_baseline(leads,y_class,nsamples=None,percent_train=1,
                                 shuffle_trainsplit=False,use_train=False,debug=True):
    
    '''
    lead               : prediction leadtimes
    y_class            : the target classes
    nsamples           : how much subsampling to do
    percent_train      : percentage of the data to use in training
    shuffle_trainsplit : Use a consistent sample (currently not supported...)
    use_train          : Use training rather thn testing set
    '''
    # Preallocate
    total_acc       = [] # [lead]
    acc_by_class    = [] # [lead x class]
    yvalpred        = [] # [lead x ensemble x time]
    yvallabels      = [] # [lead x ensemble x time]
    samples_counts  = [] # [lead x class]
    
    # Get needed dimensions
    ens,tstep       = y_class.shape
    
    # Looping for each leadtime
    for l,lead in enumerate(leads):

        # -------------------------------------------------------
        # Set [predictor] to the [target] but at the initial time
        # -------------------------------------------------------
        X                 = y_class[:,:(tstep-lead)].flatten()[:,None,None,None] # Expand dimensions to accomodate function
        y                 = y_class[:,lead:].flatten()[:,None] # Note, overwriting y again ...
        
        # ----------------------------------------------
        # Subsample prior to the split, if option is set
        # ----------------------------------------------
        if nsamples is not None:
            y_class_label,y_class_predictor,shuffidx = select_samples(nsamples,y,X)
            y_class_predictor                        = y_class_predictor.squeeze()
        else: # Otherwise, use max samples
            y_class_label     = y
            y_class_predictor = X.squeeze()
        
        # ----------------
        # Train/Test Split
        # ----------------
        if percent_train < 1:
            X_subset,y_subset = train_test_split(y_class_predictor,y_class_label,percent_train,
                                                    debug=True)
            
            # Subset data if percent train is less than 100%
            if percent_train < 1:
                X_train,X_val     = X_subset
                y_train,y_val     = y_subset
                if use_train:
                    y_class_label     = y_train
                    y_class_predictor = X_train
                else:
                    y_class_label     = y_val
                    y_class_predictor = X_val
        
        if debug:
            _,_=count_samples(nsamples,y_class_label)
        
        # ----------------------
        # Make predictions
        # ----------------------
        allsamples = y_class_predictor.shape[0]
        classval   = [0,1,2]
        correct    = np.array([0,0,0])
        total      = np.array([0,0,0])
        for n in range(allsamples):
            actual = int(y_class_label[n,0])
            y_pred = int(y_class_predictor[n])
            
            #print("For sample %i, predicted %i, actual %i" % (n,y_pred,actual))
            # Add to Counter
            if actual == y_pred:
                correct[actual] += 1
            total[actual] += 1
        
        # ----------------------------------
        # Calculate and save overall results
        # ----------------------------------
        accbyclass   = correct/total
        totalacc     = correct.sum()/total.sum() 
        
        # Append Results
        acc_by_class.append(accbyclass)
        total_acc.append(totalacc)
        #yvalpred.append(y_pred)
        yvalpred.append(y_class_predictor)
        yvallabels.append(y_class_label)
        samples_counts.append(total)
        
        # Report Results
        print("**********************************")
        print("Results for lead %i" % lead + "...")
        print("\t Total Accuracy is %.3f " % (totalacc*100) + "%")
        print("\t Accuracy by Class is...")
        for i in range(3):
            print("\t\t Class %i : %.3f" % (classval[i],accbyclass[i]*100) + "% " + "(%i/%i)" % (correct[i],total[i]))
        print("**********************************")
        # End Lead Loop
    out_dict = {
        "total_acc"      : np.array(total_acc),
        "acc_by_class"   : np.array(acc_by_class),
        "yvalpred"       : yvalpred,
        "yvallabels"     : yvallabels,
        "samples_counts" : samples_counts,
        "nsamples"       : nsamples,
        "leads"          : leads,
        "percent_train"  : percent_train,
        "y_class"        : y_class
        }
    return out_dict

def prepare_predictors_target(varnames,eparams,debug=False,
                           return_target_values=False,
                           return_nfactors=False,load_all_ens=False,
                           savestd=True,return_test_set=False):
    """ Prepares predictors and target. Works with output from:
        [prep_data_byvariable, make_landice_mask, prepare_regional_targets]
        Does the following:
        1. Loads predictors and target
        2. Applies land ice mask
        3. Normalize data
        4. Change NaNs to zero
        5. Get Threshold Values
        6. Standardize predictors in space (optional)
        7. Make classes
        8. Subset to ensemble
        
        Inputs:
            eparams  [DICT]             : Parameter dictionary, set in train_cesm_parameters.
            varnames [LIST]             : List of variable names to load.
            debug    [BOOL]             : True to print debugging messages.
            return_target_values [BOOL] : True to return target values (in addition to class).
            return_nfactors [BOOL]      : True to return normalization factors (mu and sigma)
            load_all_ens [BOOL]         : True to load all ensemble members
            savestd [BOOL]              : True to save spatial standardization factors
            return_test_set [BOOL]      : True to load the testing set
            
            
        Returns:
            data           [ARRAY : channel x ens x year x lat x lon] : Normalized + Masked Predictors
            target_class   [ARRAY : ens x year] : Target with corresponding class numbers
            thresholds_in  [LIST] : Threshold values for classification
            target         [ARRAY : ens x year] : Target with actual values (if return_target_values is True)
            nfactors_byvar [DICT : [varname][mean or std]] : Normalization factors used
        """
    # ------------------
    # 1. Load predictor and labels, lat/lon, cropped to region, eparams['detrend','region','norm']
    # ------------------
    target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'],newpath=True,norm=eparams['norm'])
    data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True,newpath=True)
    
    # ------------------
    # 2. Load land-ice mask and apply, eparams['mask']
    # ------------------
    if eparams['mask']:
        limask                         = dl.load_limask(bbox=eparams['bbox'])
        data                           = data * limask[None,None,None,:,:]  # NaN Points to Zero
    
    # ------------------
    # 3. Normalize data eparams['norm']
    # ------------------
    nchannels        = data.shape[0]
    # * Note, doing this for each channel, but in reality, need to do for all channels
    nfactors_byvar = {} # Dictionary of normalization factors
    for ch in range(nchannels):
        std_var      = np.nanstd(data[ch,...])
        mu_var       = np.nanmean(data[ch,...])
        data[ch,...] = (data[ch,...] - mu_var)/std_var
        nfactors = {}
        nfactors['mean']  = mu_var
        nfactors['std']   = std_var
        nfactors_byvar[varnames[ch]] = nfactors.copy()
        
    if debug:
        [print(normalize_ds(data[d,...])) for d in range(4)]
    
    # ------------------
    # 4. Change nan points to zero
    # ------------------
    data[np.isnan(data)] = 0 
    
    # ------------------
    # 5. Set exact threshold value, eparams['thresholds']
    # ------------------
    std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
    if eparams['quantile'] is False:
        thresholds_in = [-std1,std1]
    else:
        thresholds_in = eparams['thresholds']
    
    # ----------------------------------------------------
    # 6. Standardize predictors ins pace, if option is set
    #     Note: copied original section from check_predictor_normalization.py
    # ----------------------------------------------------
    if eparams['stdspace']:
        print("Predictors will be standardized in space. Output will be saved to the [Metrics] directory.")
        # Compute standardizing factor (and save)
        std_vars = np.std(data,(1,2)) # [variable x lat x lon]
        if savestd:
            for v in range(nchannels):
                savename = "../Metrics/%s_spatial_std.npy" % (varnames[v])
                np.save(savename,std_vars[v,:,:])
        
        # Apply standardization
        data                 = data / std_vars[:,None,None,:,:] # add singleton dim for ens and year
        data[np.isnan(data)] = 0 # Make sure NaN values are zero again
        std_vars_after       = np.std(data,(1,2))
        check                =  np.all(np.nanmax(np.abs(std_vars_after)) < 2)
        assert check, "Standardized values are not below 2!"
        
    # ------------------
    # 7. Classify AMV Events
    # ------------------
    target_class = make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
    target_class = target_class.reshape(target.shape)
    if debug:
        target_class_temp = make_classes(target[:,25:].flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
        count_samples(None,target_class_temp)
        
    # ------------------
    # 8. Subset predictor and get dimensions, eparams['ens']
    # ------------------
    if load_all_ens is False: # Subset according to ens (default)
        if return_test_set: # Grab test set before subsetting
            # Load the data
            data_test                      = data[:,eparams['ens']:,...]
            target_test                    = target[eparams['ens']:,:]
            target_test_class              = target_class[eparams['ens']:,:]
        # Subset the data
        data   = data[:,0:eparams['ens'],...]
        target = target[0:eparams['ens'],:]
    
    # Output
    load_dict = {
        'data'          : data,
        'target_class'  : target_class,
        'thresholds_in' : thresholds_in,
        'lat'           : lat,
        'lon'           : lon,
        }
    
    if return_target_values:
        load_dict['target'] = target
    if return_nfactors:
        load_dict['nfactors_byvar'] = nfactors_byvar
    if return_test_set: # Get test set
        # Add testing sets to the dictionary
        load_dict['data_test']         = data_test
        load_dict['target_test']       = target_test
        load_dict['target_class_test'] = target_test_class
    return load_dict

def apply_lead(data,target,lead,reshape=True,ens=None,tstep=None):
    """
    data : ARRAY [channel x ens x yr x lat x lon ]
        Network Inputs
    target : ARRAY [ens x yr]
        Network Outputs
    lead : INT
        Leadtime (in years)
    reshape : BOOL
        Reshape the output to combine ens x year
    Returns
        if reshape is False:
        y : [ens x lead]
        X : [channel x ens x yr x lat x lon]
    elif reshape is True:
        y : [samples x 1]
        X : [sample  x channel x lat x lon]
    """
    
    # Get dimensions
    if ens is None:
        ens = data.shape[1]
    if tstep is None:
        tstep = data.shape[2]
    nchannels,_,_,nlat,nlon = data.shape
    
    # Get Ens Indices
    #sel_ens = np.arange(0,ens+1)
    
    # Apply Lead
    y                            = target[:ens,lead:]
    X                            = (data[:,:ens,:tstep-lead,:,:])
    if reshape:
        y = y.reshape(ens*(tstep-lead),1)
        X = X.reshape(nchannels,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
    return X,y

def train_test_split(X,y,percent_train,percent_val=0,debug=False,offset=0,return_indices=False):
    
    """
    Perform train/test/val split on predictor [X: samples ,...] and label [y: samples x 1].
    Data is split into 3 blocks (in order), with an optional added offset
    [percent_train] -- [percent_test] -- [percent_val]
    
    Inputs:
        X [ARRAY: Samples x ...] : Predictors
        y [ARRAY: Samples x 1]   : Labels
        percent_train [FLOAT]    : Percentage for training
        percent_val   [FLOAT]    : Percentage for validation
        debug         [BOOL]     : Set to True to print Debuggin Messages
        offset        [FLOAT]    : Percentage to offset values for sampling (ex, offset=0.3, train set will start at 0.3 rather than 0.0)
        
    Returns:
        X_subset [LIST: X_train,X_test,X_val] : List of subsetted arrays (predictors)
        y_subset [LIST: y_train,y_test,y_val] : List of subsetted arrays (labels)
        
    """
    # Get indices
    nsamples        = y.shape[0]
    percent_splits  = [percent_train,1-percent_train-percent_val,percent_val]
    assert np.array(percent_splits).sum() == 1, "percent train/test/split must add up to 1.0. Currently %.2f" % np.array(percent_splits).sum()
        
    segments        = ("Train","Test","Validation")
    cumulative_pct  = 0
    segment_indices = []
    for p,pct in enumerate(percent_splits):
        # Add modulo accounting for offset
        pct_rng = np.array([cumulative_pct+offset,cumulative_pct+pct+offset])%1
        if (pct_rng[0] == pct_rng[1]):# and (p >0):
            if debug:
                print("Exceeded max percentage on segment [%s], Skipping..."%segments[p])
            continue
        # Add ranges to account for endpoints
        shift_flag=False # True: Offset shifts the chunk beyond 100%, 4 points required
        if pct_rng[0] > pct_rng[1]: # Shift to beginning of dataset
            pct_rng = np.array([pct_rng[0],1,0,pct_rng[1]]) # [larger idx, end, beginning, smaller idx]
            shift_flag=True
        
        # Get range of indices
        idx_rng = np.floor(nsamples*pct_rng).astype(int)
        
        if shift_flag:
            seg_idx = np.concatenate([np.arange(idx_rng[0],idx_rng[1]),np.arange(idx_rng[2],idx_rng[3])])
            segment_indices.append(seg_idx)
            if debug:
                print("Range of percent for %s segment is [%.2f to 1] and [0 to %.2f], idx [%i:%i] and [%i:%i]" % (segments[p],
                                                                                      pct_rng[0],pct_rng[3],
                                                                                      idx_rng[0],idx_rng[1],
                                                                                      idx_rng[2],idx_rng[3]
                                                                                                   ))
        else:
            idx_rng = np.floor(nsamples*pct_rng).astype(int)
            segment_indices.append(np.arange(idx_rng[0],idx_rng[1]))
            if debug:
                print("Range of percent for %s segment is %.2f to %.2f, idx %i:%i" % (segments[p],
                                                                                      pct_rng[0],pct_rng[1],
                                                                                      segment_indices[p][0],segment_indices[p][-1]
                                                                                                   ))
        cumulative_pct += pct
        # end pct Loop
    
    # Subset the data
    y_subsets = []
    X_subsets = []
    for pp in range(len(segment_indices)):
        if percent_splits == 0:
            continue
        y_subsets.append(y[segment_indices[pp],...])
        X_subsets.append(X[segment_indices[pp],...])
    if debug:
        pct_check = [y.shape[0]/nsamples for y in y_subsets]
        print("Subset percentages are %s" % pct_check)
    if return_indices:
        return X_subsets,y_subsets,segment_indices
    return X_subsets,y_subsets

def prep_traintest_classification(data,target,lead,thresholds,percent_train,
                                  ens=None,tstep=None,
                                  quantile=False,return_ic=False,return_indices=False):
    """
    Parameters
    ----------
    data : ARRAY [variable x ens x yr x lat x lon ]
        Network Inputs
    target : ARRAY [ens x yr]
        Network Outputs
    lead : INT
        Leadtime (in years)
    thresholds : List
        List of stdev thresholds. See make_classes()
    percent_train : FLOAT
        Percentage of data to use for training. Rest for validation.
    ens : INT, optional
        # of Ens to include. The default is None (all of them).
    tstep : INT, optional
        # of Years to include. The default is None (all of them).
    quantile : BOOL, optional
        Use quantiles rather than stdev based thresholds. Default is False.
    return_ic : BOOL, optional
        Return the starting class. Quantile thresholds not supported

    Returns
    -------
    None.

    """
    # Get dimensions
    if ens is None:
        ens = data.shape[1]
    if tstep is None:
        tstep = data.shape[2]
    
    # Apply the lead
    X,y            = apply_lead(data,target,lead,reshape=True,ens=ens,tstep=tstep)
    nsamples,_,_,_ = X.shape
    
    # Make the labels
    y_class = make_classes(y,thresholds,reverse=True,quantiles=quantile)
    if quantile == True:
        thresholds = y_class[1].T[0]
        y_class    = y_class[0]
    if (nsamples is None) or (quantile is True):
        nthres = len(thresholds) + 1
        threscount = np.zeros(nthres)
        for t in range(nthres):
            threscount[t] = len(np.where(y_class==t)[0])
        nsamples = int(np.min(threscount))*3
    y_val  = y.copy()
    
    # Compute class of initial state if option is set
    if return_ic:
        y_start    = target[:ens,:tstep-lead].reshape(ens*(tstep-lead),1)
        y_class_ic = make_classes(y_start,thresholds,reverse=True,quantiles=quantile)
        
        y_train_ic = y_class_ic[0:int(np.floor(percent_train*nsamples)),:]
        y_val_ic   = y_class_ic[int(np.floor(percent_train*nsamples)):,:]
    
    # Test/Train Split
    split_output = train_test_split(X,y_class,percent_train,percent_val=0,
                                           debug=False,return_indices=return_indices)
    
    X_train,X_val = split_output[0]
    y_train,y_val = split_output[1]
    
    output=[X_train,X_val,y_train,y_val]
    if return_ic:
        output = output + [y_train_ic,y_val_ic]
    if return_indices:
        output = output + split_output[2]
    return output

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ NN Training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_NN_lead(X,y,eparams,pparams,debug=False,checkgpu=True,verbose=True):
    """
    Wrapper for training neural network.
    
    Inputs:
        X (ARRAY: [samples,channels,lat,lon]) : Predictors
        y (ARRAY: [samples,1]) : Target
        eparams (dict) : Training Parameters for the experiment, see train_NN_CESM1.py
        pparams (dict) : Architecture hyperparameters for network, see train_NN_CESM1.py
    Returns:
        model : Trained PyTorch Model
        trainloss,valloss,testloss : Loss by Epoch
        trainacc,valacc,testacc : Accuracy by Epoch
        y_predicted,y_actual : Predicted and Actual value for test set
        class_acc : Test accuracy by class
        lead_acc : Total accuracy by class

    """
    
    nclasses  = len(eparams['thresholds']) + 1
    
    # Flatten input data for FNN
    ndat,nchannels,nlat,nlon = X.shape
    outsize                  = nclasses
    if "FNN" in eparams['netname']:
        inputsize            = nchannels*nlat*nlon
        X                    = X.reshape(ndat,inputsize)
    
    # --------------------------
    # 10. Train Test Split
    # --------------------------
    X_subsets,y_subsets      = train_test_split(X,y,eparams['percent_train'],
                                                   percent_val=eparams['percent_val'],
                                                   debug=debug,offset=eparams['cv_offset'])
    
    # Print classes for debugging
    if debug:
        print("For offset %.2f" % (eparams["cv_offset"]))
        _,_ = count_samples(eparams['nsamples'],y_subsets[1])
        #return # Uncomment here to check for offsetting error
    
    # Convert to Tensors
    X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
    y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
    
    # # Put into pytorch dataloaders
    data_loaders = [DataLoader(TensorDataset(X_subsets[iset],y_subsets[iset]), batch_size=eparams['batch_size']) for iset in range(len(X_subsets))]
    if len(data_loaders) == 2:
        if debug:
            print("There is no validation portion. Validation perc is set to %.2f" % (eparams['percent_val']))
        train_loader,test_loader = data_loaders
    else:
        train_loader,test_loader,val_loader = data_loaders
    
    # -------------------
    # 11. Train the model
    # -------------------
    nn_params = pparams.nn_param_dict[eparams['netname']] # Get corresponding param dict for network
    
    # Initialize model
    if "FNN" in eparams['netname']:
        layers = build_FNN_simple(inputsize,outsize,nn_params['nlayers'],nn_params['nunits'],nn_params['activations'],
                                  dropout=nn_params['dropout'],use_softmax=eparams['use_softmax'])
        pmodel = nn.Sequential(*layers)
        
    else:
        # Note: Currently not supported due to issues with timm model. Need to rewrite later...
        pmodel = transfer_model(eparams['netname'],nclasses,cnndropout=nn_params['cnndropout'],unfreeze_all=eparams['unfreeze_all'],
                                nlat=nlat,nlon=nlon,nchannels=nchannels,param_dict=nn_params)
        
    # Train/Validate Model
    model,trainloss,testloss,valloss,trainacc,testacc,valacc = train_ResNet(pmodel,eparams['loss_fn'],eparams['opt'],
                                                                               data_loaders,
                                                                               eparams['max_epochs'],early_stop=eparams['early_stop'],
                                                                               verbose=verbose,reduceLR=eparams['reduceLR'],
                                                                               LRpatience=eparams['LRpatience'],checkgpu=checkgpu,debug=debug)
    
    # ------------------------------------------------------
    # 12. Test the model separately to get accuracy by class
    # ------------------------------------------------------
    y_predicted,y_actual,test_loss = test_model(model,test_loader,eparams['loss_fn'],
                                                   checkgpu=checkgpu,debug=False)
    lead_acc,class_acc = compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False)
    
    func_output = [model,
              trainloss,valloss,testloss,
              trainacc,valacc,testacc,
              y_predicted,y_actual,
              class_acc,
              lead_acc]
    return func_output

def train_ResNet(model,loss_fn,optimizer,dataloaders,
                 max_epochs,early_stop=False,verbose=True,
                 reduceLR=False,LRpatience=3,checkgpu=True,debug=True):
    """
    inputs:
        model       - Resnet model
        loss_fn     - (torch.nn) loss function
        opt         - tuple of [optimizer_name, learning_rate, weight_decay] for updating the weights
                      currently supports "Adadelta" and "SGD" optimizers
        dataloaders - List of (torch.utils.data.DataLoader) [train, test, val]
            - trainloader - (torch.utils.data.DataLoader) for training dataset
            - testloader  - (torch.utils.data.DataLoader) for testing dataset
            - valloader   - (torch.utils.data.DataLoader) for validation dataset
        max_epochs  - number of training epochs
        early_stop  - BOOL or INT, Stop training after N epochs of increasing validation error
                     (set to False to stop at max epoch, or INT for number of epochs)
        verbose     - set to True to display training messages
        reduceLR    - BOOL, set to true to use LR scheduler
        LRpatience  - INT, patience for LR scheduler
    
    output:
    
    dependencies:
        from torch import nn,optim

    """
    # Check if there is GPU
    if checkgpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    
    # Get list of params to update
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # if verbose:
            #     print("Params to learn:")
            #     print("\t",name)
    
    # Set optimizer
    if optimizer[0] == "Adadelta":
        opt = optim.Adadelta(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == "SGD":
        opt = optim.SGD(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == 'Adam':
        opt = optim.Adam(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    
    # Set up loaders
    mode_names = ["train","test","val"]
    mode_loop  = [(mode_names[i],dataloaders[i]) for i in range(len(dataloaders))]
    val_flag = False
    if len(dataloaders) > 2:
        val_flag = True
    
    # Add Scheduler
    if reduceLR:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LRpatience)
    
    # Set early stopping threshold and counter
    if early_stop is False:
        i_thres = max_epochs
    else:
        i_thres = early_stop
    i_incr    = 0 # Number of epochs for which the validation loss increases
    bestloss  = np.infty
    
    # Preallocation (in the future can allocate this to [3 x max_epoch] array)
    losses = {'train': np.full((max_epochs),np.nan), 'test' : np.full((max_epochs),np.nan),'val' : np.full((max_epochs),np.nan)}
    accs   = {'train': np.full((max_epochs),np.nan), 'test' : np.full((max_epochs),np.nan),'val' : np.full((max_epochs),np.nan)}
    
    # Main Loop
    for epoch in tqdm(range(max_epochs)): # loop by epoch
        for mode,data_loader in mode_loop: # train/test for each epoch
            if mode == 'train':  # Training, update weights
                model.train()
            else: # Testing/Validation, freeze weights
                model.eval()
            
            runningloss = 0
            runningmean = 0
            correct     = 0
            total       = 0
            for i,data in enumerate(data_loader):
                # Get mini batch
                batch_x, batch_y = data
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # Set gradients to zero
                opt.zero_grad()
                
                # Forward pass
                pred_y = model(batch_x)
                _,predicted = torch.max(pred_y.data,1)
                
                # Calculate loss
                loss = loss_fn(pred_y,batch_y[:,0])
                
                # Track accuracy
                total   += batch_y.size(0)
                correct += (predicted == batch_y[:,0]).sum().item()
                #print("Total is now %.2f, Correct is now %.2f" % (total,correct))
                
                # Update weights
                if mode == 'train':
                    loss.backward() # Backward pass to calculate gradients w.r.t. loss
                    opt.step()      # Update weights using optimizer
                elif (mode == 'val') or (val_flag is False and mode == "test"):  # update scheduler after 1st epoch for validation
                    if reduceLR:
                        scheduler.step(loss)
                
                runningloss += float(loss.item()) # Accumulate Loss
                runningmean += correct/total
            
            # Compute the Mean Loss Across Mini-batches
            meanloss_batch = runningloss/len(data_loader) 
            meanacc_batch  = runningmean/len(data_loader) # correct/total
            
            if verbose: # Print progress message
                print('{} Set: Epoch {:02d}. loss: {:3f}. acc: {:.3f}%'.format(mode, epoch+1, \
                                                meanloss_batch,meanacc_batch*100))
            
            # Save model if this is the best loss
            if (meanloss_batch < bestloss) and ((mode == 'val') or (val_flag is False and mode == "test")):
                bestloss  = meanloss_batch
                bestmodel = copy.deepcopy(model)
                if verbose:
                    print("Best Loss of %f at epoch %i"% (bestloss,epoch+1))
            
            # Save running loss values for the epoch
            losses[mode][epoch] = meanloss_batch
            accs[mode][epoch]   = meanacc_batch
            
            # Evaluate if early stopping is needed
            if mode == 'val' or (val_flag is False and mode == "test"):
                if epoch == 0: # Save previous loss
                    lossprev = meanloss_batch
                else: # Add to counter if validation loss increases
                    if meanloss_batch > lossprev:
                        i_incr += 1 # Add to counter
                        if verbose:
                            print("Validation loss has increased at epoch %i, count=%i"%(epoch+1,i_incr))
                    else:
                        i_incr = 0 # Zero out counter
                    lossprev = meanloss_batch
                if (epoch != 0) and (i_incr >= i_thres): # Apply Early stopping and exit script
                    print("\tEarly stop at epoch %i "% (epoch+1))
                    # Decompress dicts (At some point, edit this so that you just return the dictionary...)
                    loss_arr = [losses[md] for md in mode_names]
                    train_loss,test_loss,val_loss = loss_arr
                    acc_arr = [accs[md] for md in mode_names]
                    train_acc,test_acc,val_acc = acc_arr
                    return bestmodel,train_loss,test_loss,val_loss,train_acc,test_acc,val_acc
            # Clear some memory
            #print("Before clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))
            del batch_x
            del batch_y
            torch.cuda.empty_cache()
            
            #print("After clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))
            # <End Train/Test/Val Mode Loop>
        # <End Epoch Loop>
    
    #bestmodel.load_state_dict(best_model_wts)
    # Decompress dicts (At some point, edit this so that you just return the dictionary...)
    loss_arr = [losses[md] for md in mode_names]
    train_loss,test_loss,val_loss = loss_arr
    acc_arr  = [accs[md] for md in mode_names]
    train_acc,test_acc,val_acc = acc_arr
    return bestmodel,train_loss,test_loss,val_loss,train_acc,test_acc,val_acc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ NN Testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_model(model,test_loader,loss_fn,checkgpu=True,debug=False):
    
    # Check if there is GPU
    if checkgpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    
    # Get Predictions
    with torch.no_grad():
        model.eval()
        # -----------------------
        # Test/Evaluate the model
        # -----------------------
        y_predicted  = np.asarray([])
        y_actual     = np.asarray([])
        total_loss   =  0 # track the loss for the prediction
        for i,vdata in enumerate(test_loader):
            
            # Get mini batch
            batch_x, batch_y = vdata     # For debugging: vdata = next(iter(val_loader))
            batch_x = batch_x.to(device) # [batch x input_size]
            batch_y = batch_y.to(device) # [batch x 1]
        
            # Make prediction and concatenate
            batch_pred = model(batch_x)  # [batch x class activation]
            
            # Compute Loss
            loss       = loss_fn(batch_pred,batch_y[:,0])
            total_loss += float(loss.item())
            
            # Convert predicted values
            y_batch_pred = np.argmax(batch_pred.detach().cpu().numpy(),axis=1) # [batch,]
            y_batch_lab  = batch_y.detach().cpu().numpy()            # Removed .squeeze() as it fails when batch size is 1
            y_batch_size = batch_y.detach().cpu().numpy().shape[0]
            if y_batch_size == 1:
                y_batch_lab = y_batch_lab[0,:] # Index to keep as array [1,] instead of collapsing to 0-dim value
            else:
                y_batch_lab = y_batch_lab.squeeze()
            if debug:
                print("Batch Shape on iter %i is %s" % (i,y_batch_size))
                print("\t the shape wihout squeeze is %s" % (batch_y.detach().cpu().numpy().shape[0]))

            # Store Predictions
            y_predicted = np.concatenate([y_predicted,y_batch_pred])
            if debug:
                print("\ty_actual size is %s" % (y_actual.shape))
                print("\ty_batch_lab size is %s" % (y_batch_lab.shape))
            y_actual    = np.concatenate([y_actual,y_batch_lab],axis=0)
            if debug:
                print("\tFinal shape is %s" % y_actual.shape)
    
    # Compute Metrics
    out_loss = total_loss / len(test_loader)
    return y_predicted,y_actual,out_loss

def compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False):
    """
    

    Parameters
    ----------
    y_predicted : ARRAY [samples x 1]
        Predicted target values (class)
    y_actual : ARRAY [samples x 1]
        Actual target values (class)
    nclasses : INT
        Number of clases.
    debug : BOOL, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    total_acc : TYPE
        DESCRIPTION.
    class_acc : TYPE
        DESCRIPTION.

    """
    
    # -------------------------
    # Calculate Success Metrics
    # -------------------------
    # Calculate the total accuracy
    nsamples      = y_predicted.shape[0]
    total_acc     = (y_predicted==y_actual).sum()/ nsamples
    
    # Calculate Accuracy for each class
    class_total   = np.zeros([nclasses])
    class_correct = np.zeros([nclasses])
    for i in range(nsamples):
        class_idx                = int(y_actual[i])
        check_pred               = y_actual[i] == y_predicted[i]
        class_total[class_idx]   += 1
        class_correct[class_idx] += check_pred 
        if verbose:
            print("At element %i, Predicted result for class %i was %s" % (i,class_idx,check_pred))
    class_acc = class_correct/class_total
    
    if debug:
        print("********Success rate********************")
        print("\t" +str(total_acc*100) + r"%")
        print("********Accuracy by Class***************")
        for  i in range(nclasses):
            print("\tClass %i : %03.3f" % (i,class_acc[i]*100) + "%\t" + "(%i/%i)"%(class_correct[i],class_total[i]))
        print("****************************************")
    return total_acc,class_acc


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ NN Creation/Loading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def recreate_model(modelname,nn_param_dict,inputsize,outsize,nlat=180,nlon=360):
    """
    Recreate a NN model for loading weights, based on the modelname, nn_param_dict, inputsize, outsize
    For CNN, need to specify the actual longitude/latitude dimensions (default is 1x1 deg).
    Works with nn_param_dict detailed in predict_amv_params.py

    Parameters
    ----------
    modelname       (STR)   : Name of the Model. See predcit_amv_params for supported models.
    nn_param_dict   (DICT)  : Load this from predict_amv_params. Contains all named model parameters
    inputsize       (INT)   : Size of the input/predictor
    outsize         (INT)   : Size of the model output
    nlat            (INT)   : (optional) Size of latitude input (Y)
    nlon            (INT)   : (optional) Size of longitude input (X)

    Returns
    -------
    pmodel          (torch.NN): Pytorch model with structure loaded (weights are NOT loaded!)

    """
    
    # Retrieve Parameters
    param_dict = nn_param_dict[modelname]
    if "nodropout" in modelname:
        dropout = 0
    else:
        dropout = 0.5
    # Recreate the model
    if "FNN" in modelname:
        layers = build_FNN_simple(inputsize,
                                  outsize,
                                  param_dict["nlayers"],
                                  param_dict["nunits"],
                                  param_dict["activations"],
                                  dropout=dropout)
        pmodel = nn.Sequential(*layers)
    elif modelname == "simplecnn":
        pmodel = build_simplecnn(param_dict["num_classes"],
                                 cnndropout=param_dict["cnndropout"],
                                 unfreeze_all=True,
                                 nlat=nlat,
                                 nlon=nlon,
                                 num_inchannels=param_dict["num_inchannels"])
    elif modelname == "CNN2_LRP":
        pmodel = transfer_model(modelname,param_dict["num_classes"],
                                cnndropout=param_dict["cnndropout"],unfreeze_all=True,
                                nlat=nlat,nlon=nlon,nchannels=param_dict["num_inchannels"])
    return pmodel

def transfer_model(modelname,num_classes,cnndropout=False,unfreeze_all=False
                    ,nlat=224,nlon=224,nchannels=3,param_dict=None):
    """
    Load pretrained weights and architectures based on [modelname]
    
    Parameters
    ----------
    modelname : STR
        Name of model (currently supports 'simplecnn',or any resnet/efficientnet from timms)
    num_classes : INT
        Dimensions of output (ex. number of classes)
    cnndropout : BOOL, optional
        Include dropout layer in simplecnn. The default is False.
    unfreeze_all : BOOL, optional
        Set to True to unfreeze all weights in the model. Otherwise, just
        the last layer is unfrozen. The default is False.
    param_dict : DICT, optional
        For simplecnn_paramdict, see build_simplecnn_fromdict
    
    Returns
    -------
    model : PyTorch Model
        Returns loaded Pytorch model
    """
    
    channels=nchannels
    if 'resnet' in modelname: # Load ResNet
        print("ResNet currently not supported... need to solve compatibility issues with timm. WIP.")
        return None
        # model = timm.create_model(modelname,pretrained=True)
        # if unfreeze_all is False: # Freeze all layers except the last
        #     for param in model.parameters():
        #         param.requires_grad = False
        # model.fc = nn.Linear(model.fc.in_features, num_classes) # Set last layer size
    elif modelname == "simplecnn_paramdict":
        model = build_simplecnn_fromdict(param_dict,num_classes,nlat=nlat,nlon=nlon,num_inchannels=nchannels)
        model = nn.Sequential(*model)
    elif modelname == 'simplecnn': # Use Simple CNN from previous testing framework
        # 2 layer CNN settings
        nchannels     = [32,64]
        filtersizes   = [[2,3],[3,3]]
        filterstrides = [[1,1],[1,1]]
        poolsizes     = [[2,3],[2,3]]
        poolstrides   = [[2,3],[2,3]]
        firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
        if cnndropout: # Include Dropout
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
    
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=64,out_features=num_classes)
                    ]
        else: # Do not include dropout
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),

                    nn.Linear(in_features=64,out_features=num_classes)
                    ]
        model = nn.Sequential(*layers) # Set up model
    elif modelname == "CNN2_LRP":
        nchannels     = [32,64]
        filtersizes   = [[2,3],[3,3]]
        filterstrides = [[1,1],[1,1]]
        poolsizes     = [[2,3],[2,3]]
        poolstrides   = [[2,3],[2,3]]
        firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
        
        model = CNN2(channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes)
    else: # Load Efficientnet from Timmm
        print("timm currently not supported. Need to resolve compatability issues.")
        return
        # model = timm.create_model(modelname,pretrained=True)
        # if unfreeze_all is False: # Freeze all layers except the last
        #     for param in model.parameters():
        #         param.requires_grad = False
        # model.classifier=nn.Linear(model.classifier.in_features,num_classes)
    return model

def build_simplecnn(num_classes,cnndropout=False,unfreeze_all=False
                    ,nlat=224,nlon=224,num_inchannels=3):
    
    # 2 layer CNN settings
    nchannels     = [32,64]
    filtersizes   = [[2,3],[3,3]]
    filterstrides = [[1,1],[1,1]]
    poolsizes     = [[2,3],[2,3]]
    poolstrides   = [[2,3],[2,3]]
    firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
    if cnndropout: # Include Dropout
        layers = [
                nn.Conv2d(in_channels=num_inchannels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[0]),

                nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[1]),

                nn.Flatten(),
                nn.Linear(in_features=firstlineardim,out_features=64),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),

                nn.Dropout(p=0.5),
                nn.Linear(in_features=64,out_features=num_classes)
                ]
    else: # Do not include dropout
        layers = [
                nn.Conv2d(in_channels=num_inchannels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[0]),

                nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[1]),

                nn.Flatten(),
                nn.Linear(in_features=firstlineardim,out_features=64),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),

                nn.Linear(in_features=64,out_features=num_classes)
                ]
    model = nn.Sequential(*layers) # Set up model
    return model

def build_simplecnn_fromdict(param_dict,num_classes,nlat=224,nlon=224,num_inchannels=3):
    # Same as above, but using a parameter dictionary. See "gridsearch_CNN.py"
    
    # 2 layer CNN settings
    nchannels      = param_dict['nchannels']#
    filtersizes    = param_dict['filtersizes']
    filterstrides  = param_dict['filterstrides']
    poolsizes      = param_dict['poolsizes']#[[2,3],[2,3]]
    poolstrides    = param_dict['poolstrides']#[[2,3],[2,3]]
    activations    = param_dict['activations']
    dropout        = param_dict['dropout']
    firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
    
    layers = []
    nlayers = len(nchannels)
    
    for l in range(nlayers):
        if l == 0: # 1st Layer
            # Make + Append Convolutional Layer
            conv_layer = nn.Conv2d(in_channels=num_inchannels,out_channels=nchannels[l], kernel_size=filtersizes[l], stride=filterstrides[l])
            layers.append(conv_layer)
        else: # All other layers
            # Make + Append Convolutional Layer
            conv_layer = nn.Conv2d(in_channels=nchannels[l-1], out_channels=nchannels[l], kernel_size=filtersizes[l], stride=filterstrides[l])
            layers.append(conv_layer)
        
        # Append Activation
        layers.append(activations[l])
        
        # Make+Append Pooling layer
        pool_layer = nn.MaxPool2d(kernel_size=poolsizes[l], stride=poolstrides[l])
        layers.append(pool_layer)
        
        if l == (nlayers-1): # Final Layer (Flatten and add Fully Connected)
            layers.append(nn.Flatten())
            layers.append(nn.Dropout(p=dropout))
            linear_layer = nn.Linear(in_features=firstlineardim,out_features=num_classes)
            layers.append(linear_layer)
    return layers

def build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0.5,
                     use_softmax=False):
    """
    Build a Feed-foward neural network with N layers, each with corresponding
    number of units indicated in nunits and activations. 
    
    A dropbout layer is included at the end
    
    inputs:
        inputsize:  INT - size of the input layer
        outputsize: INT  - size of output layer
        nlayers:    INT - number of hidden layers to include 
        nunits:     Tuple of units in each layer
        activations: Tuple of pytorch.nn activations
        --optional--
        dropout: percentage of units to dropout before last layer
        use_softmax : BOOL, True to end with softmax layer
        
    outputs:
        Tuple containing FNN layers
        
    dependencies:
        from pytorch import nn
        
    """
    # Check nunits and duplicate if it is not the same
    if type(nunits) == int:
        nunits = [nunits,] * nlayers
        
    # Verify each later to check that the size matches
    while len(activations) < nlayers:
        print("Warning: Not all activations were specified. Duplicating the last layer")
        activations.append(activations[-1]) # Just Append the Last Activation
    
    layers = []
    for n in range(nlayers+1):
        #print(n)
        if n == 0:
            #print("First Layer")
            layers.append(nn.Linear(inputsize,nunits[n]))
            layers.append(activations[n])
            
        elif n == (nlayers):
            #print("Last Layer")
            if use_softmax:
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(nunits[n-1],outsize))
                layers.append(nn.Softmax(dim=0))
            else:
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(nunits[n-1],outsize))
            
        else:
            #print("Intermediate")
            layers.append(nn.Linear(nunits[n-1],nunits[n]))
            layers.append(activations[n])
    return layers

def calc_layerdims(nx,ny,filtersizes,filterstrides,poolsizes,poolstrides,nchannels):
    """
    For a series of N convolutional layers, calculate the size of the first fully-connected
    layer

    Inputs:
        nx:           x dimensions of input
        ny:           y dimensions of input
        filtersize:   [ARRAY,length N] sizes of the filter in each layer [(x1,y1),[x2,y2]]
        poolsize:     [ARRAY,length N] sizes of the maxpooling kernel in each layer
        nchannels:    [ARRAY,] number of out_channels in each layer
    output:
        flattensize:  flattened dimensions of layer for input into FC layer

    """
    N = len(filtersizes)
    xsizes = [nx]
    ysizes = [ny]
    fcsizes  = []
    for i in range(N):
        
        
        # Apply initial convolution
        xsizes.append(np.floor((xsizes[i]-filtersizes[i][0])/filterstrides[i][0])+1)
        ysizes.append(np.floor((ysizes[i]-filtersizes[i][1])/filterstrides[i][1])+1)
        
        if i > (len(poolsizes)-1):
            fcsizes.append(np.floor(xsizes[i]*ysizes[i]*nchannels[i]))
            continue
        else: # Apply pooling if needed
            xsizes[i+1] = np.floor((xsizes[i+1] - poolsizes[i][0])/poolstrides[i][0]+1)
            ysizes[i+1] = np.floor((ysizes[i+1] - poolsizes[i][1])/poolstrides[i][1]+1)
            fcsizes.append(np.floor(xsizes[i+1]*ysizes[i+1]*nchannels[i]))
    #print("Dimension at layer %i is %i" % (i,np.floor(xsizes[i+1]*ysizes[i+1]*nchannels[i])))
        
        
    return int(fcsizes[-1])

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#%% Analysis
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ General Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_LRP_composites(topN,in_acc,correct_id,relevances,absval=False,normalize_sample=0):
    """
    topN        [INT]           : Top N models to composite
    in_acc      [ARRAY]         : Accuracies of each model [models]
    correct_id  [LIST]          : Indices of correct predictions for a given class [model][correct_samples]
    relevances  [ARRAY]         : Relevances to composite [model x sample x channels x lat x lon]
    absval      [BOOL]          : Set to True to take absolute value of relevances prior to compositing
    normalize_sample [BOOL]     : Set to True to normalize relevances in each sample to [0,1]
    Output
    ------
    composite_rel [ARRAY]       : Composited relevances [lat x lon]
    """
    # Get indices
    idtopN  = get_topN(in_acc,topN,sort=True)
    id_plot = np.array(correct_id)[idtopN]
    # Preallocate
    nlat    = relevances.shape[3]
    nlon    = relevances.shape[4]
    composite_rel = np.zeros((nlat,nlon))
    for NN in range(topN): # Select correct samples for each model
        relevances_sel = relevances[idtopN[NN],id_plot[NN],:,:,:].squeeze() # [Correct_Samples x Channel x Lat x Lon]
        if normalize_sample:
            relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...] # Divide by max relevance
        if absval:
            relevances_sel = np.abs(relevances_sel)
        composite_rel        += relevances_sel.mean(0) # Add Composite for each model
    composite_rel /= topN
    return composite_rel

def calc_confmat(ypred,ylabel,c,getcounts=True,debug=True):
    """
    Calculate Confusion Matrices
      TP  FP
            
      FN  TN
    
    ypred     : [N x 1]
    ylabel    : [N x 1]
    c         : the class number or label, as found in ypred/ylabel
    getcounts : Set True to return indices,counts,total_counts,accuracy
    debug     : Set True to print debugging messages
    """
    
    nsamples = ypred.shape[0]
    TP       = ((ypred==c) * (ylabel==c))
    FP       = ((ypred==c) * (ylabel!=c))
    TN       = ((ypred!=c) * (ylabel!=c))
    FN       = ((ypred!=c) * (ylabel==c))
    cm       = np.array([TP,FP,FN,TN],dtype='object') # [4,#samples]
    
    if debug:
        TP,FP,FN,TN = cm
        print("Predict 0: %i, Actual 0: %i, TN Count: %i " % ((ypred!=c).sum(),(ylabel!=c).sum(),TN.sum())) # Check True Negative 
        print("Predict 1: %i, Actual 1: %i, TP Count: %i " % ((ypred==c).sum(),(ylabel==c).sum(),TP.sum())) # Check True Positive
        print("Predict 1: %i, Actual 0: %i, FP Count: %i (+ %i = %i total)" % ((ypred==c).sum(),(ylabel!=c).sum(),FP.sum(),TP.sum(),FP.sum()+TP.sum())) # Check False Positive
        print("Predict 0: %i, Actual 1: %i, FN Count: %i (+ %i = %i total)" % ((ypred!=c).sum(),(ylabel==c).sum(),FN.sum(),TN.sum(),FN.sum()+TN.sum())) # Check False Negative
    if getcounts: # Get counts and accuracy
        cm_counts               = np.array([c.sum() for c in cm])#.reshape(2,2)
        #cm                      = cm.reshape(2,2,nsamples) #
        count_pred_total        = np.ones(4) * nsamples #np.ones((2,2)) * nsamples  #np.vstack([np.ones(2)*(ypred==c).sum(),np.ones(2)*(ypred!=c).sum()]) # [totalpos,totalpos,totalneg,totalneg]
        cm_acc                  = cm_counts / nsamples     #count_pred_total
        return cm,cm_counts,count_pred_total,cm_acc
    else:
        return cm.reshape(4,nsamples)#.reshape(2,2,nsamples)
    
def calc_confmat_loop(y_pred,y_class):
    """
    Given predictions and labels, retrieves confusion matrix indices in the
    following order for each class: 
        [True Positive, False Positive, False Negative, True Positive]
    
    Parameters
    ----------
    y_pred : ARRAY [nsamples x 1]
        Predicted Class.
    y_class : ARRAY[nsamples x 1]
        Actual Class.
        
    Returns
    -------
    cm_ids :    ARRAY [Class,Confmat_qudrant,Indices]
        Confusion matrix Boolean indices
    cm_counts : ARRAY [Class,Confmat_qudrant]
        Counts of predicted values for each quadrant
    cm_totals : ARRAY [Class,Confmat_qudrant]
        Total count (for division)
    cm_acc :    ARRAY [Class,Confmat_qudrant]
        Accuracy values
    cm_names :  ARRAY [Class,Confmat_qudrant]
        Names of each confmat quadrant
    """
    nsamples = y_pred.shape[0]
    
    # Preallocate for confusion matrices
    cm_ids     = np.empty((3,4,nsamples),dtype='object')# Confusion matrix Boolean indices [Class,Confmat_quadrant,Indices]
    cm_counts  = np.empty((3,4),dtype='object')         # Counts of predicted values for each [Class,Actual_class,Pred_class]
    cm_totals  = cm_counts.copy() # Total count (for division)
    cm_acc     = cm_counts.copy() # Accuracy values
    cm_names   = ["TP","FP","FN","TN"] # Names of each
    
    for th in range(3):
        # New Script
        confmat,ccounts,tcounts,acc = calc_confmat(y_pred,y_class,th)
        cm_ids[th,:]    = confmat.copy().squeeze()
        cm_counts[th,:] = ccounts.copy()
        cm_totals[th,:] = tcounts.copy()
        cm_acc[th,:]    = acc.copy()
    return cm_ids,cm_counts,cm_totals,cm_acc,cm_names
        

def get_topN(arr,N,bot=False,sort=False,absval=False):
    """
    Copied from proc on 2022.11.01
    Get the indices for the top N values of an array.
    Searches along the last dimension. Option to sort output.
    Set [bot]=True for the bottom 5 values
    
    Parameters
    ----------
    arr : TYPE
        Input array with partition/search dimension as the last axis
    N : INT
        Top or bottom N values to find
    bot : BOOL, optional
        Set to True to find bottom N values. The default is False.
    sort : BOOL, optional
        Set to True to sort output. The default is False.
    absval : BOOL, optional
        Set to True to apply abs. value before sorting. The default is False.
        
    Returns
    -------
    ids : ARRAY
        Indices of found values
    """
    
    if absval:
        arr = np.abs(arr)
    if bot is True:
        ids = np.argpartition(arr,N,axis=-1)[...,:N]
    else:
        ids = np.argpartition(arr,-N,axis=-1)[...,-N:]
         # Parition up to k, and take first k elements
    if sort:
        if bot:
            return ids[np.argsort(arr[ids])] # Least to greatest
        else:
            return ids[np.argsort(-arr[ids])] # Greatest to least
    return ids

def get_barcount(y_in,axis=0):

    # Get Existing Classes
    classes  = np.unique(y_in.flatten())
    nclasses = len(classes)

    # Get count of each class along an axis
    counts = []
    for c in range(nclasses):
        count_tot = (y_in==c).sum(axis) # [time,]
        counts.append(count_tot)
    return counts

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Metrics Output Handling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_result(fn,debug=False,load_dict=False):
    """
    Load results for each of the variable names (testacc, etc)
    input: fn (str), Name of the file
    
    vnames/output are:
        train_loss
        test_loss
        val_loss
        train_acc
        test_acc
        val_acc
        total_acc
        acc_by_class
        yvalpred
        yvallabels
        sampled_idx
        thresholds_all
        exp_params
        sample_sizes
    
    Copied from viz_acc_by_predictor.py on 2023.01.25
    """
    
    ld     = np.load(fn,allow_pickle=True)
    vnames = ld.files
    if debug:
        print(vnames)
    output = []
    for v in vnames:
        output.append(ld[v])
    if load_dict:
        return ld,vnames
    return output,vnames

def load_metrics_byrun(flist,leads,debug=False,runmax=None,no_val=True,use_dict=True):
    """
    Given a list of metric files [flist] and leadtimes for each training run,
    Load the output and append.
    Dependencies: load_result()
    """
    flist.sort()
    if runmax is None:
        nruns = len(flist)
    else:
        nruns = runmax
    # Load Result for each model training run
    totalm    = [] # Total Test Accuracy
    classm    = [] # Test Accuracy by Class
    ypredm    = [] # Predicted Class
    ylabsm    = [] # Actual Class
    shuffidsm = [] # Shuffled Indices
    for i in range(nruns): # Load for [nruns] files
        output,vnames = load_result(flist[i],debug=debug,load_dict=True)
        output_ori,vnames_ori = load_result(flist[i],debug=debug)
        # if len(output[4]) > len(leads):
        #     print("Selecting Specific Leads!")
        #     output = [out[leads] for out in output]
        if use_dict: # no_val has no use in this case
            totalm.append(output['total_acc'])
            classm.append(output['acc_by_class'])
            ypredm.append(output['yvalpred'])
            ylabsm.append(output['yvallabels'])
            shuffidsm.append(output['sampled_idx'])
        else:
            if no_val:
                totalm.append(output_ori[4])
                classm.append(output_ori[5])
                ypredm.append(output_ori[6])
                ylabsm.append(output_ori[7])
                shuffidsm.append(output_ori[8])
                if debug: # Check to make sure they are the same
                    for iii in range(4,9,1):
                        print(np.all(output[vnames[iii]][0] == output_ori[iii][0]))
            else:
                totalm.append(output_ori[6])
                classm.append(output_ori[7])
                ypredm.append(output_ori[8])
                ylabsm.append(output_ori[9])
                shuffidsm.append(output_ori[10])
                if debug: # Check to make sure they are the same
                    for iii in range(6,11,1):
                        print(np.all(output[vnames[iii]][0] == output_ori[iii][0]))
        print("\tLoaded %s, %s, %s, and %s for run %02i" % (vnames[4],vnames[5],vnames[6],vnames[7],i))
    return totalm,classm,ypredm,ylabsm,shuffidsm,vnames
    
def make_expdict(flists,leads,no_val=True):
    """
    Given a nested list of metric files for the 
    training runs for each experiment, ([experiment][run]),
    Load out the data into arrays and create and experiment dictionary for analysis
    This data can later be unpacked by unpack_expdict
    Set no_val=True to use old loading format, where validation data is not included. 
    
    Contents of expdict: 
        totalacc = [] # Accuracy for all classes combined [exp x run x leadtime]
        classacc = [] # Accuracy by class                 [exp x run x leadtime x class]
        ypred    = [] # Predictions                       [exp x run x leadtime x sample]
        ylabs    = [] # Labels                            [exp x run x leadtime x sample]
        shuffids = [] # Indices                           [exp x run x leadtime x sample]
    
    Dependencies: 
        - load_metrics_byrun
        - load_result
    """
    # Check the # of runs
    nruns = [len(f) for f in flists]
    if len(np.unique(nruns)) > 1:
        print("Warning, limiting experiments to %i runs" % np.min(nruns))
    runmax = np.min(nruns)
    
    # Preallocate
    totalacc = [] # Accuracy for all classes combined [exp x run x leadtime]
    classacc = [] # Accuracy by class                 [exp] [run x leadtime x class]
    ypred    = [] # Predictions                       [exp x run x leadtime x sample] # Last array (tercile based) is not an even sample size...
    ylabs    = [] # Labels                            [exp x run x leadtime x sample]
    shuffids = [] # Indices                           [exp x run x leadtime x sample]
    for exp in range(len(flists)):
        # Load metrics for a given experiment
        if isinstance(no_val,list):
            in_no_val = no_val[exp]
        else:
            in_no_val = no_val
        exp_metrics = load_metrics_byrun(flists[exp],leads,runmax=runmax,no_val=in_no_val)
        
        # Load out and append variables
        totalm,classm,ypredm,ylabsm,shuffidsm,vnames = exp_metrics
        totalacc.append(totalm)
        classacc.append(classm)
        ypred.append(ypredm)
        ylabs.append(ylabsm)
        shuffids.append(shuffidsm)
        print("Loaded data for experiment %02i!" % (exp+1))
    
    # Add to dictionary
    outputs = [totalacc,classacc,ypred,ylabs,shuffids]
    expdict = {}
    dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
    
    for k,key in enumerate(dictkeys):
        expdict[key] = outputs[k] # Removed np.array(output[k])
    return expdict

def unpack_expdict(expdict,dictkeys=None):
    """
    Unpack expdict generated by load_result from the metrics file
    
    Copied from viz_acc_by_predictor.py on 2023.01.25
    
    """
    if dictkeys is None:
        dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
    unpacked = [expdict[key] for key in expdict]
    return unpacked

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#%% Network Architectures (maybe move to a different script)
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
class CNN2(nn.Module):
    
    def __init__(self,channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels=channels,out_channels=nchannels[0] ,kernel_size=filtersizes[0])
        self.pool1  = nn.MaxPool2d(kernel_size=poolsizes[0])
        self.activ1 = nn.ReLU()
        self.conv2  = nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1])
        self.pool2  = nn.MaxPool2d(kernel_size=poolsizes[1])
        self.activ2 = nn.ReLU()
        self.fc1    = nn.Linear(in_features=firstlineardim,out_features=64)
        self.activ3 = nn.ReLU()
        self.fc2    = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activ2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.activ3(x)
        x = self.fc2(x)
        return x

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#%% amv.proc.copy (eventually delete this after redirecting the reference)
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to sum along. 0 or 1
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        
    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
        elif dim == 1:    
            okdata = data[okpts,:]
    else:
        okdata = data[okpts]
        
    return okdata,knan,okpts


def eof_simple(pattern,N_mode,remove_timemean):
    """
    Simple EOF function based on script by Yu-Chiao
    
    
    Inputs:
        1) pattern: Array of Space x Time [MxN], no NaNs
        2) N_mode:  Number of Modes to output
        3) remove_timemean: Set 1 to remove mean along N
    
    Outputs:
        1) eof: EOF patterns   [M x N_mode]
        2) pcs: PC time series [N x N_mode]
        3) varexp: % Variance explained [N_mode]
    
    Dependencies:
        import numpy as np
    
    """
    pattern1 = pattern.copy()
    nt = pattern1.shape[1] # Get time dimension size
    ns = pattern1.shape[0] # Get space dimension size
    
    # Preallocate
    eofs = np.zeros((ns,N_mode))
    pcs  = np.zeros((nt,N_mode))
    varexp = np.zeros((N_mode))
    
    # Remove time mean if option is set
    if remove_timemean == 1:
        pattern1 = pattern1 - pattern1.mean(axis=1)[:,None] # Note, the None adds another dimension and helps with broadcasting
    
    # Compute SVD
    [U, sigma, V] = np.linalg.svd(pattern1, full_matrices=False)
    
    # Compute variance (total?)
    norm_sq_S = (sigma**2).sum()
    
    for II in range(N_mode):
        
        # Calculate explained variance
        varexp[II] = sigma[II]**2/norm_sq_S
        
        # Calculate PCs
        pcs[:,II] = np.squeeze(V[II,:]*np.sqrt(nt-1))
        
        # Calculate EOFs and normalize
        eofs[:,II] = np.squeeze(U[:,II]*sigma[II]/np.sqrt(nt-1))
    return eofs, pcs, varexp

def coarsen_byavg(invar,lat,lon,deg,tol,bboxnew=False,latweight=True,verbose=True):
    """
    Coarsen an input variable to specified resolution [deg]
    by averaging values within a search tolerance for each new grid box.
    To take the area-weighted average, set latweight=True
    
    Dependencies: numpy as np

    Parameters
    ----------
    invar : ARRAY [TIME x LAT x LON]
        Input variable to regrid
    lat : ARRAY [LAT]
        Latitude values of input
    lon : ARRAY [LON]
        Longitude values of input
    deg : INT
        Resolution of the new grid (in degrees)
    tol : TYPE
        Search tolerance (pulls all lat/lon +/- tol)
    
    OPTIONAL ---
    bboxnew : ARRAY or False
        New bounds to regrid in order - [lonW, lonE, latS, latN]
        Set to False to pull directly from first and last coordinates
    latweight : BOOL
        Set to true to apply latitude weighted-average
    verbose : BOOL
        Set to true to print status
    

    Returns
    -------
    outvar : ARRAY [TIME x LAT x LON]
        Regridded variable       
    lat5 : ARRAY [LAT]
        New Latitude values of input
    lon5 : ARRAY [LON]
        New Longitude values of input

    """

    # Make new Arrays
    if not bboxnew:
        lon5 = np.arange(lon[0],lon[-1]+deg,deg)
        lat5 = np.arange(lat[0],lat[-1]+deg,deg)
    else:
        lon5 = np.arange(bboxnew[0],bboxnew[1]+deg,deg)
        lat5 = np.arange(bboxnew[2],bboxnew[3]+deg,deg)
    
    # Check to see if any longitude values are degrees Easy
    if any(lon>180):
        lonflag = True
    
    # Set up latitude weights
    if latweight:
        _,Y = np.meshgrid(lon,lat)
        wgt = np.cos(np.radians(Y)) # [lat x lon]
        invar *= wgt[None,:,:] # Multiply by latitude weight
    
    # Get time dimension and preallocate
    nt = invar.shape[0]
    outvar = np.zeros((nt,len(lat5),len(lon5)))
    
    # Loop and regrid
    i=0
    for o in range(len(lon5)):
        for a in range(len(lat5)):
            lonf = lon5[o]
            latf = lat5[a]
            
            # check longitude
            if lonflag:
                if lonf < 0:
                    lonf+=360
            
            lons = np.where((lon >= lonf-tol) & (lon <= lonf+tol))[0]
            lats = np.where((lat >= latf-tol) & (lat <= latf+tol))[0]
            
            varf = invar[:,lats[:,None],lons[None,:]]
            
            if latweight:
                wgtbox = wgt[lats[:,None],lons[None,:]]
                varf = np.sum(varf/np.sum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
            else:
                varf = varf.mean((1,2))
            outvar[:,a,o] = varf.copy()
            i+= 1
            msg="\rCompleted %i of %i"% (i,len(lon5)*len(lat5))
            print(msg,end="\r",flush=True)
    return outvar,lat5,lon5

def regress_2d(A,B,nanwarn=1):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
    
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    
    return beta,b

def sel_region(var,lon,lat,bbox,reg_avg=0,reg_sum=0,warn=1):
    """
    
    Select Region
    
    Inputs
        1) var: ARRAY, variable with dimensions [lon x lat x otherdims]
        2) lon: ARRAY, Longitude values
        3) lat: ARRAY, Latitude values
        4) bbox: ARRAY, bounding coordinates [lonW lonE latS latN]
        5) reg_avg: BOOL, set to 1 to return regional average
        6) reg_sum: BOOL, set to 1 to return regional sum
        7) warn: BOOL, set to 1 to print warning text for region selection
    Outputs:
        1) varr: ARRAY: Output variable, cut to region
        2+3), lonr, latr: ARRAYs, new cut lat/lon
    
    
    """    
        
    # Find indices
    klat = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
    if bbox[0] < bbox[1]:
        klon = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
    elif bbox[0] > bbox[1]:
        if warn == 1:
            print("Warning, crossing the prime meridian!")
        klon = np.where((lon <= bbox[1]) | (lon >= bbox[0]))[0]
    
    
    lonr = lon[klon]
    latr = lat[klat]
    
    #print("Bounds from %.2f to %.2f Latitude and %.2f to %.2f Longitude" % (latr[0],latr[-1],lonr[0],lonr[-1]))
        
    
    # Index variable
    varr = var[klon[:,None],klat[None,:],...]
    
    if reg_avg==1:
        varr = np.nanmean(varr,(0,1))
        return varr
    elif reg_sum == 1:
        varr = np.nansum(varr,(0,1))
        return varr
    return varr,lonr,latr

def calc_AMV(lon,lat,sst,bbox,order,cutofftime,awgt,lpf=1):
    """
    Calculate AMV Index for detrended/anomalized SST data [LON x LAT x Time]
    given bounding box [bbox]. Applies area weight based on awgt
    
    Parameters
    ----------
    lon : ARRAY [LON]
        Longitude values
    lat : ARRAY [LAT]
        Latitude Values
    sst : ARRAY [LON x LAT x TIME]
        Sea Surface Temperature
    bbox : ARRAY [LonW,LonE,LonS,LonN]
        Bounding Box for Area Average
    order : INT
        Butterworth Filter Order
    cutofftime : INT
        Filter Cutoff, expressed in same timesteps as input data
    awgt : INT
        0 = No weight, 1 = cos(lat), 2 = sqrt(cos(lat))
        
    Returns
    -------
    amv: ARRAY [TIME]
        AMV Index (Not Standardized)
    
    aa_sst: ARRAY [TIME]
        Area Averaged SST

    # Dependencies
    functions
        area_avg
    modules
        numpy as np
        from scipy.signal import butter,filtfilt
    """
    
    # Take the weighted area average
    aa_sst = area_avg(sst,bbox,lon,lat,awgt)

    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Compute AMV Index
    amv = filtfilt(b,a,aa_sst)

    return amv,aa_sst


def detrend_poly(x,y,deg):
    """
    Matrix for of polynomial detrend
    # Based on :https://stackoverflow.com/questions/27746297/detrend-flux-time-series-with-non-linear-trend
    
    Inputs:
        1) x --> independent variable
        2) y --> 2D Array of dependent variables
        3) deg --> degree of polynomial to fit
    
    """
    # Transpose to align dimensions for polyfit
    if len(y) != len(x):
        y = y.T
    
    # Get the fit
    fit = np.polyfit(x,y,deg=deg)
    
    # Prepare matrix (x^n, x^n-1 , ... , x^0)
    #inputs = np.array([np.power(x,d) for d in range(len(fit))])
    inputs = np.array([np.power(x,d) for d in reversed(range(len(fit)))])
    # Calculate model
    model = fit.T.dot(inputs)
    # Remove trend
    ydetrend = y - model.T
    return ydetrend,model

def lon360to180(lon360,var):
    """
    Convert Longitude from Degrees East to Degrees West 
    Inputs:
        1. lon360 - array with longitude in degrees east
        2. var    - corresponding variable [lon x lat x time]
    """
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    var = np.concatenate((var[kw,...],var[ke,...]),0)
    
    return lon180,var


def area_avg(data,bbox,lon,lat,wgt):
    
    """
    Function to find the area average of [data] within bounding box [bbox], 
    based on wgt type (see inputs)
    
    Inputs:
        1) data: target array [lat x lon x otherdims]
        2) bbox: bounding box [lonW, lonE, latS, latN]
        3) lon:  longitude coordinate
        4) lat:  latitude coodinate
        5) wgt:  number to indicate weight type
                    0 = no weighting
                    1 = cos(lat)
                    2 = sqrt(cos(lat))
    
    Output:
        1) data_aa: Area-weighted array of size [otherdims]
        
    Dependencies:
        numpy as np
    

    """
        
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
        
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # If wgt == 1, apply area-weighting 
    if wgt != 0:
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        
        # Calculate Area Weights (cosine of latitude)
        if wgt == 1:
            wgta = np.cos(np.radians(yy)).T
        elif wgt == 2:
            wgta = np.sqrt(np.cos(np.radians(yy))).T
        
        # Remove nanpts from weight, ignoring any pt with nan in otherdims
        nansearch = np.sum(sel_data,2) # Sum along otherdims
        wgta[np.isnan(nansearch)] = 0
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]

    
    # Take average over lon and lat
    if wgt != 0:

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
        data_aa = np.nanmean(sel_data,(0,1))
    
    return data_aa

def regress2ts(var,ts,normalizeall=0,method=1,nanwarn=1):
    """
    regress variable var [lon x lat x time] to timeseries ts [time]
    
    Parameters
    ----------
    var : TYPE
        DESCRIPTION.
    ts : TYPE
        DESCRIPTION.
    normalizeall : TYPE, optional
        DESCRIPTION. The default is 0.
    method : TYPE, optional
        DESCRIPTION. The default is 1.
    nanwarn : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    var_reg : TYPE
        DESCRIPTION.
    
    """

    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
        
    # Get variable shapes
    londim = var.shape[0]
    latdim = var.shape[1]
    
    # 1st method is matrix multiplication
    if method == 1:
        
        # Combine the spatial dimensions 

        var = np.reshape(var,(londim*latdim,var.shape[2]))
        
        
        # Find Nan Points
        # sumvar = np.sum(var,1)
        
        # # Find indices of nan pts and non-nan (ok) pts
        # nanpts = np.isnan(sumvar)
        # okpts  = np.invert(nanpts)
    
        # # Drop nan pts and reshape again to separate space and time dimensions
        # var_ok = var[okpts,:]
        #var[np.isnan(var)] = 0
        
        
        # Perform regression
        #var_reg = np.matmul(np.ma.anomalies(var,axis=1),np.ma.anomalies(ts,axis=0))/len(ts)
        var_reg,_ = regress_2d(ts,var,nanwarn=nanwarn)
        
        
        # Reshape to match lon x lat dim
        var_reg = np.reshape(var_reg,(londim,latdim))
    
    
    
    
    # 2nd method is looping point by point
    elif method == 2:
        
        
        # Preallocate       
        var_reg = np.zeros((londim,latdim))
        
        # Loop lat and long
        for o in range(londim):
            for a in range(latdim):
                
                # Get time series for that period
                vartime = np.squeeze(var[o,a,:])
                
                # Skip nan points
                if any(np.isnan(vartime)):
                    var_reg[o,a]=np.nan
                    continue
                
                # Perform regression 
                r = np.polyfit(ts,vartime,1)
                #r=stats.linregress(vartime,ts)
                var_reg[o,a] = r[0]
                #var_reg[o,a]=stats.pearsonr(vartime,ts)[0]
    
    return var_reg

## Plotting ----
def plot_AMV(amv,ax=None):
    
    """
    Plot amv time series
    
    Dependencies:
        
    matplotlib.pyplot as plt
    numpy as np
    """
    if ax is None:
        ax = plt.gca()
    
    
    htimefull = np.arange(len(amv))
    
    ax.plot(htimefull,amv,color='k')
    ax.fill_between(htimefull,0,amv,where=amv>0,facecolor='red',interpolate=True,alpha=0.5)
    ax.fill_between(htimefull,0,amv,where=amv<0,facecolor='blue',interpolate=True,alpha=0.5)

    return ax

def plot_AMV_spatial(var,lon,lat,bbox,cmap,cint=[0,],clab=[0,],ax=None,pcolor=0,labels=True,fmt="%.1f",clabelBG=False,fontsize=10):
    fig = plt.gcf()
    
    if ax is None:
        ax = plt.gca()
        ax = plt.axes(projection=ccrs.PlateCarree())
        
    # Add cyclic point to avoid the gap
    var,lon1 = add_cyclic_point(var,coord=lon)
    

    
    # Set  extent
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.LAND,color=[0.4,0.4,0.4])
    
    
    if len(cint) == 1:
        # Automaticall set contours to max values
        cmax = np.nanmax(np.abs(var))
        cmax = np.round(cmax,decimals=2)
        cint = np.linspace(cmax*-1,cmax,9)
    
    
    
    if pcolor == 0:

        # Draw contours
        cs = ax.contourf(lon1,lat,var,cint,cmap=cmap)
    
    
    
        # Negative contours
        cln = ax.contour(lon1,lat,var,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())
    
        # Positive Contours
        clp = ax.contour(lon1,lat,var,
                    cint[cint>=0],
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())    
                          
        if labels is True:
            clabelsn= ax.clabel(cln,colors=None,fmt=fmt,fontsize=fontsize)
            clabelsp= ax.clabel(clp,colors=None,fmt=fmt,fontsize=fontsize)
            
            # if clabelBG is True:
            #     [txt.set_backgroundcolor('white') for txt in clabelsn]
            #     [txt.set_backgroundcolor('white') for txt in clabelsp]
    else:
        
        cs = ax.pcolormesh(lon1,lat,var,vmin = cint[0],vmax=cint[-1],cmap=cmap)
        
                                
                
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    gl.xlabel_style={'size':8}
    gl.ylabel_style={'size':8}
    if len(clab) == 1:
        cbar= fig.colorbar(cs,ax=ax,fraction=0.046, pad=0.04,format=fmt)
        cbar.ax.tick_params(labelsize=8)
    else:
        cbar = fig.colorbar(cs,ax=ax,ticks=clab,fraction=0.046, pad=0.04,format=fmt)
        cbar.ax.tick_params(labelsize=8)
    #cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in cint], fontsize=10, weight='bold')
    
    return ax

def deseason_lazy(ds,return_scycle=False):
    """
    Deseason function without reading out the values. Remove the seasonal cycle by subtracting the monthly anomalies
    Input:
        ds : DataArray
            Data to be deseasoned
        return_scycle : BOOL (Optional)
            Set to true to return the seasonal cycle that was removed
    Output:
        data_deseason : DataArray
            Deseasoned data
    """
    data_deseason = ds.groupby('time.month') - ds.groupby('time.month').mean('time')
    
    if return_scycle:
        return data_deseason,ds.groupby('time.month').mean('time')
    return data_deseason

def init_map(bbox,crs=ccrs.PlateCarree(),ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox,crs)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND,facecolor='k',zorder=-1)
    
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.top_labels = gl.right_labels = False
    

    
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    
    return ax

        