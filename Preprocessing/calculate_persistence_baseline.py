#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the persistence baseline, based on [Persistence_Classification_Baseline.y]
but generalized to include other datasets.

Created on Thu Apr  6 21:13:24 2023

@author: gliu
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# -------------------------------
# %% Import Experiment Parameters
# -------------------------------

import os
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl 

# ---------------------------------
#%% User Edits/Specifications
# ---------------------------------

# Dataset information
dataset_name  = "CESM1" # [Currently supports HadISST, CESM1]
region        = "NAT"     # Region over which Index was computed
detrend       = True     # Whether or not data was detrended
quantile      = False
thresholds    = [-1,1]

# Arguments necessary for baseline computations
leads         = np.arange(0,26,1)
nsamples      = None # Subsampling of each class
percent_train = 1    # Percentage of training data
repeat_calc   = 1  # Number of times to resample and repeat the calculations
save_baseline = True
test_ens      = np.arange(42)[-10:] # Indicate indices of ensemble members reserved for testing (CESM1)

outpath       = "../../CESM_Data/Baselines/"
savename      = "%spersistence_baseline_%s_%s_detrend%i_quantile%i_nsamples%s_repeat%i.npz" % (outpath,dataset_name,
                                                                                            region,detrend,
                                                                                            quantile,nsamples,repeat_calc)
if save_baseline:
    print("Data will be saved to %s" % savename)
# ---------------
#%% Load the data
# ---------------
print("Loading data for %s..." % dataset_name)
if dataset_name == "HadISST":
    # Load the target dataset
    target = dl.load_target_reanalysis(dataset_name,region,detrend=detrend)
    target = target[None,:] # ens x year
elif dataset_name == "CESM1":
    
    target = dl.load_target_cesm(detrend=detrend,region=region,newpath=True,norm=True)

    
else:
    print("This script only supports computing baselines for: [CESM1, HadISST]")
    
nens,ntime = target.shape
# --------------------------------------------------
# %% Preprocess and classify based on specifications
# --------------------------------------------------


# Get Standard Deviation Threshold
print("Original thresholds are %i stdev" % thresholds[0])
std1   = target.std(1).mean() * thresholds[1] # Multiple stdev by threshold value 
if quantile is False:
    in_thresholds = [-std1,std1]
    print(r"Setting Thresholds to +/- %.2f" % (std1))
else:
    in_thresholds = thresholds

# Convert target to class
y       = target[:nens,:].reshape(nens*ntime,1)
y_class = am.make_classes(y,in_thresholds,reverse=True,exact_value=True,quantiles=quantile)
y_class = y_class.reshape(nens,(ntime)) # Reshape to [ens x lead]

# Subset ensemble members if CESM1
if test_ens is not None and dataset_name == "CESM1":
    print("Selecting the following ensemble members" % (test_ens))
    y_class = y_class[test_ens,:]
    nens=len(test_ens)

# Get necessary dimension sizes/values
nclasses     = len(thresholds)+1
nlead        = len(leads)

# Print some messages
for l,lead in enumerate(leads):
    y_class_in = y_class[:,lead:]
    print("Lead %i" % lead)
    idx_by_class,count_by_class=am.count_samples(nsamples,y_class_in)


# -------------------
#%% Compute baselines
# -------------------
all_dicts = []
for N in tqdm(range(repeat_calc)):
    out_dict = am.compute_persistence_baseline(leads,y_class,nsamples=nsamples,percent_train=percent_train)
    all_dicts.append(out_dict)
if repeat_calc == 1:
    all_dicts = out_dict
print(out_dict.keys())

# ----------------
#%% Save Baselines
# ----------------

if (save_baseline) and (repeat_calc == 1):
    np.savez(savename,**all_dicts,allow_pickle=True)
else:
    print("Currently only supports repeat_calc=1")
    
#test = np.load(savename,allow_pickle=True)
    
    
    
# ------------------
# %% Do some quick visualization
# ------------------

fig,ax = plt.subplots(1,1)
for N in range(repeat_calc):
    ax.plot(leads,all_dicts[N]['total_acc'],alpha=0.4)
ax.set_title("Total Accuracy")
#ax.set_xticks("Lead Times")


#%%
fig,axs = plt.subplots(1,3,figsize=(16,3))

for c in range(3):
    ax = axs[c]
    
    for N in range(repeat_calc):
        ax.plot(leads,np.array(all_dicts[N]['acc_by_class'])[:,c],alpha=0.4)
    ax.set_title("Total Accuracy")
    ax.set_xlabel("Lead Times")
    ax.set_title("Class = %i" % c)
