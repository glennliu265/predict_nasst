#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Copied for Regrid Ocean Variable (HMXL)

Regrids CESM1-LENS ocean output from tripolar to cartesian lat/lon grid.

Saves output to                 : CESM_data/CESM1_Ocean_Regridded/[varname]/
    Filename structure is (ex.) : SSS_HTR_bilinear_regridded_ens01.nc

Uses xesmf environment. 

Created on Mon Jun 5 13:23 2023 
@author: gliu
"""

import time
import numpy as np
import xarray as xr
from tqdm import tqdm
import xesmf as xe
import sys

# -------------
#%% User Edits
# -------------

# Indicate Machine
machine       = "stormtrack"

# Indicate the Variable and Scenario
vnames        = ["HMXL",]#["BSF","HMXL"]
mconfig       = "HTR" # 'HTR' or 'RCP85'

# Indicate Regridding Options
method        = "bilinear"  # regridding method
reference_var = "LANDFRAC"  # Reference variable from CAM output

# Indicate paths
outpath       = "../../CESM_data/CESM1_Ocean_Regridded/"
datpath       = "/stormtrack/data4/glliu/01_Data/CESM1_LE/"

limit_ens     = [40,41] # ensemble INDEX

# ----------
#%% Import Packages + Paths based on machine
# ----------
# Copied from make_landice_mask.py

# Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams

# Get paths based on machine
machine_paths = pparams.machine_paths[machine]

# Import custom modules
sys.path.append(machine_paths['amv_path'])
from amv import loaders,proc

# Get data path for raw CESM1 output
if datpath is None:
    datpath     = machine_paths['datpath_raw_ocn']
datpath_atm = machine_paths['datpath_raw_atm']
    

# Get experiment bounding box for preprocessing
bbox_crop    = pparams.bbox_crop

# ----------
#%% Set some additional settings based on user input
# ----------
st = time.time()

# Set Ensemble Member Names and Restrict to the common period
if mconfig == "RCP85":
    mnum    = np.concatenate([np.arange(1,36),np.arange(101,106)])
    ntime = 1140
elif mconfig == "HTR":
    mnum    = np.concatenate([np.arange(1,36),np.arange(101,108)])
    ntime = 1032
    
if limit_ens is not None:
    nens = len(limit_ens)
    loop_ens = limit_ens
else:
    nens = len(mnum)
    loop_ens = np.arange(nens)
    
# ---------
#%% Set some functions
# ---------

# Define preprocessing variable
def preprocess(ds,keepvars=None):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    if keepvars is not None:
        ds = proc.ds_dropvars(ds,keepvars=keepvars)
    
    # # Correct first month (Note this isn't working)
    ds = proc.fix_febstart(ds)
    return ds

# ---------------------------------
#%% Load Lat/Lon Universal Variable
# ---------------------------------

# Set up reference lat/lon
ds_atm = loaders.load_htr(reference_var,1,datpath=datpath_atm)
ds_atm = ds_atm.isel(time=0)
lon    = ds_atm.lon.values
lat    = ds_atm.lat.values
ds_out = xr.Dataset({'lat':lat,'lon':lon})

# ---------------------------------
#%% Load the land and ice fractions
# ---------------------------------
nvars    = len(vnames)

for v in range(nvars):
    
    varname = vnames[v]
    
    # Adjust data path on stormtrack machine (hack fix...)
    if machine == "stormtrack": 
        if datpath is None:
            if varname == "SSS":
                datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/processed/ocn/proc/tseries/monthly/"
            else:
                datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/" 
        
    # Set preprocessing function (don't need this since I load to dataarray)
    #preprocess_var = lambda x : preprocess(x,keepvars=["TLONG","TLAT","time",varname])
    
    # Loop for ensemble members
    for e in tqdm(limit_ens):
        
        # Load the dataarray
        N = mnum[e]
        #print(N)
        ds = loaders.load_htr(varname,N,datpath=datpath,atm=False)
        ds = preprocess(ds.load())
        
        # Rename Latitude/Longitude to prepare for regridding
        ds = ds.rename({"TLONG": "lon", "TLAT": "lat"})
        ds = ds.to_dataset()
        
        # Initialize Regridder
        regridder = xe.Regridder(ds,ds_out,method,periodic=True)
        
        # Regrid
        daproc = regridder(ds) # Need to input dataarray
        
        # Save
        savename = "%s%s_%s_%s_regridded_ens%02i.nc" % (outpath,varname,
                                                  mconfig,method,e+1)
        daproc.to_netcdf(savename,
                         encoding={varname: {'zlib': True}})
print("Regridded all ocean variables in %.2fs" % (time.time()-st))







