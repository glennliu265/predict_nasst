#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare data by variable (for ML Prediction)

- Crops the data to a given region/time period
- Merges by ensemble and computes ensemble average
- Deseasons, Detrends, Normalizes, and optionally Regrids

  Works with regridded ocean data preprocessed by [regrid_ocean_variables.ppy]
  or with raw CESM1-LE atmospheric variables. Note that detrending step
  has been moved to [prep_detrended_predictor] script.
  
Output Files: (located at "../../CESM_data/Predictors"). Example Names:
    Ens Avg            : CESM1LE_SSS_FULL_HTR_bilinear_ensavg_1920to2005.nc
    Concatenated Data  : CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc
    Predictor          : CESM1LE_SSS_NAtl_19200101_20051201_bilinear_detrend0_regridNone.nc
    
For the given large ensemble dataset/variable, perform the following:
    
    -------
    Part 1-- : Merging and subsetting data
    -------
    1. Crop the Data in Time (1920 - 2005)
    2. Crop to Region (*****NOTE: assumes degrees West = neg!)
    3. Concatenate to ensemble member
    4. Intermediate Save Option
    
    -------
    Part 2-- : Deseason, Detrend, Normalize, Regrid
    -------
    1. Calculate Monthly Anomalies + Annual Average
    2. Perform regridding (if option is set)
    3. Output in array ['ensemble','year','lat','lon']
    
Copies sections from:
    - prep_mld_PIC.py (stochmod repo)
    - prepare_training_validation_data.py
    
Created on Thu Oct 20 16:35:33 2022

@author: gliu

"""

import numpy as np
import xarray as xr
import xesmf as xe
import glob
import time
from tqdm import tqdm
import sys

# -----------------------------------------------------------------------------
#%% User Edits
# -----------------------------------------------------------------------------

stall         = time.time()
machine       = "stormtrack"

# Dataset Information
varnames      = ["TS","SSH","SSS","PSL","FSNS","FLNS","FSNS","LHFLX","SHFLX","TAUX","TAUY","BSF","HMXL"]

mconfig       = "FULL_HTR"
method        = "bilinear" # regridding method for POP ocean data

# Processing Options
regrid        = None  # Set to desired resolution. Set None for no regridding.
regrid_step   = True  # Set to true if regrid indicates the stepsize rather than total dimension size..
save_concat   = False  # Set to true to save the concatenated dataset (!! before annual anomaly calculation)
save_ensavg   = False # Set to true to save ensemble average (!! before annual anomaly calculation)
load_concat   = False # Set to true to load concatenated data

# Cropping Options
bbox          = None # Crop Selection, defaults to value indicated in predict_amv_params
ystart        = 1920 # Start year
yend          = 2005 # End year

# Data Path
datpath_manual = None # Manually set datpath
outpath        = "../../CESM_data/Predictors/"

# Other Toggles
debug         = True # Set to True for debugging flat

# -----------------------------------------------------------------------------
#%% Import Packages + Paths based on machine
# -----------------------------------------------------------------------------

# Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams

# Get paths based on machine
machine_paths = pparams.machine_paths[machine]

# Import custom modules
sys.path.append(machine_paths['amv_path'])
from amv import loaders,proc

# Get experiment bounding box for preprocessing
if bbox is None:
    bbox  = pparams.bbox_crop
nvars = len(varnames)

#%% Start variable loop
for v in range(nvars):
    
    varname = varnames[v]
    
    if varname == "TS":
        rename_flag   = True
        varname_new   = "SST" # New variable name,
    elif varname == "PSL":
        rename_flag   = True
        varname_new   = "SLP"
    else:
        rename_flag       = False
    
    # Get data path for raw CESM1 output
    atm = True # Assume variables are raw unless otherwise specified
    if datpath_manual is None:
        if varname in pparams.vars_dict.keys():
            vdict = pparams.vars_dict[varname]
            if vdict['datpath'] is None:
                datpath = machine_paths['datpath_raw_atm'] # Process from Raw CESM1 Atmopsheric Data
            else:
                atm     = False
                datpath = vdict['datpath'] # Used datpath specified in variable dictionary
                # This is usually "../../CESM_data/CESM1_Ocean_Regridded/" for ocn variables
                # And another unique path for Net Heat Flux (TBD)
        else: # Assume it is an atmospheric variable
            datpath = machine_paths['datpath_raw_atm'] # Process from Raw CESM1 Atmopsheric Data
    else:
        datpath = datpath_manual
        
    # -----------------------------------------------------------------------------
    #%% Get list of netcdfs
    # -----------------------------------------------------------------------------
    
    if atm: # Load from raw CESM1-LE files
    
        if "HTR" in mconfig:
            scenario_str = "b.e11.B20TRC5CNBDRD*"
        elif "RCP85" in mconfig:
            scenario_str = "b.e11.BRCP85C5CNBDRD"
            
        nclist = glob.glob("%s%s/%s%s*.nc" % (datpath,varname,scenario_str,varname))
        nclist = [nc for nc in nclist if "OIC" not in nc]
        
    else:
        nclist = glob.glob("%s*%s*.nc" % (datpath,varname))
    nclist.sort()
    nens     = len(nclist)
    print("Found %i files!" % (nens))
    if debug:
        print(*nclist,sep="\n")
    
    # -----------------------------------------------------------------------------
    #%% Preprocessing: Concatenation + Cropping (Part 1)
    # -----------------------------------------------------------------------------
    
    """
    -----------------------
    The Preprocessing Steps
    -----------------------
        Part 1-- : Merging and subsetting data
             Takes 5m24s (_) for ocn (atm) dataset (versus 30m without loading ds)
        1. Crop the Data in Time (1920 - 2005)
        2. Crop to Region (*****NOTE: assumes degrees West = neg!)
        3. Concatenate to ensemble member
        4. Intermediate Save Option
    """
    
    # 
    for e in tqdm(range(nens)): # Process by ens. member
        
        # Load into dataset
        ds = xr.open_dataset(nclist[e])
        
        # 1. Correct time if needed, then crop to range ******
        ds = proc.fix_febstart(ds)
        ds = ds.sel(time=slice("%s-01-01"%(ystart),"%s-12-31"%(yend)))
        ds = ds[varname].load()
        
        # 2. Flip longitude and crop to region ******
        if np.any(ds.lon.values > 180): # F
            ds = proc.lon360to180_xr(ds)
            print("Flipping Longitude (includes degrees west)")
        dsreg = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
        
        # 3. Concatenate to ensemble ******
        if e == 0:
            ds_all = dsreg.copy()
        else:
            ds_all = xr.concat([ds_all,dsreg],dim="ensemble",join='override')
    
    
    # Rename variable if set
    if rename_flag:
        ds_all = ds_all.rename(varname_new)
        varname=varname_new
    
    
    # Set encoding dictionary
    encoding_dict = {varname : {'zlib': True}} 
    
    # Get rid of additional dimensions
    if "z_t" in ds_all.dims:
        ds_all = ds_all.squeeze()
        ds_all = ds_all.drop_vars('z_t')
        ds_all  = ds_all
    
    # Transpose to [ens x time x lat x lon]
    ds_all  = ds_all.transpose('ensemble','time','lat','lon')
    
    # Compute and save the ensemble average
    if save_ensavg:
        ensavg = ds_all.mean('ensemble')
        outname = "%sCESM1LE_%s_%s_%s_ensavg_%sto%s.nc" % (outpath,varname,mconfig,method,ystart,yend)
        ensavg.to_netcdf(outname,encoding=encoding_dict)
    
    # Save Dataset
    outname_concat = "%sCESM1LE_%s_NAtl_%s0101_%s0101_%s.nc" % (outpath,varname,ystart,yend,method)
    if save_concat:
        ds_all.to_netcdf(outname_concat,encoding=encoding_dict)
        print("Merged data in %.2fs" % (time.time()-stall))
    
    
    # -------
    # %% Part 2 Processing (Deseason Normalization, Regridding)
    # -------
    
    """
    -------
    Part 2-- : Deseason, Detrend, Normalize, Regrid
    -------
    Based on procedure in prepare_training_validation_data.py
    ** does not apply land mask!
        1. Calculate Monthly Anomalies + Annual Average
        2. Perform regridding (if option is set)
        3. Output in array ['ensemble','year','lat','lon']
    """
    
    if load_concat: # Load data if option is set
        ds_all = xr.open_dataset(outname_concat)[varname].load()
    
    # --------------------------------
    # Deseason and take annual average
    # --------------------------------
    st = time.time() # 38.21 sec
    ds_all_anom = (ds_all.groupby('time.month') - ds_all.groupby('time.month').mean('time')).groupby('time.year').mean('time')
    print("Deseasoned in %.2fs!" % (time.time()-st))
    
    # ------------------------
    # Regrid, if option is set 
    # ------------------------
    if regrid is not None:
        print("Data will be regridded to %i degree resolution." % regrid)
        # Prepare Latitude/Longitude
        lat = ds_all_anom.lat
        lon = ds_all_anom.lon
        if regrid_step:
            lat_out = np.arange(lat[0],lat[-1]+regrid,regrid)
            lon_out = np.arange(lon[0],lon[-1]+regrid,regrid)
        else:
            lat_out = np.linspace(lat[0],lat[-1],regrid)
            lon_out = np.linspace(lon[0],lon[-1],regrid)
        
        # Make Regridder
        ds_out    = xr.Dataset({'lat': (['lat'], lat_out), 'lon': (['lon'], lon_out) })
        regridder = xe.Regridder(ds_all_anom, ds_out, 'nearest_s2d')
    
        # Regrid
        ds_out = regridder( ds_all_anom.transpose('ensemble','year','lat','lon') )
    else:
        print("Data will not be regridded.")
        ds_out = ds_all_anom.transpose('ensemble','year','lat','lon')
        
    # ---------------
    # Save the output
    # ---------------
    st = time.time() #387 sec
    if varname != varname.upper():
        print("Capitalizing variable name.")
        ds_out = ds_out.rename({varname:varname.upper()})
        varname = varname.upper()
    encoding_dict = {varname : {'zlib': True}} 
    outname       = "%sCESM1LE_%s_NAtl_%s0101_%s1201_%s_detrend0_regrid%s.nc" % (outpath,varname,ystart,yend,method,regrid)
    ds_out.to_netcdf(outname,encoding=encoding_dict)
    print("Saved output tp %s in %.2fs!" % (outname,time.time()-st))
    
    print("Completed processing %s in %.2fs!" % (varname,time.time()-stall))
