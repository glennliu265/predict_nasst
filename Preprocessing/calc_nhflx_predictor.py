#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Net Heat Flux (Qnet) from the flux components procesed with prep_data_byvariable

Created on Tue Jun  6 23:07:02 2023

@author: gliu

"""
import numpy as np
import xarray as xr
import sys
import time

import matplotlib.pyplot as plt
# ------------
#%% User Edits
# ------------

machine     = "stormtrack"
detrend     = False # Detrending is currently not applied
regrid      = None # Set to desired resolution. Set None for no regridding.

datpath     = "../../CESM_data/Predictors/" # Path to SST data processed by prep_data_byvariable.py

vnames      = ["FSNS","FLNS","LHFLX","SHFLX"]
varname_new = "NHFLX" 

debug       = True

# -----------------------------------------------------------------------------
#%% Import Packages + Paths based on machine
# -----------------------------------------------------------------------------

# Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams
import amv_dataloader as dl

# Get paths based on machine
machine_paths = pparams.machine_paths[machine]

# Import custom modules
sys.path.append(machine_paths['amv_path'])
from amv import loaders,proc


# Get experiment bounding box for preprocessing
bbox_crop   = pparams.bbox_crop
bbox_SP     = pparams.bbox_SP#[-60,-15,40,65]
bbox_ST     = pparams.bbox_ST#[-80,-10,20,40]
bbox_TR     = pparams.bbox_TR#[-75,-15,10,20]
bbox_NA     = pparams.bbox_NA#[-80,0 ,0,65]
regions     = pparams.regions
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,) # Bounding Boxes
   # regionlong  = pparams.regionlong
    #regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic","Subtropical (East)","Subtropical (West)",)


# --------------------------------
# Load the DataArray
# --------------------------------
nvars  = len(vnames)
ds_all = []
for v in range(nvars):
    ncname = "%sCESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (datpath,vnames[v],detrend,regrid)
    ds = xr.open_dataset(ncname)[vnames[v]][:,:,:,:] # [Ensemble x Year x Lat x Lon]
    if vnames[v] == "FSNS":
        ds_out = ds.values * -1 # Positive upwards
    else:
        ds_out = ds.values
    ds_all.append(ds_out)

# -------------------------------
# Sum to get NHFLX
# -------------------------------
ds_all   = np.array(ds_all) # [Flux x Ens x Year x Lat x Lon]
nhflx    = ds_all.sum(0) # [Ens x Year x Lat x Lon]
lon      = ds.lon.values
lat      = ds.lat.values
year     = ds.year.values
ensemble = ds.ensemble.values

if debug:
    plt.pcolormesh(lon,lat,nhflx[2,44,:,:]),plt.colorbar(),plt.show()

# -------------------------------
#%% Replace into dataarray and save
# -------------------------------

coords   ={"ensemble":ensemble,
           "year":year,
           "lat":lat,
           "lon":lon}
ds_nhflx =  xr.DataArray(nhflx,
            dims=coords,
            coords=coords,
            name = varname_new,
            )
savename = "%sCESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (datpath,varname_new,detrend,regrid)
st = time.time()
ds_nhflx.to_netcdf(savename,
         encoding={varname_new: {'zlib': True}})
print("Saving netCDF to %s in %.2fs"% (savename,time.time()-st))
    