#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

-----------------
Make Landice Mask
-----------------

Make Landice Mask for the CESM1-LE, Historical Period. 
Additionally Mask out sections from the Pacific Ocean.
Uses ICEFRAC and LANDFRAC variables downloaded for CESM1-LE, located
in 'datpath_raw_atm' specified in the [predict_amv_parameters] script.

Copied sections from:
    [merge_cesm1_atm.py] on 2023.06.05
    [landicemask_comparison.ipynb] on 2023.06.05

Procedure:
    (1) Creates Land and Ice Masks based on max fraction and thresholds
    (2) Combines above to make land-ice mask
    (3) Sums across ensemble members
    (4) Remove Pacific Points
    (5) Save Output as DataArray [1 x Lat x Lon]
            ex: "CESM_Data/Masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc

Created on Mon Jun  5 08:35:38 2023
@author: gliu
"""

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import xarray as xr

import time
import sys
import glob

# ----------
#%% User Edits
# ----------

# Indicate machine here
machine         = "stormtrack"

# Set custom paths
outpath         = "../../CESM_data/Masks/" # Mask Output Location
datpath         = None # Can manually set here, or will use path in predict_amv_params

# Indicate Mask Settings
vnames          = ("LANDFRAC","ICEFRAC")   # Variables
mthres          = (0.30,0.05)              # Mask out if grid ever exceeds this value
mask_sep        = True                     # Save separate masks (in addition to combined)
save_max        = True                     # Output max concentration for debugging

# General Information
nens            = 42    # Number of ensemble members to process
mconfig         = "HTR" # Scenario, where htr=Historical, rcp85=RCP 8.5 

# Other Toggles
debug           = True  # Set to True to see debugging plots

# ----------
#%% Import Packages + Paths based on machine
# ----------

# Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams

# Get paths based on machine
machine_paths = pparams.machine_paths[machine]

# Import custom modules
sys.path.append(machine_paths['amv_path'])
from amv import loaders,proc,viz

# Get data path for raw CESM1 output
if datpath is None:
    datpath = machine_paths['datpath_raw_atm']

# Get experiment bounding box for preprocessing
bbox_crop    = pparams.bbox_crop

# ----------
#%% Set some additional settings based on user input
# ----------

# Set Ensemble Member Names and Restrict to the common period
if mconfig == "RCP85":
    mnum    = np.concatenate([np.arange(1,36),np.arange(101,106)])
    ntime = 1140
elif mconfig == "HTR":
    mnum    = np.concatenate([np.arange(1,36),np.arange(101,108)])
    ntime = 1032

# ---------------------------------
#%% Load the land and ice fractions
# ---------------------------------
nvars    = len(vnames)
maskvars = []
for v in range(nvars):
    vname = vnames[v]
    outvar,times,lat,lon = loaders.load_atmvar(vname,mnum,mconfig,datpath)
    maskvars.append(outvar)

# ---------------------------------
#%% Calculate the max ice/land concentration, and make the masks
# ---------------------------------

maskvars_max = []
masks        = []
for v in range(nvars):
    
    # Calculate Max concentration
    max_frac  = maskvars[v].max(1) # Take max along time axis --> [Ens x Lat x Lon]
    if save_max:
        savename    = "%sCESM1LE_%s_%s_max.nc" % (outpath,mconfig,vnames[v])
        proc.numpy_to_da(max_frac,np.arange(1,43),lat,lon,vnames[v],savenetcdf=savename)
        print(savename)
    maskvars_max.append(max_frac)
    
    # Make the mask
    mask               = np.ones(max_frac.shape) * np.nan
    ocean_points       = max_frac < mthres[v] # 1=Ok, 0= Mask Out
    mask[ocean_points] = 1 # 1 = Ocean, NaN = Land or Ice # [Ens x Lat x Lon]
    masks.append(mask)

# -----------------
# %% Save the masks
# -----------------

# Combine land and ice mask
limask = masks[0] * masks[1]

# Save for all ensemble members [Ens x Lat x Lon]
savename    = "%sCESM1LE_%s_limask_allens.nc" % (outpath,mconfig)
da          = proc.numpy_to_da(limask,np.arange(1,43),lat,lon,"MASK",savenetcdf=savename)
print(savename)

# Save separate masks for land and ice [Ens x Lat x Lon]
if mask_sep:
    for v in range(nvars):
        savename    = "%sCESM1LE_%s_%s_mask_allens.nc" % (outpath,mconfig,vnames[v])
        da          = proc.numpy_to_da(masks[v],np.arange(1,43),lat,lon,"MASK",savenetcdf=savename)
        print(savename)

if debug: # Check ensemble sum,  land-ice mask
    plt.pcolormesh(limask.prod(0)),plt.colorbar(),plt.show()

# -----------------
#%% Manually Make Pacific Ocean Mask for points SE of Mesoamerica
# -----------------
# Taken from landicemask_comparison.ipynb

# Load the data from above
savename    = "%sCESM1LE_%s_limask_allens.nc" % (outpath,mconfig)
ds          = xr.open_dataset(savename)
limask      = ds.MASK.values
lon         = ds.lon.values
lat         = ds.lat.values


# -----------------
# Do some preprocessing
# -----------------
# Get the ensemble-sum mask
limask_enssum = limask.prod(0) # [Lat x Lon]

# Flip to lon180
lon180,limask_180 = proc.lon360to180(lon,limask_enssum.T[...,None],)
limask_180 = limask_180.squeeze().T # [Lat x Lon]
if debug:
    plt.pcolormesh(limask_180),plt.colorbar(),plt.show()

# ------------------------------------------------------
# Draw a line and try to manually identify cutoff points
# -------------------------------------------------------
ptstart = [-100,20]
ptend   = [-70,8]
if debug:
    pts     = np.vstack([ptstart,ptend])
    print(pts)
    bboxplot = [-100,-60,-20,40] # Simulation Box
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax=ax,bbox=bboxplot)
    pcm = ax.pcolormesh(lon180,lat,limask_180)
    ax.plot(pts[:,0],pts[:,1],color="y",marker="x") 
    plt.show()


# ---------------
# %% Fix the mask, check output
# ---------------

pmfix = proc.linear_crop(limask_180.T,lat,lon180,ptstart,ptend,belowline=True,along_x=True,debug=debug)
pmfix = pmfix.T # [Lat x Lon180]
if debug:
    plt.close()
    plt.pcolormesh(pmfix),plt.colorbar(),plt.show()



# Restrict to the study site
maskreg,lonr,latr = proc.sel_region(pmfix.T[...,None],lon180,lat,bbox_crop,)
maskreg = maskreg.squeeze().T # [Lat x Lon]
if debug:
    plt.pcolormesh(lonr,latr,maskreg),plt.colorbar(),plt.show()

# ----------------
#%% Save mask that will be used for the project
# ----------------

locfn,loctitle = proc.make_locstring_bbox(bbox_crop)

savename    = "%sCESM1LE_%s_limask_pacificmask_enssum_%s.nc" % (outpath,mconfig,locfn)
da          = proc.numpy_to_da(maskreg[None,...],[1,],latr,lonr,"MASK",savenetcdf=savename)
print(savename)


#%%


