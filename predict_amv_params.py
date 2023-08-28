#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict AMV, Parameter File

Sections

    Set Project Paths                       : Path to figures and processed data
    Module and (Raw) Data Paths             : Machine paths to raw data, modules
    ML Model Parameters/Dictionary          : NN Architecture Parameter Dictionaries
    Target and Plotting Regions             : Regional Names and bounding boxes
    Predictor Information                   : Predictor/variable names, colors, plotting info, etc
    Class Names and Information             : Names and colors of the classes
    Plotting Parameters                     : Plotting parameters/ticks
    CESM1 Variable Dictionary/Profiles      : Same as [Predictor Information] but as a dictionary
    LENS Parameters                         : CMIP5 and some CMIP6 information
    CMIP6 Dictionary                        : CMIP6 variable and dictionary information
    Reanalysis Dictionary                   : Information on reanalysis datasets

Created on Mon Jan 16 13:32:37 2023
@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from torch import nn

# -----------------------------------------------------------------------
#%% Set Project Paths
# -----------------------------------------------------------------------
# Path to data (raw model output, observations, NN data). Assumed to be in same directory as predict_nasst repo.
datpath       = "../../CESM_data/"

# Figure output path. User selected.
figpath       = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230720/"

# -----------------------------------------------------------------------
#%% Module and (Raw) Data Paths
# -----------------------------------------------------------------------
# Add Relative paths here

# Added info from scm. First is the template
mdict0 = {
    "machine"           : None,     # Name of the machine
    "amv_path"          : 0,        # Path to amv module (with proc,viz)
    "datpath_raw_atm"   : 0,        # Path to CESM1-LENS Atmospheric Variables
    "datpath_raw_ocn"   : 0,        # Path to CESM1-LENS Ocean Variables
    "cesm2path"         : 0,        # Path to CESM2 Data
    "lenspath"          : 0,        # Large Ensemble Data (CMIP5)
    "lrp_path"          : 0,        # Path to Pytorch-LRP module
    }

# Default, uses relative paths and assumes a particular file structure
# where the additional modules are in General_Utilities and data in Raw_Data
mdict0 = {
    "machine"           : "default", # Name of the machine
    "amv_path"          : "../../General_Utilities/amv/",# Path to amv module (with proc,viz)
    "datpath_raw_atm"   : "../../Raw_Data/CESM1/atm/", # Path to CESM1-LENS Atmospheric Variables
    "datpath_raw_ocn"   : "../../Raw_Data/CESM1/ocn/", # Path to CESM1-LENS Ocean Variables
    "cesm2path"         : "../../Raw_Data/CESM2/", # Path to CESM2 Data
    "lenspath"          : "../../Raw_Data/LENS_CMIP5/", # Large Ensemble Data (CMIP5)
    "lrp_path"          : "../../General_Utilities/Pytorch-LRP-master/",
    }

# Stormtrack Server
mdict1 = {
    "machine"           : "stormtrack", # Name of the machine
    "amv_path"          : "/home/glliu/00_Scripts/01_Projects/00_Commons/",# Path to amv module (with proc,viz)
    "datpath_raw_atm"   : "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/", # Path to CESM1-LENS Atmospheric Variables
    "datpath_raw_ocn"   : "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/", # Path to CESM1-LENS Ocean Variables
    "cesm2path"         : 0, # Path to CESM2 Data
    "lenspath"          : "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/", # Large Ensemble Data (CMIP5)
    "lrp_path"          : "../../tools/Pytorch-LRP-master/",
    }

# Astraeus Local
mdict2 = {
    "machine"           : "Astraeus",
    "amv_path"          : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/",
    "datpath_raw_atm"   : 0, # Path to CESM1-LENS Atmospheric Variables
    "datpath_raw_ocn"   : 0, # Path to CESM1-LENS Ocean Variables
    "cesm2path"         : "/Users/gliu/Globus_File_Transfer/CESM2_LE/1x1/",
    "lenspath"          : 0, # Large Ensemble Data (CMIP5)
    "lrp_path"          : "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/ml_demo/Pytorch-LRP-master/",
    }

# Make the [machine_paths] dictionary
machine_path_dicts  = (mdict0,mdict1,mdict2,)
machine_names       = [d["machine"] for d in machine_path_dicts]
machine_paths       = dict(zip(machine_names,machine_path_dicts))

# -----------------------------------------------------------------------
#%% ML Model Parameters/Dictionary
# -----------------------------------------------------------------------

"""
Descriptions taken from NN training script
cnndropout : Set to 1 to test simple CNN with dropout layer
"""

# FNN2
FNN2_dict={
    "nlayers"     : 2,
    "nunits"      : [20,20],
    "activations" : [nn.ReLU(),nn.ReLU()],
    "dropout"     : 0.5}

# FNN4_120
FNN120_dict={
    "nlayers"     : 4,
    "nunits"      : [120,120,120,120],
    "activations" : [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
    "dropout"     : 0.5}

# FNN4_128
FNN128_dict={
    "nlayers"     : 4,
    "nunits"      : [128,128,128,128],
    "activations" : [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
    "dropout"     : 0.5}

# simplecnn
simplecnn_dict={
    "cnndropout"     : True,
    "num_classes"    : 3, # 3 AMV States
    "num_inchannels" : 1, # Single Predictor
    }

# Assemble the dictionaries ...
modelnames = ("FNN2"   , "FNN4_120"   , "FNN4_128"   , "simplecnn", "CNN2_LRP")
indicts    = (FNN2_dict, FNN120_dict  , FNN128_dict  , simplecnn_dict, simplecnn_dict)
nn_param_dict = dict(zip(modelnames,indicts))

# -----------------------------------------------------------------------
#%% Target and Plotting Regions
# -----------------------------------------------------------------------

# Regions (Bounding Boxes and Names)
regions       = ("NAT","SPG","STG","TRO")#("NAT","SPG","STG","TRO")
rcolors       = ("k","b",'r',"orange")
bbox_SP       = [-60,-15,40,65]
bbox_ST       = [-80,-10,20,40]
bbox_TR       = [-75,-15,10,20]
bbox_NA       = [-80,0 ,0,65]
bbox_NA_new   = [-80,0,10,65]
bbox_ST_w     = [-80,-40,20,40]
bbox_ST_e     = [-40,-10,20,40]
bboxes        = (bbox_NA,bbox_SP,bbox_ST,bbox_TR,) # Bounding Boxes

# Plotting Information
proj          = ccrs.PlateCarree()
bbox          = [-80,0,0,65]
plotbbox      = [-80,0,0,62]   # Plotting extent
amvbbox       = [-80,0,0,65]   # AMV Calculation box
bbox_crop     = [-90,20,0,90]  # Preprocessing box

# -----------------------------------------------------------------------
#%% Predictor Information (Includes outdated versions...)
# -----------------------------------------------------------------------

# Variables (allpred)
allpred       = ("SST","SSS","PSL","SSH")
apcolors      = ("r","limegreen","pink","darkblue")

# Variables (all, old pre 2022.12.09)
varnames      = ("SST","SSS","PSL","SSH","BSF","HMXL",)
varcolors     = ("r","limegreen","pink","darkblue","purple","cyan")
threscolors   = ("r","gray","cornflowerblue")

# Variables (all, new since 2022.12.09, updated 2023.02.07 adding UOHC/UOSC)
varnames      = ("SST","SSS","PSL","BSF","SSH","HMXL","UOHC","UOSC","NHFLX")
varnamesplot  = ("SST","SSS","SLP","BSF","SSH","MLD","UOHC","UOSC","NHFLX")
varnames_long = ("Temperature","Salinity","Pressure","Barotropic Streamfunction",
                 "Sea Surface Height","Mixed-Layer Depth",
                 "Upper Ocean Heat Content","Upper Ocean Salt Content","Net Heat Flux")
vunits        = ("$\degree$C","psu","mb","cm","Sv","cm","$J\,m^{-2}$","$J\,m^{-2}$","$W m^{-2}$")
varcolors          = ("r","violet","gold","darkblue","dodgerblue","cyan","lightcoral","orchid","orange")
varcolors_dark     = ("r","violet","gold","darkblue","dodgerblue","cyan","lightcoral","orchid","orange")
varmarker     = ("o","d","x","v","^","*","1","2","3")

# Variables (all, 2023.07.10) -------------------------------------------------
varnames      = ("SST",
                  "SSS",
                  "SLP",
                  "SSH",
                  "NHFLX",
                  "TAUX",
                  "TAUY",
                  "TAUCURL")
varnamesplot  = varnames
varnames_long = ("Temperature",
                  "Salinity",
                  "Pressure",
                  "Sea Surface Height",
                  "Net Heat Flux",
                  "Zonal Wind Stress",
                  "Meridional Wind Stress",
                  "Wind Stress Curl")
vunits        = ("$\degree$C",
                  "psu",
                  "mb",
                  "cm",
                  "$W m^{-2}$",
                  "N m^{-2}$",
                  "N m^{-2}$",
                  "N m^{-3}$",)
varcolors          = ("r",
                      "violet",
                      "gold",
                      "dodgerblue",
                      "cyan",
                      "limegreen",
                      "darkblue",
                      "pink",
                      )
varcolors_dark     = varcolors
varmarker     = ("o",
                  "d",
                  "x",
                  "^",
                  "*",
                  "1",
                  "2",
                  "3")

# -----------------------------------------------------------------------
#%% Class Names and Information
# -----------------------------------------------------------------------

# Class Names and colors
classes         = ["NASST+","Neutral","NASST-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
classes_amv     = ["AMV+","Neutral","AMV-"]
class_colors    = ("salmon","gray","cornflowerblue")

# -----------------------------------------------------------------------
#%% Plotting Parameters
# -----------------------------------------------------------------------

# Font sizes (adjust to accuracy by predictor)
fsz_sptitle      = 22 # Overall title across all subplots
fsz_title        = 20 # Subplot Title
fsz_splbl        = 16 # Axis Subplot (numbering) labels
fsz_axlbl        = 14 # Axis Labels
fsz_ticks        = 12 # Axis Tick labels
fsz_legend       = 12 # Legend Text

# Plotting (acc by leadtime)
leadticks24      = np.arange(0,25,3)
leadticks25      = np.arange(0,26,5)

# -----------------------------------------------------------------------
#%% CESM1 Variable Dictionary/Profiles
# -----------------------------------------------------------------------

vdict0 = {
    "varname"     : 0, #Name of the variable
    "other_names" : 0, # Other Names
    "vnames_plot" : 0, # plotting name
    "longname"    : 0, # Long Name
    "realm"       : 0, # Atm or Ocn
    "units"       : 0, # Units
    "color"       : 0, # Variable Color
    "marker"      : 0, # Marker for plotting
    "linestyle"   : 0, # Line Style
    "datpath"     : 0,  # Location of variable
    }

vdict1 = {
    "varname"     : "SST"           , #Name of the variable
    "other_names" : ['sst','ts','TS'], # Other 
    "vnames_plot" : "SST", # plotting name
    "longname"    : "Sea Surface Temperature", # Long Name
    "realm"       : "atm"           , # Atm or Ocn
    "units"       : "$\degree$C"    , # Units
    "color"       : "r", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : None, # Location of variable, None for CESM1-Raw
    }

vdict15 = {
    "varname"     : "TS"           , #Name of the variable
    "other_names" : ['sst','ts','sst'], # Other 
    "vnames_plot" : "SST", # plotting name
    "longname"    : "Sea Surface Temperature", # Long Name
    "realm"       : "atm"           , # Atm or Ocn
    "units"       : "$\degree$C"    , # Units
    "color"       : "r", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : None, # Location of variable, None for CESM1-Raw
    }

vdict2 = {
    "varname"     : "SSH", #Name of the variable
    "other_names" : ["ssh"], # Other Names
    "vnames_plot" : "SSH", # plotting name
    "longname"    : "Sea Surface Height", # Long Name
    "realm"       : "ocn", # Atm or Ocn
    "units"       : "cm", # Units
    "color"       : "dodgerblue", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : "../../CESM_data/CESM1_Ocean_Regridded/",  # Location of variable
    }

vdict3 = {
    "varname"     : "SSS", #Name of the variable
    "other_names" : ["sss"], # Other Names
    "vnames_plot" : "SSS", # plotting name
    "longname"    : "Sea Surface Salinity", # Long Name
    "realm"       : "ocn", # Atm or Ocn
    "units"       : "psu", # Units
    "color"       : "limegreen", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : "../../CESM_data/CESM1_Ocean_Regridded/",  # Location of variable
    }

vdict4 = {
    "varname"     : "PSL", #Name of the variable
    "other_names" : ["psl","slp","SLP"], # Other Names
    "vnames_plot" : "SLP", # plotting name
    "longname"    : "Sea Level Pressure", # Long Name
    "realm"       : "atm", # Atm or Ocn
    "units"       : "mb", # Units
    "color"       : "gold", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : None,  # Location of variable
    }

vdict5 = {
    "varname"     : "BSF", #Name of the variable
    "other_names" : ["bsf"], # Other Names
    "vnames_plot" : "BSF", # plotting name
    "longname"    : "Barotropic Streamfunction", # Long Name
    "realm"       : "ocn", # Atm or Ocn
    "units"       : "Sv", # Units
    "color"       : "purple", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : "../../CESM_data/CESM1_Ocean_Regridded/",  # Location of variable
    }

vdict6 = {
    "varname"     : "HMXL", #Name of the variable
    "other_names" : ["HMXL"], # Other Names
    "vnames_plot" : "MLD", # plotting name
    "longname"    : "Mixed Layer Depth", # Long Name
    "realm"       : "ocn", # Atm or Ocn
    "units"       : "cm", # Units
    "color"       : "violet", # Variable Color
    "marker"      : "o", # Marker for plotting
    "linestyle"   : "solid", # Line Style
    "datpath"     : "../../CESM_data/CESM1_Ocean_Regridded/",  # Location of variable
    }

indicts_vars      = [vdict1,vdict15,vdict2,vdict3,vdict4,vdict5,vdict6]
indicts_vars_keys = [d["varname"] for d in indicts_vars]
vars_dict         = dict(zip(indicts_vars_keys,indicts_vars))

# -----------------------------------------------------------------------
#%% LENS Parameters
# -----------------------------------------------------------------------

# CMIP5
dataset_names  = ("canesm2_lens" ,"csiro_mk36_lens","gfdl_esm2m_lens","mpi_lens"  ,"CESM1")
dataset_long   = ("CCCma-CanESM2","CSIRO-MK3.6"    ,"GFDL-ESM2M"     ,"MPI-ESM-LR","NCAR-CESM1")
dataset_colors = ("r"            ,"b"              ,"magenta"        ,"gold" ,"limegreen")
dataset_starts = (1950           ,1920             ,1950             ,1920        ,1920)

# CMIP6 (Potentially delete this section as I have organized this into dictionaries later on)
"""
"CESM2"
"IPSL-CM6A-LR"
"CanESM5"
"MIROC6"
"ACCESS-ESM1-5"
"""
cmip6_varnames       = ("tos","sos","zos")
cmip6_varnames_remap = ("sst","sss","ssh") # Also the CESM2 variable names...
cmip6_varcolors      = ("r"  ,"violet","dodgerblue")
cmip6_varnames_long  = ("Temperature"  ,"Salinity","Sea Surface Height")

cmip6_names          = ("ACCESS-ESM1-5","CanESM5","IPSL-CM6A-LR","MIROC6","MPI-ESM1-2-LR","CESM2")
cmip6_markers        = ("o"            ,"d"      ,"x"           ,"v"     ,"^"            ,"*")
cmip6_colors         = ("orange"       ,"r"      ,"magenta"     ,"b"     ,"gold"         ,"limegreen")
zos_units            = ("m"            ,"m"      ,"m"           ,"cm"    ,"m")

# -----------------------------------------------------------------------
#%% CMIP6 Dictionary
# -----------------------------------------------------------------------

access_dict = {
    "dataset_name" : "ACCESS-ESM1-5",
    "mrk"          : "o",
    "col"          : "orange",
    "zos_units"    : "m",
    }

canesm_dict = {
    "dataset_name" : "CanESM5",
    "mrk"          : "d",
    "col"          : "r",
    "zos_units"    : "m",
    }

ispl_dict = {
    "dataset_name" : "IPSL-CM6A-LR",
    "mrk"          : "x",
    "col"          : "magenta",
    "zos_units"    : "m",
    }

miroc_dict = {
    "dataset_name" : "MIROC6",
    "mrk"          : "v",
    "col"          : "b",
    "zos_units"    : "cm",
    }

mpi_dict = {
    "dataset_name" : "MPI-ESM1-2-LR",
    "mrk"          : "^",
    "col"          : "gold",
    "zos_units"    : "m",
    }

cesm_dict = {
    "dataset_name" : "CESM2",
    "mrk"          : "*",
    "col"          : "limegreen",
    "zos_units"    : "cm",
    }

indicts_cmip6      = [access_dict    ,canesm_dict,ispl_dict     ,miroc_dict, mpi_dict       ,cesm_dict]
indicts_cmip6_keys = [d["dataset_name"] for d in indicts_cmip6]
cmip6_dict         = dict(zip(indicts_cmip6_keys,indicts_cmip6))

# -----------------------------------------------------------------------
#%% Reanalysis Dictionary
# -----------------------------------------------------------------------
#% Dictionary for reanalysis variable names

had_dict = {
    'dataset_name' : 'HadISST',
    'sst'          : 'sst',
    'lat'          : 'latitude',
    'lon'          : 'longitude',
    'ystart'       : 1870
    }

indicts          = (had_dict,)
reanalysis_names = [d['dataset_name'] for d in indicts]
reanalysis_dict  = dict(zip(reanalysis_names,indicts))
