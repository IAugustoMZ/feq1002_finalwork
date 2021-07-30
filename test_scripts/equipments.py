## SCRIPT TO TEST PROPERTY PREDICTION METHODS

# imports
import os
import sys
import warnings
import pandas as pd
import numpy as np
sys.path.append('./')
from modules import property
from modules import equipments

################################### IMPORTTANT DEFINITIONS ################################
warnings.filterwarnings('ignore')
RANDOM_SEED = 1002
fp = property.FluidProperty()
design = equipments.RefrigerationCycle(fluid_code = 2, Q_evap = 1500)

###########################################################################################

# simulating compressor design
T_suc = 16.85 + 273.15
P_suc = fp.vapor_pressure(2, T_suc)

T = 26.85 + 273.15
P_disc = fp.vapor_pressure(2, T)

comp = design.compressor_unit([T_suc, P_suc, 1], P_disc, 0.8, 100)
print(comp)

# simulating condenser design
inlet = [comp['Real_discharge']['T'], comp['Real_discharge']['P'], comp['Real_discharge']['x']]
outlet = [T, P_disc, 0]

condenser = design.condenser(inlet, outlet, 100)
print(condenser)