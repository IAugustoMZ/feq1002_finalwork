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
design = equipments.RefrigerationCycle(fluid_code = 2)

# testing the calculation of enthalpy / entropy
prop = fp.calculate_thermo_prop(300, 0.1, 1, 0)

# testing calculation of vapor pressure
T = 26.85
T += 273.15

Psat = fp.vapor_pressure(2, T)
print(Psat*1000)

# testing px calculation
results = fp.P_x_prop_calc(Psat, 1, 2)
print(results)

# liquid water properties
T = 20
T +=273.15

results = fp.calculate_thermo_prop(T, 0.1, 3, 1)
print(results)

# testing the P_s calculation method (simulating compressor)
T_suc = 16.85 + 273.15
P_suc = fp.vapor_pressure(2, T_suc)
h_suc = fp.calculate_thermo_prop(T_suc, P_suc, 2, 1)[0]
s = fp.calculate_thermo_prop(T_suc, P_suc, 2, 1)[1]

T = 26.85 + 273.15
P_disc = fp.vapor_pressure(2, T)
cond_disc_is = fp.P_s_h_prop_calc(P_disc, 's', s, 2)
print(cond_disc_is)

h_is = cond_disc_is[1]
w_is = h_is - h_suc
print(w_is)

eta = 0.8
w_real = w_is / eta

print(w_real)
h_real = h_suc + w_real
cond_disc_real = fp.P_s_h_prop_calc(P_disc, 'h', h_real, 2)
print(cond_disc_real)


comp = design.compressor_unit([T_suc, P_suc, 1], P_disc, 0.8, 100)
print(comp)