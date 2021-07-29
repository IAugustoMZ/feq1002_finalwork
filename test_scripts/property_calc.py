## SCRIPT TO TEST PROPERTY PREDICTION METHODS

# imports
import os
import sys
import warnings
import pandas as pd
import numpy as np
sys.path.append('./')
from modules import property

################################### IMPORTTANT DEFINITIONS ################################
warnings.filterwarnings('ignore')
RANDOM_SEED = 1002
fp = property.FluidProperty()

# testing the calculation of enthalpy / entropy
prop = fp.calculate_thermo_prop(300, 0.1, 1, 0)

# testing calculation of vapor pressure
T = 16.85
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