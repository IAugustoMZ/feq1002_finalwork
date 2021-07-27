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
FLUID_CODE = 1 # methane
fp = property.FluidProperty()

prop = fp.calculate_thermo_prop(111.51, 0.1, FLUID_CODE, 1)
