import os
import sys
import pandas as pd
import numpy as np
sys.path.append('.')
from modules.property import FluidProperty

# class for refrigeration cycle
class RefrigerationCycle():

    # important constants and definitions
    COMP_HEAT_LOSS = 0.05               # fraction of liquid power lost as heat by the compressor

    # construct method
    def __init__(self, fluid_code) -> None:
        # when initializing the class
        # the fluid code must be informed
        self.fluid_code = fluid_code

    # compressor design
    def compressor_unit(self, sucction: list, P_discharge: float, eta: float, flowrate: float = None) -> dict:
        """
        method for designing the refrigerant compression machinery

        args: list of sucction conditions containing sucction temperature [K], pressure [MPa] and vapor fraction,
            discharge pressure [MPa], isentropic efficiency (fractional), fluid flowrate (optional) [kg/s]

        returns: a dictionary containing isentropic discharge temperature [K], real discharge temperature [K],
            isentropic specific work [kJ/kg], real specific work [kJ/kg], if the flowrate is informed, both 
            isentropic and real powers demanded are returned [kW] 
        """

        # property class object instance
        fp = FluidProperty()

        # output dictionary
        output = {}

        # extract conditions of sucction
        Ts = sucction[0]
        Ps = sucction[1]
        xs = sucction[2]

        # calculate thermodynamic properties (enthalpy and entropy) of sucction stream
        prop_s = fp.calculate_thermo_prop(Ts, Ps, self.fluid_code, xs)
        hs = prop_s[0]
        ss = prop_s[1]

        # calculate discharge conditions using isentropic assumption
        prop_d_is = fp.P_s_h_prop_calc(P_discharge, 's', ss, self.fluid_code)
        Td_is = prop_d_is[0]
        hd_is = prop_d_is[1]
        sd_is = prop_d_is[2]

        # calculate isentropic work
        w_is = hd_is - hs

        # calculate heat loss
        heat_loss = w_is*self.COMP_HEAT_LOSS

        # correct isentropic work by using isentropic efficiency
        w_real = (w_is/eta) + heat_loss

        # calculate new discharge enthalpy
        hd_real = hs + w_real

        # calculate new discharge conditions
        prop_d_real = fp.P_s_h_prop_calc(P_discharge, 'h', hd_real, self.fluid_code)
        Td_real = prop_d_real[0]
        hd_real = prop_d_real[1]
        sd_real = prop_d_real[2]

        # calculate power, if flowrate is available
        if flowrate:
            W_is = flowrate*w_is
            W_real = flowrate*w_real

        # append design information to output dictionary
        output['Sucction']['T'] = Ts
        output['Sucction']['P'] = Ps

        return output


# class for steam power cycle