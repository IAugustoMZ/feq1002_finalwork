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
    T_cond_in = 20 + 273.15             # inlet cold utility temperature at condenser [K]
    T_cond_out = 60 + 273.15            # outlet cold utility temperature at condenser [K]
    P_cond = 100/1000                   # cold utility pressure [MPa]
    cond_fluid_code = 3                 # cold utility fluid code
    cond_x = 0                          # cold utility vapor fraction

    # construct method
    def __init__(self, fluid_code, Q_evap) -> None:
        """
        class to design the refrigeration cycle

        args: refrigerant fluid code, total refrigeration load (RT)
        """
        self.fluid_code = fluid_code
        self.Q_evap = Q_evap

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
        output = {
            'Sucction':{
                'T': 0,
                'P': 0,
                'h': 0,
                's': 0,
                'x': 0
            },
            'Isoentropic_discharge': {
                'T': 0,
                'P': 0,
                'h': 0,
                's': 0,
                'x': 0
            },
            'Real_discharge': {
                'T': 0,
                'P': 0,
                'h': 0,
                's': 0,
                'x': 0
            },
            'Isoentropic_work': 0,
            'Real_work': 0,
            'Iso_Eff': 0,
            'Isoentropic_power': None,
            'Real_power': None,
            'Heat_loss': 0
        }

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
            output['Isoentropic_power'] = flowrate*w_is/1000
            output['Real_power'] = flowrate*w_real/1000

        # append design information to output dictionary
        # sucction information
        output['Sucction']['T'] = Ts
        output['Sucction']['P'] = Ps
        output['Sucction']['h'] = hs/1000
        output['Sucction']['s'] = ss/1000
        output['Sucction']['x'] = xs

        # isoentropic discharge
        output['Isoentropic_discharge']['T'] = Td_is
        output['Isoentropic_discharge']['P'] = P_discharge
        output['Isoentropic_discharge']['h'] = hd_is/1000
        output['Isoentropic_discharge']['s'] = sd_is/1000
        output['Isoentropic_discharge']['x'] = prop_d_is[3]

        # real discharge
        output['Real_discharge']['T'] = Td_real
        output['Real_discharge']['P'] = P_discharge
        output['Real_discharge']['h'] = hd_real/1000
        output['Real_discharge']['s'] = sd_real/1000
        output['Real_discharge']['x'] = prop_d_real[3]

        # other information
        output['Iso_Eff'] = eta
        output['Isoentropic_work'] = w_is/1000
        output['Real_work'] = w_real/1000
        output['Heat_loss'] = heat_loss/1000


        return output

    # condenser design
    def condenser(self, inlet: list, outlet: list, flowrate: float) -> dict:
        """
        performs the design of the condenser unit

        args: list containing inlet conditions (temperature [K], pressure [MPa], vapor_fraction),
            list containing outlet conditions (temperature [K], pressure [MPa], vapor_fraction)
            refrigerant flowrate [kg/s]

        returns: dictionary containing the design information
        """
        # property class object instance
        fp = FluidProperty()

        # create output dictionary
        output = {
            'Inlet':{
                'T': 0,
                'P': 0,
                'h': 0,
                's': 0,
                'x': 0,
            },
            'Outlet':{
                'T': 0,
                'P': 0,
                'h': 0,
                's': 0,
                'x': 0,
            },
            'Cold_utility_load': 0,
            'Spec_heat_load': 0,
            'Total_heat_load': 0 
        }

        # extract inlet information
        Tin = inlet[0]
        Pin = inlet[1]
        xin = inlet[2]

        # extract outlet information
        Tout = outlet[0]
        Pout = outlet[1]
        xout = outlet[2]

        # calculate inlet thermodynamic properties
        prop_in = fp.calculate_thermo_prop(Tin, Pin, self.fluid_code, xin)
        hin = prop_in[0]
        sin = prop_in[1]

        # calculate outlet thermodynamic properties
        prop_out = fp.calculate_thermo_prop(Tout, Pout, self.fluid_code, xout)
        hout = prop_out[0]
        sout = prop_out[1]

        # calculate specific heat load
        q = hin - hout

        # calculate total heat load
        Q = flowrate*q

        # calculate inlet and outlet enthalpies of cold utility
        h_cold_in = fp.calculate_thermo_prop(self.T_cond_in, self.P_cond, self.cond_fluid_code, self.cond_x)[0]
        h_cold_out = fp.calculate_thermo_prop(self.T_cond_out, self.P_cond, self.cond_fluid_code, self.cond_x)[0]

        # calculate cold utility load
        m_cold = Q/(h_cold_out - h_cold_in)

        # append output information to dictionary
        # inlet stream
        output['Inlet']['T'] = Tin
        output['Inlet']['P'] = Pin
        output['Inlet']['h'] = hin/1000
        output['Inlet']['s'] = sin/1000
        output['Inlet']['x'] = xin

        # outlet stream
        output['Outlet']['T'] = Tout
        output['Outlet']['P'] = Pout
        output['Outlet']['h'] = hout/1000
        output['Outlet']['s'] = sout/1000
        output['Outlet']['x'] = xout

        # design information
        output['Cold_utility_load'] = m_cold
        output['Spec_heat_load'] = q/1000
        output['Total_heat_load'] = Q/1000

        return output





# class for steam power cycle