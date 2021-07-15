import pandas as pd
import numpy as np
import os
import json
from utils_functions import utils

# calculate fluid property
class FluidProperty():

    # numerical convergence tolerance
    eps = 0.0001

    # construct method
    def __init__(self) -> None:
        # create component key dictionary
        f = open('../data/fluid_prop.json', 'r')
        self.fluid_dict = json.load(f)

    # calculate saturation temperature or pressure, depending on input
    def vapor_pressure(self, fluid_code, T):
        """
        calculates the vapor pressure for the given fluid at the selected
        temperature

        args: fluid code, temperature in Kelvin

        returns: the saturation pressure in MPa
        """

        # extract the correlation constants
        c = []
        for item in self.fluid_dict[str(fluid_code)]['vapor_pressure']:
            c.append(self.fluid_dict[str(fluid_code)]['vapor_pressure'][item])

        # calculates the correlation terms
        t =[1,1,1,1]
        t[0] = c[0]
        t[1] = c[1]/T
        t[2] = c[2]*np.log(T)
        t[3] = c[3]*(T**c[4])

        # calculates the saturation pressure in MPa
        P = np.exp(np.sum(t))/(10**6)

        return P

    def sat_temperature(self, fluid_code, P):
        """
        calculates the saturation temperature, given the pressure

        args: fluid_code, pressure in MPa

        returns: saturation temperature in K
        """

        # extract the initial pair of temperature to be tested
        a = self.fluid_dict[str(fluid_code)]['sat_T_limits'][0]
        b = self.fluid_dict[str(fluid_code)]['sat_T_limits'][1]

        # calculate initial evaluations of the vapor pressures
        Fa = P - self.vapor_pressure(fluid_code, a)
        Fb = P - self.vapor_pressure(fluid_code, b)

        # loop until the convergence criterion is achieved
        while(abs(Fa*Fb) > self.eps):

            # calculate the point c by the regula false equation
            c = b - ((Fb*(b-a))/(Fb-Fa))

            # evaluate function at point c
            Fc = P - self.vapor_pressure(fluid_code, c)

            # evaluate options
            if (Fa*Fc < 0):
                b = c
            elif(Fc*Fb < 0):
                a = c
            elif(abs(Fc) <= self.eps):
                return c

            # recalculate the evaluations of Fa and Fb
            Fa = P - self.vapor_pressure(fluid_code, a)
            Fb = P - self.vapor_pressure(fluid_code, b)

        return c
