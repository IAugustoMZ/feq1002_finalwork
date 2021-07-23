from numpy.lib.function_base import corrcoef
import pandas as pd
import numpy as np
import os
import json
from utils_functions import utils

# calculate fluid property
class FluidProperty():

    # constantes importantes
    eps = 0.0001            # numerical convergence tolerance
    R = 8.314472            # ideal gas constant (J/mol.K)
    Tref = 298.15           # reference state temperature (K)
    Pref = 101325           # reference state pressure (Pa)

    # construct method
    def __init__(self) -> None:
        # create component key dictionary
        f = open('../data/fluid_prop.json', 'r')
        self.fluid_dict = json.load(f)

    # calculate saturation temperature or pressure, depending on input
    def vapor_pressure(self, fluid_code: int, T: float) -> float:
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

    def sat_temperature(self, fluid_code:int, P: float) -> float:
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

        # for the cases where a or b are the roots
        if (abs(Fa) < self.eps):
            return a
        elif (abs(Fb) < self.eps):
            return b

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

    # calculate parameters of cubic equation of state
    def eos_solve(self, T:float, P: float, fluid_code: int, eos: str) -> float:
        """
        calculate the root of cubic equation of state, based on
        compressibility factor polynomial form

        args: temperature in Kelvin, pressure in MPa
            fluid code, eos code

        returns: compressibility factor root
        """

        # extract critical constants for the selected fluid
        Tc = self.fluid_dict[str(fluid_code)]['Tc']                 # critical temperature [K]
        Pc = self.fluid_dict[str(fluid_code)]['Pc']*(1*(10**6))     # critical pressure [Pa]
        w = self.fluid_dict[str(fluid_code)]['omega']               # accentric factor

        # calculate the reduced temperature
        Tr = T/Tc

        # convert inputed pressure to Pa
        P = P*(1*(10**6))

        # calculate attraction and repulsion parameters based on chosen eos
        if eos == 'vW':
            # use van der Waals constants
            u = 0
            m = 0
            b = (self.R*Tc)/(8*Pc)
            a = (27*(self.R**2)*(Tc**2))/(64*Pc)
        elif eos == 'rk':
            # use Redlich-Kwong constants
            u = 1
            m = 0
            b = (0.08664*self.R*Tc)/Pc
            a = (0.42748*((self.R**2)*(Tc**(2.5))))/(Pc*(T**(0.5)))
        elif eos == 'srk':
            # use Soave-Redlich-Kwong constants
            u = 1
            m = 0
            b = (0.08664*self.R*Tc)/Pc
            f_w = 0.48 + (1.574*w) - (0.176*(w**2))
            corr_fac = (1+(f_w*(1-(Tr**(0.5)))))**2
            a = (0.42748*((self.R*Tc)**2))/(Pc)
            a = a*corr_fac
        elif eos == 'pr':
            # use Peng-Robinson constants
            u = 2
            m = -1
            b = (0.07780*self.R*Tc)/Pc
            f_w = 0.37464 + (1.54226*w) - (0.26992*(w**2))
            corr_fac = ((1+(f_w*(1-(Tr**(0.5)))))**2)
            a = (0.45724*((self.R*Tc)**2))/(Pc)
            a = a*corr_fac

        # calculate constants A and B
        A = (a*P)/((self.R*T)**2)
        B = (b*P)/(self.R*T)

        # calculate polynomial coefficients
        z2 = -(1+B-(u*B))
        z1 = (A+(m*(B**2))-(u*B)-(u*(B**2)))
        z0 = -((A*B)+(m*(B**2))+(m*(B**3)))

        # solve polynomial form of compressibility factor
        Z = np.roots([1, z2, z1, z0])

        # get the max root
        Z = float(max(Z))

        return Z

    # ideal gas heat capacity
    def ideal_gas_cp(self, T: float, fluid_code: int) -> float:
        """
        evaluates the expression of ideal gas heat capacity based on 
        chosen fluid

        args: temperature in Kelvin, fluid code

        returns: ideal gas heat capacity in J/kg.K
        """

        # extract fluid molar mass and correlation constants
        MM = self.fluid_dict[str(fluid_code)]['molar_mass']
        cp_cts = []
        for item in self.fluid_dict[str(fluid_code)]['ideal_gas_cp']:
            cp_cts.append(self.fluid_dict[str(fluid_code)]['ideal_gas_cp'][item])

        # calculate the ideal gas heat capacity
        t1 = cp_cts[0]
        t2 = cp_cts[1]*(((cp_cts[2]/T)/np.sinh(cp_cts[2]/T))**2)
        t3 = cp_cts[3]*(((cp_cts[4]/T)/np.cosh(cp_cts[4]/T))**2)

        Cp = t1 + t2 + t3

        # convert the Cp to J/kg.K
        Cp = Cp/MM

        return Cp

    # liquid heat capacity
    def liquid_cp(self, T: float, fluid_code: int) -> float:
        """
        calculates the value of liquid heat capacity

        args: temperature in Kelvin, fluid code

        returns: the liquid heat capacity in J/kg.K
        """

        # extract fluid molar mass, critical temperature and correlation constants
        MM = self.fluid_dict[str(fluid_code)]['molar_mass']
        Tc = self.fluid_dict[str(fluid_code)]['Tc']
        cp_cts = []
        for item in self.fluid_dict[str(fluid_code)]['liquid_cp']:
            cp_cts.append(self.fluid_dict[str(fluid_code)]['liquid_cp'][item])

        # if the fluid is ammonia or methane, use the adequate correlation
        if fluid_code in [1,2]:
            t = 1-(T/Tc)
            t1 = (cp_cts[0]**2)/t
            t2 = cp_cts[1]
            t3 = -2*cp_cts[0]*cp_cts[2]*t
            t4 = -cp_cts[0]*cp_cts[3]*(t**2)
            t5 = -(((cp_cts[2]**2)*(t**3))/3)
            t6 = -(((cp_cts[2]*cp_cts[3])*(t**4))/2)
            t7 = -(((cp_cts[3]**2)*(t**5))/5)
            Cp = t1 + t2 + t3 + t4 + t5 + t6 + t7
        else:
            t1 = cp_cts[0]
            t2 = cp_cts[1]*T
            t3 = cp_cts[2]*(T**2)
            t4 = cp_cts[3]*(T**3)
            t5 = cp_cts[4]*(T**4)
            Cp = t1 + t2 + t3 + t4 + t5

        # convert the Cp to J/kg.K
        Cp = Cp/MM

        return Cp
