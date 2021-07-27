import pandas as pd
import numpy as np
import json
import joblib
import sys
import os
sys.path.append('.')
from modules.utils_functions import utils

# calculate fluid property
class FluidProperty():

    # constantes importantes
    eps = 0.0001                        # numerical convergence tolerance
    R = 8.314472                        # ideal gas constant (J/mol.K)
    cache_integral = {}                 # cache dictionary to store already calculated integrals
    MODELS_SAVEPATH = './models/'       # saturated models filepath

    # construct method
    def __init__(self) -> None:
        # create component key dictionary
        f = open('./data/fluid_prop.json', 'r')
        self.fluid_dict = json.load(f)

    # calculate saturation temperature or pressure, depending on input
    def vapor_pressure(self, fluid_code: int, T: float) -> float:
        """
        calculates the vapor pressure for the given fluid at the selected
        temperature

        args: fluid code, temperature [K]

        returns: the saturation pressure [MPa]
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

        args: fluid_code, pressure [MPa]

        returns: saturation temperature [K]
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

        args: temperature [K], pressure [MPa]
            fluid code, eos code

        returns: compressibility factor root [adm.]
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
        if eos == 'rk':
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

        args: temperature [K], fluid code

        returns: ideal gas heat capacity in [J/kg.K]
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

        args: temperature [K], fluid code

        returns: the liquid heat capacity in [J/kg.K]
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

    # integrate cp to calculate enthalpy and entropy
    def integrate_cp(self, T_list: list, fluid_code: int, therm_func: str, state: str) -> float:
        """
        performs a numerical integration of heat capacity by using the Simpson composite rule

        args: list containing temperature integration limits [K], fluid_code, thermodynamic function code
            ('h' for enthalpy and 's' for entropy), state of aggregation code ('l' for liquid and 'v' for vapor)

        returns: enthalpy [J/kg] or entropy [J/kg.K]
        """

        # if initial temperature is not informed, use reference state
        Tf = T_list[0]
        if len(T_list) == 1:
            Ti = self.fluid_dict[str(fluid_code)]['Tref']
        else:
            Ti = T_list[1]

        # create hash key to lookup cached integrals
        hash_key = str(round(Tf,2))+str(round(Ti,2))+str(fluid_code)+therm_func+state

        # before calculate the integral, try to found an already calculated value
        try:
            integral = self.cache_integral[hash_key]
        except:
            # if no cached value is saved, calcule the integral numerically

            # define grid points number
            N = 1000

            # create integration grid mesh
            T = np.linspace(Ti, Tf, N)

            # calculate integration grid step
            h = T[1] - T[0]

            # iterate over all temperature points to calculate integral
            integral = 0
            for i in range(N):

                # evaluate function based on state of aggregation informed
                if state == 'v':
                    f = self.ideal_gas_cp(T[i], fluid_code)
                else:
                    f = self.liquid_cp(T[i], fluid_code)

                # if the desired thermodynamic funciont is entropy, divide by T
                if therm_func == 's':
                    f /= T[i]

                # perform correction based on Simpson's rule
                if ((i == 0) | (i == N-1)):
                    f = f
                elif i % 2 == 1:
                    f *= 4
                elif i % 2 == 0:
                    f *= 2

                # update sum of integral
                integral += f

            # after looping, multiply integral by h/3
            integral *= (h/3)

            # cache integral value for future reference
            self.cache_integral[hash_key] = integral

        return integral

    # residual properties
    def residual_properties(self, T: float, P: float, fluid_code: int, eos_code: str, therm_func: str) -> float:
        """
        calculates the residual thermodynamic property using the equation of state approach

        args: temperature [K], pressure [MPa], fluid code, eos code, desired residual thermodynamic function

        returns: residual enthalpy [J/kg] or residual entropy [J/kg.K]
        """

        # extract physical data of the fluid
        MM = self.fluid_dict[str(fluid_code)]['molar_mass']             # molar mass [kg/kmol]
        Tc = self.fluid_dict[str(fluid_code)]['Tc']                     # critical temperature [K]
        Pc = self.fluid_dict[str(fluid_code)]['Pc']                     # critical pressure [MPa]
        w = self.fluid_dict[str(fluid_code)]['omega']                   # accentric factor [adm.]

        # calculate reduced coordinates
        Tr = T/Tc                                                       # reduced temperature
        Pr = P/Pc                                                       # reduced pressure
        
        # create eos parameter dictionary
        eos_dict ={
            'rk':{
                'alpha_Tr': Tr**(-0.5),
                'sigma': 1,
                'eps': 0,
                'omega': 0.08664,
                'psi': 0.42748
            },
            'srk':{
                'alpha_Tr': (1+((0.48 + (1.574*w) - (0.176*(w**2)))*(1-(Tr**(0.5)))))**2,
                'sigma': 1,
                'eps': 0,
                'omega': 0.08644,
                'psi': 0.42748
            },
            'pr':{
                'alpha_Tr': (1+((0.37464 + (1.54226*w) - (0.26992*(w**2)))*(1-(Tr**(0.5)))))**2,
                'sigma': 1+np.sqrt(2),
                'eps': 1-np.sqrt(2),
                'omega': 0.07780,
                'psi': 0.45724
            }
        }

        # select parameters based on eos code
        alpha_Tr = eos_dict[eos_code]['alpha_Tr']
        sigma = eos_dict[eos_code]['sigma']
        eps = eos_dict[eos_code]['eps']
        omega = eos_dict[eos_code]['omega']
        psi = eos_dict[eos_code]['psi']

        # calculate parameters independent from compressibility factor
        beta = omega*(Pr/Tr)
        q = (psi*alpha_Tr)/(omega*Tr)

        # estimate derivative of alpha_Tr
        diff_alphaTr = self.differential_alphaTr(T, fluid_code, eos_code)

        # calculate the compressibility factor
        Z = self.eos_solve(T, P, fluid_code, eos_code)

        # calculate the I factor
        I = (1/(sigma-eps))*np.log((Z+(sigma*beta))/(Z+(eps*beta)))

        # calculate the residual properties
        # residual enthalpy [J/kg.K]
        H_R = ((Z-1+((diff_alphaTr-1)*q*I))*(self.R*T))/(MM/1000)

        # residual entropy [J/kg.K]
        S_R = ((np.log(Z-beta)+(diff_alphaTr*q*I))*self.R)/(MM/1000)

        # return the desired thermodynamic function
        if therm_func == 'h':
            return H_R
        else:
            return S_R

    # derivative of alpha_Tr
    def differential_alphaTr(self, T: float, fluid_code: int, eos_code: str) -> float:
        """
        calculates the derivative of ln (alpha_Tr) in respect to ln (Tr) by numerical approximation
        of centered differences

        args: temperature [K], fluid code, equation of state code

        returns: the derivative estimation of ln (alpha_Tr) in respect to ln (Tr)
        """

        # extract physical properties of the fluid
        w = self.fluid_dict[str(fluid_code)]['omega']           # accentric factor [adm.]
        Tc = self.fluid_dict[str(fluid_code)]['Tc']             # critical temperature [K]

        # calculate the limits of derivative
        T_high = T + self.eps
        T_low = T - self.eps

        # calculate the reduced coordinates of the limits
        Tr_high = T_high/Tc
        Tr_low = T_low/Tc

        # calculate the variation of logarithm of Tr
        lnTr_high = np.log(Tr_high)
        lnTr_low = np.log(Tr_low)
        delta_lnTr = lnTr_high - lnTr_low

        # calculate the alphaTr according to selected eos
        if eos_code == 'rk':
            alphaTr_high = Tr_high**(-0.5)
            alphaTr_low = Tr_low**(-0.5)
        elif eos_code == 'srk':
            alphaTr_high = (1+((0.48 + (1.574*w) - (0.176*(w**2)))*(1-(Tr_high**(0.5)))))**2
            alphaTr_low = (1+((0.48 + (1.574*w) - (0.176*(w**2)))*(1-(Tr_low**(0.5)))))**2
        elif eos_code == 'pr':
            alphaTr_high = (1+((0.37464 + (1.54226*w) - (0.26992*(w**2)))*(1-(Tr_high**(0.5)))))**2
            alphaTr_low = (1+((0.37464 + (1.54226*w) - (0.26992*(w**2)))*(1-(Tr_low**(0.5)))))**2

        # calculate the variation of logarithm of alphaTr
        ln_alphaTr_high = np.log(alphaTr_high)
        ln_alphaTr_low = np.log(alphaTr_low)
        delta_ln_alphaTr = ln_alphaTr_high-ln_alphaTr_low

        # calculate the derivative
        return delta_ln_alphaTr/delta_lnTr

    # thermodynamic property 
    def superHeatVap_prop(self, T: float, P: float, fluid_code: str, eos_code: str, therm_prop: str) -> float:
        """
        calculates the real gas thermodynamic property using the a equation of state approach
        to calculate the residual properties. The ideal gas approach is calculated by integration of cp
        This method applies only to real vapor phase

        args: temperature [K], pressure [MPa], fluid code, eos code, thermodynamic function

        returns: enthalpy [J/kg] / entropy [J/kg.K]
        """

        # extract fluid molar mass
        MM = self.fluid_dict[str(fluid_code)]['molar_mass']

        # calculate the ideal gas part of the thermodynamic property
        ideal_part = self.integrate_cp([T, self.fluid_dict[str(fluid_code)]['Tref']], fluid_code, 
                                        therm_prop, state = 'v')

        # if the entropy is desired, the pressure correction must be made
        if therm_prop == 's':
            ideal_part -= (self.R/(MM/1000))*np.log(P/self.fluid_dict[str(fluid_code)]['Pref'])

        # calculate the residual property
        residual_part = self.residual_properties(T, P, fluid_code, eos_code, therm_prop)

        # calculate the desired property
        return ideal_part + residual_part

    # thermodynamic hypothetic paths to calculate properties
    def calculate_thermo_prop(self, T: float, P: float, fluid_code: int, x: float) -> list:
        """
        calcultes the thermodynamic properties (enthalpy and entropy) based on the 
        selected fluid according to the appropriate reference state

        args: temperature [K], pressure [MPa], fluid code, vapor fraction

        returns: list with enthalpy [J/kg] and entropy [J/kg.K]
        """

        # extract fluid reference state and eos
        Tref = self.fluid_dict[str(fluid_code)]['Tref']
        Pref = self.fluid_dict[str(fluid_code)]['Pref']
        eos_code = self.fluid_dict[str(fluid_code)]['eos']
        comp_name = self.fluid_dict[str(fluid_code)]['name']
        MM = self.fluid_dict[str(fluid_code)]['molar_mass']

        # load saturation models
        models_list = os.listdir(self.MODELS_SAVEPATH)
        models_list = [k for k in models_list if k.find(comp_name)!=-1]
        h_models = [k for k in models_list if k.find('Enthalpy')!=-1]
        s_models = [k for k in models_list if k.find('Entropy')!=-1]

        # select case for property calculation
        case = self.property_case(T, P, fluid_code)
        
        # calculate properties based on the thermodynamic case
        if case in [1, 2, 4]:
            # if the case is 1, 2 or 4, then the fluid is in superheated vapor state, and then
            # the properties are only the delta between the real state and the reference state
            h = self.superHeatVap_prop(T, P, fluid_code, eos_code, 'h') - self.superHeatVap_prop(Tref, Pref, fluid_code, eos_code, 'h')
            s = self.superHeatVap_prop(T, P, fluid_code, eos_code, 's') - self.superHeatVap_prop(Tref, Pref, fluid_code, eos_code, 's')
        elif case in [5]:
            # if the case is 3, then the fluid is saturated mixture at
            # reference temperature and thus, the saturation properties must be applied
            h_vap_model = joblib.load(os.path.join(self.MODELS_SAVEPATH, [m for m in h_models if m.find('Vapor')!=-1][0]))
            h_liquid_model = joblib.load(os.path.join(self.MODELS_SAVEPATH, [m for m in h_models if m.find('Liquid')!=-1][0]))

            # prediction of saturated vapor and liquid enthalpy and conversion to J/kg
            h_sat_vap = (h_vap_model.predict([[T,P]]))*1000/(MM/1000)
            h_sat_liq = (h_liquid_model.predict([[T,P]]))*1000/(MM/1000)


            # correction in respect to reference state
            h_sat_vap -= self.superHeatVap_prop(Tref, Pref, fluid_code, eos_code, 'h')
            h_sat_liq -= self.superHeatVap_prop(Tref, Pref, fluid_code, eos_code, 'h')

        # return [h,s]

    # case for property calculation
    def property_case(self, T: float, P: float, fluid_code: int) -> int:
        """
        determines which is the case of fluid property calculation by comparing 
        the system's temperature and pressure to reference state

        args: temperature [K], pressure [MPa], fluid code
        """

        # extract fluid reference state and eos
        Tref = self.fluid_dict[str(fluid_code)]['Tref']
        Pref = self.fluid_dict[str(fluid_code)]['Pref']
        eos_code = self.fluid_dict[str(fluid_code)]['eos']

        # compare and return approriate cases
        if (T>Tref):
            if P < Pref:
                return 1
            elif P == Pref:
                return 2
            else:
                return 3
        elif (T == Tref):
            if P < Pref:
                return 4
            elif P == Pref:
                return 5
            else:
                return 6
        else:
            if P < Pref:
                return 7
            elif P == Pref:
                return 8
            else:
                return 9
