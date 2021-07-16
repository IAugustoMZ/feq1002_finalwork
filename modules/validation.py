import pandas as pd
import numpy as np
import os
import json
import property
import utils_functions
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# validation methods
class validate():

    # constructor method
    def __init__(self) -> None:
        # create component key dictionary
        f = open('../data/fluid_prop.json', 'r')
        self.fluid_dict = json.load(f)

    # vapor pressure validation
    def vapor_pressure(self, data, fluid_code):
        """
        method for validation of vapor pressure using experimental data

        args: experimental data, fluid code
        returns: root main squared error, mean absolute error, R2, plots
        """

        # create class object instance
        fp = property.FluidProperty()
        ut = utils_functions.utils()

        # iterate over experimental temperatures and calculate vapor pressure
        vp_hat = []
        for i in range(data.shape[0]):
            T = data.loc[i,'t']
            vp_hat.append(fp.vapor_pressure(fluid_code = fluid_code, T = T))

        # calculate performance metrics
        MAE = mean_squared_error(data['p'], vp_hat)
        RMSE = np.sqrt(mean_squared_error(data['p'], vp_hat))
        r2 = r2_score(data['p'], vp_hat)

        # print performance information
        print('[INFO] Mean Absolute Error: %.4f'%MAE)
        print('[INFO] Root Mean Squared Error: %.4f'%RMSE)
        print('[INFO] R2 Score: %.4f'%r2)

        # create parameters for plot method
        values_list = [list(data['p']), vp_hat]
        var_name = str('Vapor Pressure [MPa]')

        # plot prediction analysis
        ut.scatter_model_residues(values_list, var_name)
    
    # saturation temperature
    def sat_temperature(self, data, fluid_code):
        """
        method for validation of saturation temperature model

        args: experimental data, fluid code

        returns: mean absolute error, root mean squared error, r2 score, plots
        """

        # create class object instance
        fp = property.FluidProperty()
        ut = utils_functions.utils()

        # iterate over experimental temperatures and calculate vapor pressure
        st_hat = []
        for i in range(data.shape[0]):
            P = data.loc[i,'p']
            st_hat.append(fp.sat_temperature(fluid_code, P))

        # calculate performance metrics
        MAE = mean_squared_error(data['t'], st_hat)
        RMSE = np.sqrt(mean_squared_error(data['t'], st_hat))
        r2 = r2_score(data['t'], st_hat)

        # print performance information
        print('[INFO] Mean Absolute Error: %.4f'%MAE)
        print('[INFO] Root Mean Squared Error: %.4f'%RMSE)
        print('[INFO] R2 Score: %.4f'%r2)

        # create parameters for plot method
        values_list = [list(data['t']), st_hat]
        var_name = str('Saturation Temperature [K]')

        # plot prediction analysis
        ut.scatter_model_residues(values_list, var_name)
