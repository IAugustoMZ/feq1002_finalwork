import os
import json
import joblib
import property
import utils_functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# validation methods
class validate():

    # folder containing models (this file system architecture)
    MODEL_SAVE_FILEPATH = '../models'

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

    # saturation properties
    def saturation_properties(self, data, fluid_code, property):
        """
        method to validate the machine learning models to predict saturation properties using 
        temperature and pressure as inputs

        args: experimental data, fluid code representing the desired fluid, the desired property to 
            be validated

        returns: plots comparing the experimental data to predicted values
        """

        # create a dictionary to select models
        property_dict = {
            'v': 'Volume',
            'h': 'Enthalpy',
            's': 'Entropy'
        }

        # import fluid name
        comp_name = self.fluid_dict[str(fluid_code)]['name']

        # concatenate property + fluid name
        query_str = property_dict[property] + '_' + comp_name

        # load list of models
        model_list = os.listdir(self.MODEL_SAVE_FILEPATH)

        ######################################### SATURATED LIQUID PROPERTIES ############################################
        
        # create dataframe to receive predictions
        pred_liq = data.loc[data['x']==0, ['t', 'p']]

        # query full model list by state, property and fluid
        query_str_liq = 'SatLiquid'+query_str
        model_selected = [k for k in model_list if k.find(query_str_liq) != -1]

        # load queried model
        model = joblib.load(os.path.join(self.MODEL_SAVE_FILEPATH, model_selected[0]))

        # predict the desired property using temperature and pressure
        pred_liq[property] = model.predict(pred_liq[['t', 'p']])

        ######################################### SATURATED VAPOR PROPERTIES #############################################

        # create dataframe to receive predictions
        pred_vap = data.loc[data['x']==1, ['t', 'p']]

        # query full model list by state, property and fluid
        query_str_vap = 'SatVapor'+query_str
        model_selected = [k for k in model_list if k.find(query_str_vap) != -1]

        # load queried model
        model = joblib.load(os.path.join(self.MODEL_SAVE_FILEPATH, model_selected[0]))

        # predict the desired property using temperature and pressure
        pred_vap[property] = model.predict(pred_vap[['t', 'p']])

        ###################################################################################################################

        # plot saturation curves
        fig = plt.figure(figsize=(14,7))

        # saturation propery as function of temperature
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(data[property], data['t'],'ko', label = 'Experimental Data')
        ax1.plot(pred_liq[property], pred_liq['t'],'r-', label = 'Model Prediction - Saturated Liquid')
        ax1.plot(pred_vap[property], pred_vap['t'],'r-', label = 'Model Prediction - Saturated Vapor')
        ax1.set_xlabel(property_dict[property] + ' - [' + self.fluid_dict['prop_units'][property] + ']', size = 14)
        ax1.set_ylabel('Temperature - [' + self.fluid_dict['prop_units']['T'] + ']', size = 14)
        ax1.legend(prop={'size': 12})
        ax1.grid()

        # saturation propery as function of pressure
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(data[property], data['p'],'ko', label = 'Experimental Data')
        ax2.plot(pred_liq[property], pred_liq['p'],'r-', label = 'Model Prediction - Saturated Liquid')
        ax2.plot(pred_vap[property], pred_vap['p'],'r-', label = 'Model Prediction - Saturated Vapor')
        ax2.set_xlabel(property_dict[property] + ' - [' + self.fluid_dict['prop_units'][property] + ']', size = 14)
        ax2.set_ylabel('MPa - [' + self.fluid_dict['prop_units']['P'] + ']', size = 14)
        ax2.legend(prop={'size': 12})
        ax2.grid()        