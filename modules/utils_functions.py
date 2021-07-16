import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# utilitary functions
class utils:

    # construct method
    def __init__(self) -> None:
        # create component key dictionary
        f = open('../data/fluid_prop.json', 'r')
        self.fluid_dict = json.load(f)


    # import dataset and filter by refrigerant and saturated state
    def load_filter_data(self, vapor_fraction, fluid_code, complete_filepath):
        """
        load data from the selected filepath, filters it by the selected saturation state
        and by the selected fluid

        args: vapor fraction representing the desired saturated state 
            (0 for saturared liquid or 1 for saturated vapor),
            fluid_code (1 - methane, 2 - ammonia, 3 - water),
            filepath where the saturation data is stored
        
        returns: a dataframe containing the filtered data
        """

        # extract the desired fluid
        fluid = self.fluid_dict[str(fluid_code)]['name']

        # load the data from the filepath
        data = pd.read_excel(complete_filepath, sheet_name=fluid)

        # filter data by saturation state
        data = data.loc[data['x']==vapor_fraction,:]

        return data

    # scatterplot of residues and predictions
    def scatter_model_residues(self, values_list, var_name):
        """
        plots the expected and the predicted values from a specific model

        args: vector of expected (real) values, vector of predicted values
            string containing variable name

        returns: plots
        """

        # extract values
        expected = values_list[0]
        predicted = values_list[1]

        # calculate residues
        residues = [e-p for e,p in zip(expected, predicted)]


        # build plots
        fig = plt.figure(figsize=(14,7))

        # expected vs predicted
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(expected, expected, 'r-', label = 'Perfect Prediction')
        ax1.plot(expected, predicted, 'ko', label = 'True Prediction')
        ax1.set_xlabel(var_name + '\nExpected Values', size = 14)
        ax1.set_ylabel(var_name + '\nPredicted Values', size = 14)
        ax1.set_title('Prediction Assessment\n' + var_name, size = 16)
        ax1.legend(prop={'size': 12})
        ax1.grid()

        # residues histogram
        ax2 = fig.add_subplot(1,2,2)
        ax2.hist(residues, color = 'lightgreen', alpha = 0.5)
        ax2.set_xlabel(var_name + '\nResidues', size = 14)
        ax2.set_ylabel('Frequency', size = 14)
        ax2.set_title('Prediction Errors Distribution\n' + var_name, size = 16)
        ax2.grid()
