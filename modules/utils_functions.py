import pandas as pd
import numpy as np
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