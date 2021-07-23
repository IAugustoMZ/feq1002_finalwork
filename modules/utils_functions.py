import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# utilitary functions
class utils:

    # important constants
    RANDOM_SEED = 1002                  # seed for reproducibility
    MODEL_SAVE_FILEPATH = '../models'   # filepath for saving models (depends on the file system architecture)

    # construct method
    def __init__(self) -> None:
        # create component key dictionary
        f = open('../data/fluid_prop.json', 'r')
        self.fluid_dict = json.load(f)


    # import dataset and filter by fluid and saturated state
    def load_filter_data(self, vapor_fraction: float, fluid_code: int, complete_filepath: str) -> pd.DataFrame:
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
    def scatter_model_residues(self, values_list: list, var_name: str) -> None:
        """
        plots the expected and the predicted values from a specific model

        args: vector of expected (real) values, vector of predicted values
            string containing variable name

        returns: plots
        """

        # extract values
        expected = values_list[0]
        predicted = values_list[1]

        # create dataframe for residue calculation
        res_df = pd.DataFrame(expected, columns = ['real'])
        res_df['predicted'] = predicted
        res_df['residues'] = res_df['real'] - res_df['predicted']

        # build plots
        fig = plt.figure(figsize=(14,7))

        # expected vs predicted
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(res_df['real'], res_df['real'], 'r-', label = 'Perfect Prediction')
        ax1.plot(res_df['real'], res_df['predicted'], 'ko', label = 'True Prediction')
        ax1.set_xlabel(var_name + '\nExpected Values', size = 14)
        ax1.set_ylabel(var_name + '\nPredicted Values', size = 14)
        ax1.set_title('Prediction Assessment\n' + var_name, size = 16)
        ax1.legend(prop={'size': 12})
        ax1.grid()

        # residues histogram
        ax2 = fig.add_subplot(1,2,2)
        ax2.hist(res_df['residues'], color = 'lightgreen', alpha = 0.5)
        ax2.set_xlabel(var_name + '\nResidues', size = 14)
        ax2.set_ylabel('Frequency', size = 14)
        ax2.set_title('Prediction Errors Distribution\n' + var_name, size = 16)
        ax2.grid()

    # model training and evaluation
    def model_training_selection(self, x, y, property_name, times_to_interpolate = 1, decision_score = 'mae'):
        """
        performs model training, cross validation and selection using
        KFold cross validation. The models tested are LinearRegression or DecisionTrees
        
        args: predictors arrays and target array, string containing property name,
            integer defining the number os successive linear interpolations,
            string containing the score that will be used to select model (default is 'MAE')
        
        returns: save best model and average performance index
        """

        # create model dictionary
        model_dict = {
            1: {
                'name': 'Linear Regression',
                'model': LinearRegression(),
                'cross_val_results': 0,
                'test_perf_index': {
                    'mae': 0,
                    'rmse': 0,
                    'r2': 0
                },
            },
            2: {
                'name': 'Decision Tree Regressor',
                'model': DecisionTreeRegressor(random_state=self.RANDOM_SEED),
                'cross_val_results': 0,
                'test_perf_index':{
                    'mae': 0,
                    'rmse': 0,
                    'r2': 0
                }
            }
        }

        # perform data augmentation using prior chemical engineering knowledge
        x, y = self.data_augmentation(x, y, times_to_interpolate=times_to_interpolate)

        # performing train-test split to create a holdout set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.RANDOM_SEED)

        # using the train set, perform K fold cross validation
        cv = KFold(n_splits = 10, shuffle=True, random_state=self.RANDOM_SEED)

        # train and cross validate each model
        for item in model_dict:

            # selecting model
            model = model_dict[item]['model']

            # training model using cross validation
            scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error',
                    cv = cv)

            # store results in model_dictionary
            model_dict[item]['cross_val_results'] = np.mean(abs(scores))

            # log of results
            print('[INFO] ' + model_dict[item]['name'] + ' cross validation MAE: %.4f'%(np.mean(abs(scores))))

            # after cross validation, train model in full train set
            model_dict[item]['model'] = model_dict[item]['model'].fit(x_train, y_train)

            # make predictions at test set
            yhat = model_dict[item]['model'].predict(x_test)

            # calculate performance parameters
            model_dict[item]['test_perf_index']['mae'] = mean_absolute_error(y_test, yhat)
            model_dict[item]['test_perf_index']['rmse'] = np.sqrt(mean_squared_error(y_test, yhat))
            model_dict[item]['test_perf_index']['r2'] = r2_score(y_test, yhat)

            # log of results
            print('[INFO] ' + model_dict[item]['name'] + ' holdout set MAE: %.4f'%(model_dict[item]['test_perf_index']['mae']))
            print('[INFO] ' + model_dict[item]['name'] + ' holdout set RMSE: %.4f'%(model_dict[item]['test_perf_index']['rmse']))
            print('[INFO] ' + model_dict[item]['name'] + ' holdout set R2: %.4f'%(model_dict[item]['test_perf_index']['r2']))
            print('-------------------------------------------------------------------------------------------------')

        # after training and evaluating models, select based on desired performance score
        if decision_score in ['mae', 'rmse']:
            # select model based on lower mae or rmse
            BEST_SCORE = 1000
            for item in model_dict:
                if model_dict[item]['test_perf_index'][decision_score] < BEST_SCORE:
                    BEST_SCORE = model_dict[item]['test_perf_index'][decision_score]
                    BEST_MODEL = model_dict[item]['model']
                    model_name = model_dict[item]['name'].lower().replace(' ', '') + '_' + property_name + '.m'

        elif decision_score in ['r2']:
            # select model based on higher r2 score
            BEST_SCORE = 0
            for item in model_dict:
                if model_dict[item]['test_perf_index'][decision_score] > BEST_SCORE:
                    BEST_SCORE = model_dict[item]['test_perf_index'][decision_score]
                    BEST_MODEL = model_dict[item]['model']
                    model_name = model_dict[item]['name'].lower().replace(' ', '') + '_' + property_name + '.m'
        
        # after selecting the model, print saving information and save the model
        print('[INFO] Saving ' + model_name[:-2])
        joblib.dump(BEST_MODEL, os.path.join(self.MODEL_SAVE_FILEPATH, model_name))

    # data augmentation through interpolation
    def data_augmentation(self, x, y, sort_var_name = 't', times_to_interpolate = 1):
        """
        performs data augmentation based on prior knowledge of chemical engineering
        which states that if we have a thermodynamic relation between two intensive variables X and Y,
        and provided that we have at least two experimental points from this relation - (X1, Y1) and (X2, Y2), then
        the value of Y = (Y1 + Y2)/2 can be estimated by a model evaluated at X = (X1 + X2)/2
        This method extends this knowledge by calculating the average of experimental points iteratively and thus,
        promoting a kind of data augmentation.

        args: predictors matrix, target matrix, sorting variable name, desired number of successive interpolations

        returns: augmented dataset of predictors and targets
        """
        # joining sorting variable in target dataset
        y = y.join(x[sort_var_name])

        # ensure sorting before begin interpolations
        x = x.sort_values(by=[sort_var_name])
        y = y.sort_values(by=[sort_var_name])

        # loop to calculate successive interpolations
        for i in range(times_to_interpolate):
            
            # interpolate the predictors matrix and the target matrix
            x_aux = x.rolling(2).mean().dropna()
            y_aux = y.rolling(2).mean().dropna()

            # concatenate the augmented data to original values
            x = pd.concat([x, x_aux])
            y = pd.concat([y, y_aux])

            # sort datasets to prepare them for new interpolation
            x = x.sort_values(by=[sort_var_name]).reset_index().drop(['index'], axis = 1)
            y = y.sort_values(by=[sort_var_name]).reset_index().drop(['index'], axis = 1)

        # drop unused columns from target dataset
        y = y.drop([sort_var_name], axis = 1)

        return x, y
