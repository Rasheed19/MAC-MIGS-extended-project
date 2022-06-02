import os
import pickle
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.svm import NuSVR
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import time
import pprint
from utils import utils_gn
import importlib
importlib.reload(utils_gn)
xgb.set_config(verbosity=0)



def exponential_decay(x, a, k, b):
    return a * np.exp(-k*x) + b

def exponential_growth(x, a, k, b):
    return a * np.exp(k*x) + b

def linear_model(x, a, b):
    return a + b*x

def curve_fitting(model_func, 
                  initial_guess,
                  x_data,
                  y_data,
                  plot=False):

    # curve fit
    popt, pcov = curve_fit(model_func, x_data, y_data, initial_guess, maxfev=50000)
    
    # test result
    if plot==True:
        
        fig, ax = plt.subplots()
        ax.plot(x_data, model_func(x_data, *popt), color='r', label='fit function')
        ax.plot(x_data, y_data, 'bo', label='data with noise')
        ax.legend(loc='best')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid('On')

        plt.show()

    return popt

def metrics_calculator(y_true, y_pred):
    '''
    A function that calcualtes regression metrics.

    Arguments:
              y_true:  an array containing the true values of y
              y_pred:  an array containin the predicted values of y
    Returns:
            MAE, MAPE, MSE, RMSE, and R2 score.
    '''
    return  {'MAE': mean_absolute_error(y_true, y_pred), 'MAPE': mean_absolute_percentage_error(y_true, y_pred), 'MSE': mean_squared_error(y_true, y_pred),
               'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 'R2 score': r2_score(y_true, y_pred)}


def plot_prediction_experimental(y_train_true, y_train_pred, y_test_true, y_test_pred, fname):
    '''
    A function that plots predicted EOL against experimental EOL.
    Args:
         y_train_true, y_test_true:   the true values of EOL for the training and test respectively
         y_train_pred, y_test_pred:   the predicted values of EOL for the training and test respectively
         fname:    name to save the figure with
    '''

    fig, ax = plt.subplots(1, 2, figsize=(15,10))

    for i, s, pair in zip((0, 1), ('(train)', '(test)'), ((y_train_true, y_train_pred), (y_test_true, y_test_pred))):
        ax[i].scatter(pair[0], pair[1], s=100, color='purple', alpha=0.5, zorder=10)
        lims = [
        np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
        np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
        ]
        # now plot both limits against each other
        ax[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax[i].set_aspect('equal')
        ax[i].set_xlim(lims)
        ax[i].set_ylim(lims)
        ax[i].set_xlabel('Experimental EOL ' + s, fontsize=12)
        ax[i].set_ylabel('Predicted EOL ' + s, fontsize=12)

    plt.savefig(fname="plots/"+"pred_vs_true_"+fname, bbox_inches='tight')
    plt.show()

def repeated_kfold_cross_validation(model, df, n_splits, n_repeats, feature_selection, scaling, k=None):
    '''
    A function that perfroms k-fold cross validation n_repeats times.

    Arguments:
              model:  fitted model to be validated 
              df:     dataframe containing features and target to be used for validation
              n_splits: number of splits to be used during validation
              n_repeats: how many times to repeat the process of k-fold cross validation
              feature_selection:  a boolean to specify whether to perfrom feature selection or not
              scaling:            a boolean to specify whether to perform data scaling or not
              k:                  fraction of features to select if feature selection is set to true
    Returns:
            a dictionary with key as test score and value as (score value, std of score value).
    '''

    # get the features and the target 
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

    # feature selection option
    if feature_selection == True:
        X, _, _ = utils_gn.univariate_feature_selection(df, k=k)
    
    # scaling option
    if scaling == True:
        X = utils_gn.scaler(X)

    # define the cv object 
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    # define metrics to be used
    metrics = {'MAE': 'neg_mean_absolute_error', 'MAPE': 'neg_mean_absolute_percentage_error', 'MSE': 'neg_mean_squared_error', 'R2 score': 'r2'}

    # calculate scores
    scores = cross_validate(model, X, y, scoring=metrics, cv=cv, n_jobs=-1)
    scores = {key:(abs(val).mean(), abs(val).std()) for key, val in scores.items() if key in ['test_'+metric for metric in metrics.keys()]}
    
    return scores


def fit_tree_based_regression(df, test_size, feature_selection, scaling, params, plot, fname, model_type='xgb', k=None):
    '''
    A function that fits XGBoost Regression to data.

    Arguments:
              df:                 pandas dataframe that contains the features and the target
              test_size:          percentage of data to be used for testing
              feature_selection:  a boolean to specify whether to perfrom feature selection or not
              scaling:            a boolean to specify whether to perform data scaling or not
              params:             a dictionary of parameters to pass to the regression object
              plot:               a boolean to specify whether to plot feature importance
              fname:              name to save the model
              model_type:         either to fit XGBoost ('xgb') or Extratrees ('ext') regression
              k:                  fraction of features to select if feature selection is set to true
    
    Returns:
            model object and print dictionary of metrics(mae, mape, mse, rmse, r2)
              
    '''

    # get the features and the target 
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

    # feature selection option
    if feature_selection == True:
        X, _, _ = utils_gn.univariate_feature_selection(df, k=k)
    
    # scaling option
    if scaling == True:
        X = utils_gn.scaler(X)
    
    # perform test-train split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # fit the model and get the feature importances 
    print("Tree-based regression has started...") 
    start_time = time.time()
    
    if model_type == 'xgb':
        model = XGBRegressor(**params)
    
    if model_type == 'ext':
        model = ExtraTreesRegressor(**params)

    model = model.fit(X_train, y_train)
    
    # make predictions
    y_pred_train = model.predict(X_train)  # for the train
    y_pred_test = model.predict(X_test)   # for the test

    end_time = time.time()
    print('Tree-based regression has ended after {} seconds'.format(np.round(end_time - start_time, 2)))
    
    # calculate metrics 
    metrics = metrics_calculator(y_test, y_pred_test)
    print('------------------')
    print('Model metrics:')
    print('------------------')
    pprint.pprint(metrics)
    
    
    # option to plot feature importance, plot the first 30 most important features
    if plot == True:
        features, feature_importance = utils_gn.feature_importance_ordering(df.columns[:-1], model.feature_importances_)
        utils_gn.feature_importance_barchart(features[:30], feature_importance[:30], 'Normalized feature importance', fname)
        plot_prediction_experimental(y_train, y_pred_train, y_test, y_pred_test, fname)
    
    # save the model as pickle file
    with open(os.path.join("models", fname), "wb") as fp:
        pickle.dump(model, fp)
    
    return model, metrics


def fit_nusvr(df, test_size, feature_selection, scaling, params, fname, plot=False, k=None):
    '''
    A function that fits XGBoost Regression to data.

    Arguments:
              df:                 pandas dataframe that contains the features and the target
              test_size:          percentage of data to be used for testing
              feature_selection:  a boolean to specify whether to perfrom feature selection or not
              scaling:            a boolean to specify whether to perform data scaling or not
              params:             a dictionary of parameters to pass to the regression object
              plot:               a boolean to specify whether to plot feature importance
              fname:              name to save the model
              k:                  fraction of features to select if feature selection is set to true
    
    Returns:
            model object and print dictionary of metrics(mae, mape, mse, rmse, r2)
              
    '''

    # get the features and the target 
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

    # feature selection option
    if feature_selection == True:
        X, _, _ = utils_gn.univariate_feature_selection(df, k=k)
    
    # scaling option
    if scaling == True:
        X = utils_gn.scaler(X)
    
    # perform test-train split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # fit the model and get the feature importances 
    print("NuSVR training has started...") 
    start_time = time.time()

    model = NuSVR(**params)
    model = model.fit(X_train, y_train)
        
    # make predictions
    y_pred_train = model.predict(X_train)  # for the train
    y_pred_test = model.predict(X_test)   # for the test

    end_time = time.time()
    print('NuSVR training has ended after {} seconds'.format(np.round(end_time - start_time, 2)))
    
    # calculate metrics 
    metrics = metrics_calculator(y_test, y_pred_test)
    print('------------------')
    print('Model metrics:')
    print('------------------')
    pprint.pprint(metrics)
    
    if params['kernel']=='linear' and plot==True:
        features, feature_importance = utils_gn.feature_importance_ordering(df.columns[:-1], np.abs(np.ravel(model.coef_)))
        utils_gn.feature_importance_barchart(features[:30], feature_importance[:30], 'Normalized feature weight', fname)

    # option to plot feature importance, plot the first 30 most important features
    if plot == True:
        plot_prediction_experimental(y_train, y_pred_train, y_test, y_pred_test, fname)
    
    # save the model as pickle file
    with open(os.path.join("models", fname), "wb") as fp:
        pickle.dump(model, fp)
    
    return model, metrics








