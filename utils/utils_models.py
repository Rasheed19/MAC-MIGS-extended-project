import os
import pickle
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import NuSVR
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_validate, train_test_split
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
    '''
    A function that returns an exponential decay function
    '''
    return a * np.exp(-k*x) + b

def exponential_growth(x, a, k, b):
    '''
    A function that returns an exponential growth function
    '''
    return a * np.exp(k*x) + b

def linear_model(x, a, b):
    '''
    A function that returns a linear function
    '''
    return a + b*x

def curve_fitting(model_func, 
                  initial_guess,
                  x_data,
                  y_data,
                  plot=False):
    '''
    A function that fits a given function to a data and return the corresponding estimated parameters
    '''

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
               'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 'R2 score': r2_score(y_true, y_pred), 'Corr': stats.pearsonr(y_true, y_pred)[0]}


def plot_prediction_experimental(y_train_true, y_train_pred, y_test_true, y_test_pred, fname, title=None, plot_mode=1):
    '''
    A function that plots predicted EOL against experimental EOL.
    Args:
         y_train_true, y_test_true:   the true values of EOL for the training and test respectively
         y_train_pred, y_test_pred:   the predicted values of EOL for the training and test respectively
         fname:                       name to save the figure with
         plot_mode:                   plot type to use 
    '''
    def axis_to_fig(axis):
       fig = axis.figure
       def transform(coord):
            return fig.transFigure.inverted().transform(
                axis.transAxes.transform(coord))
       return transform
    
    def add_sub_axes(axis, rect):
        fig = axis.figure
        left, bottom, width, height = rect
        trans = axis_to_fig(axis)
        figleft, figbottom = trans((left, bottom))
        figwidth, figheight = trans([width,height]) - trans([0,0])
        return fig.add_axes([figleft, figbottom, figwidth, figheight])
    
    if plot_mode == 0:

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
            ax[i].set_xlabel('Experimental end-of-life ' + s, fontsize=12)
            ax[i].set_ylabel('Predicted end-of-life ' + s, fontsize=12)

    elif plot_mode == 1:

        fig, axes = plt.subplots(1, 2, figsize=(15,10))

        for axis, s, pair in zip(axes.ravel(), ('(train)', '(test)'), ((y_train_true, y_train_pred), (y_test_true, y_test_pred))):
            
            axis.scatter(pair[0], pair[1], s=100, color='green', alpha=0.5, zorder=10)
            lims = [
            np.min([axis.get_xlim(), axis.get_ylim()]),  # min of both axes
            np.max([axis.get_xlim(), axis.get_ylim()]),  # max of both axes
            ]
            # now plot both limits against each other
            axis.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            axis.set_aspect('equal')
            axis.set_xlim(lims)
            axis.set_ylim(lims)
            axis.set_xlabel('Experimental end-of-life ' + s, fontsize=12)
            axis.set_ylabel('Predicted end-of-life ' + s, fontsize=12)

            subaxis = add_sub_axes(axis, [0.6, 0.25, 0.3, 0.2])
            res = pair[0]-pair[1]
            x_min = min(res)
            x_max = max(res)
            subaxis.hist(res, bins=15, range=(x_min, x_max), color='green', ec='black')
            subaxis.set_xlabel('Residual', fontsize=12)
            subaxis.set_ylabel('Frequency', fontsize=12)
    
    elif plot_mode == 2:

        fig, ax = plt.subplots(figsize=(5,5))
        
        ax.scatter(y_test_pred, y_test_true, s=100, color='purple', alpha=0.5, zorder=10)
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against each other
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Experimental end-of-life', fontsize=14)
        ax.set_ylabel('Predicted end-of-life', fontsize=14)
        ax.set_title(title, fontsize=20)

        subaxis = add_sub_axes(ax, [0.6, 0.15, 0.3, 0.2])
        res = y_test_true - y_test_pred
        x_min = min(res)
        x_max = max(res)
        subaxis.hist(res, bins=8, range=(x_min, x_max), color='purple', ec='black')
        subaxis.set_xlabel('Residual', fontsize=12)
        subaxis.set_ylabel('Frequency', fontsize=12)
    
    elif plot_mode == 3:

        fig, axes = plt.subplots(1, 2, figsize=(15,10))

        for axis, s, pair in zip(axes.ravel(), ('(train)', '(test)'), ((y_train_true, y_train_pred), (y_test_true, y_test_pred))):
            
            axis.scatter(pair[0], pair[1], s=100, color='purple', alpha=0.5, zorder=10)
            lims = [
            np.min([axis.get_xlim(), axis.get_ylim()]),  # min of both axes
            np.max([axis.get_xlim(), axis.get_ylim()]),  # max of both axes
            ]
            # now plot both limits against each other
            axis.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            axis.set_aspect('equal')
            axis.set_xlim(lims)
            axis.set_ylim(lims)
            axis.set_xlabel('Experimental End-of-life ' + s, fontsize=14)
            axis.set_ylabel('Predicted End-of-life ' + s, fontsize=14)

            subaxis = add_sub_axes(axis, [0.6, 0.3, 0.3, 0.2])
            res = pair[0]-pair[1]
            x_min = min(res)
            x_max = max(res)
            subaxis.hist(res, bins=15, range=(x_min, x_max), color='purple', ec='black')
            subaxis.set_xlabel('Residual', fontsize=12)
            subaxis.set_ylabel('Frequency', fontsize=12)
        
        fig = axes[0].get_figure()
        fig.tight_layout()
        fig.subplots_adjust(top=1.2)
        plt.suptitle(title, fontsize=18)

    plt.savefig(fname="plots/"+"pred_vs_true_"+fname, bbox_inches='tight')

def repeated_kfold_cross_validation(model, df, n_splits, n_repeats, feature_selection, scaling, k=None):
    '''
    A function that perfroms k-fold cross validation n_repeats times.

    Arguments:
              model:              fitted model to be validated 
              df:                 dataframe containing features and target to be used for validation
              n_splits:           number of splits to be used during validation
              n_repeats:          how many times to repeat the process of k-fold cross validation
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
        _, X, _, _ = utils_gn.univariate_feature_selection(df, k=k)
    
    # scaling option
    if scaling == True:
        X = utils_gn.scaler(X)

    # define the cv object 
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    # define metrics to be used
    metrics = {'MAE': 'neg_mean_absolute_error', 'MAPE': 'neg_mean_absolute_percentage_error', 'RMSE': 'neg_root_mean_squared_error', 'R2 score': 'r2'}

    # calculate scores
    scores = cross_validate(model, X, y, scoring=metrics, cv=cv, n_jobs=-1)
    scores = {key:(abs(val).mean(), abs(val).std()) for key, val in scores.items() if key in ['test_'+metric for metric in metrics.keys()]}
    
    return scores


def fit_linear_regression(df, scaling, params, fname, model_type='least_square', title=None, plot=False, verbose=0):
    '''
    A function that fits some linear regression models to data.

    Arguments:
              df:                 pandas dataframe that contains the features and the target
              scaling:            a boolean to specify whether to perform data scaling or not
              params:             a dictionary of parameters to pass to the regression object
              plot:               a boolean to specify whether to plot feature importance
              fname:              name to save the model
              model_type:         type of linear model to be fitted
                                  'nusvr': Nu Support Vector Regression
                                  'least_square': Ordinary Least Square Regression 
              title:              plot title
              verbose:            controls the amout of infomation printed on the screen during training
    
    Returns:
             model object and print dictionary of metrics (mae, mape, mse, rmse, r2, correlation coefficient)
              
    '''

    # get the features and the target 
    label = df['train'].columns[-1]
    X_train, y_train = df['train'].drop(label, axis=1).values, df['train'][label].values
    X_test, y_test = df['test'].drop(label, axis=1).values, df['test'][label].values
    
    # scaling option
    if scaling == True:
        sc = StandardScaler()
        sc = sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

    # fit the model and get the feature importances 
    print("linear model training has started...") 
    start_time = time.time()

    if model_type == 'nusvr':
        model = NuSVR(**params)
        
    elif model_type == 'least_square':
        model = LinearRegression(**params)

    model = model.fit(X_train, y_train)
        
    # make predictions
    y_pred_train = model.predict(X_train)  # for the train
    y_pred_test = model.predict(X_test)   # for the test

    end_time = time.time()
    print('linear model training has ended after {} seconds'.format(np.round(end_time - start_time, 2)))
    
   # calculate metrics 
    metrics = metrics_calculator(y_train, y_pred_train), metrics_calculator(y_test, y_pred_test)

    if verbose > 0:
        print('------------------')
        print('Model metrics:')
        print('------------------')
        print('Train:')
        pprint.pprint(metrics[0])
        print('Test:')
        pprint.pprint(metrics[1])
    
    if model_type == 'nusvr':
        if params['kernel']=='linear' and plot==True:
            features, feature_importance = utils_gn.feature_importance_ordering(df['train'].columns[:-1], np.abs(np.ravel(model.coef_)))
            utils_gn.feature_importance_barchart(features[:30], feature_importance[:30], 'Feature weight', fname)

    # option to plot feature importance, plot the first 30 most important features
    if plot == True:
        plot_prediction_experimental(y_train, y_pred_train, y_test, y_pred_test, fname, title)
    
    # save the model as pickle file
    with open(os.path.join("models", fname), "wb") as fp:
        pickle.dump(model, fp)
    
    return model, metrics


def fit_tree_based_regression(df, scaling, params, plot, fname, title=None, model_type='xgb', verbose=0):
    '''
    A function that fits some tree-based models to data.

    Arguments:
              df:                 pandas dataframe that contains the features and the target
              scaling:            a boolean to specify whether to perform data scaling or not
              params:             a dictionary of parameters to pass to the regression object
              plot:               a boolean to specify whether to plot feature importance
              fname:              name to save the model
              title:              plot title
              model_type:         either to fit Decsiion Trees ('decision_trees), Random Forest ('random_forest'), XGBoost ('xgb'), Extratrees ('ext') regression
              k:                  fraction of features to select if feature selection is set to true
              verbose:            controls the amout of infomation printed on the screen during training
    
    Returns:
            model object and print dictionary of metrics (mae, mape, mse, rmse, r2, correlation coefficient)
              
    '''

    # get the features and the target 
    label = df['train'].columns[-1]
    X_train, y_train = df['train'].drop(label, axis=1).values, df['train'][label].values
    X_test, y_test = df['test'].drop(label, axis=1).values, df['test'][label].values

    # scaling option
    if scaling == True:
        sc = StandardScaler()
        sc = sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
    
    # fit the model and get the feature importances 
    print("tree-based regression has started...") 
    start_time = time.time()
    
    if model_type == 'xgb':
        model = XGBRegressor(**params)
    
    elif model_type == 'ext':
        model = ExtraTreesRegressor(**params)
    
    elif model_type == 'decision_trees':
        model = DecisionTreeRegressor(**params)
    
    elif model_type == 'random_forest':
        model = RandomForestRegressor(**params)

    model = model.fit(X_train, y_train)
    
    # make predictions
    y_pred_train = model.predict(X_train)  # for the train
    y_pred_test = model.predict(X_test)   # for the test

    end_time = time.time()
    print('Tree-based regression has ended after {} seconds'.format(np.round(end_time - start_time, 2)))
    
    # calculate metrics 
    metrics = metrics_calculator(y_train, y_pred_train), metrics_calculator(y_test, y_pred_test)

    if verbose > 0:
        print('------------------')
        print('Model metrics:')
        print('------------------')
        print('Train:')
        pprint.pprint(metrics[0])
        print('Test:')
        pprint.pprint(metrics[1])
        
    
    # option to plot feature importance, plot the first 30 most important features
    if plot == True:
        features, feature_importance = utils_gn.feature_importance_ordering(df['train'].columns[:-1], model.feature_importances_)
        utils_gn.feature_importance_barchart(features[:30], feature_importance[:30], 'Feature importance', fname)
        plot_prediction_experimental(y_train, y_pred_train, y_test, y_pred_test, fname, title)
    
    # save the model as pickle file
    with open(os.path.join("models", fname), "wb") as fp:
        pickle.dump(model, fp)
    
    return model, metrics



def hyperparameter_tuning(df, estimator, param_grid, scoring, cv, scaling=True):
    '''
    A function that performs grid search over the space of parameters given.

    Arguments:
              df:                 dataframe from which train and test data will be extracted 
              estimator:          the model object
              param_grid:         a dictionary containing the parameter spaces 
              scoring:            the scoring function to use for picking the best parameters
              cv:                 number of folds to use with cross-validation process
              feature_selection:  a boolean to specify whether to carry out feature selection or not 
              k:                  fraction of features to select if feature selection is set to true

    Returns: best parameter setting, best score
    '''
    # get the features and the target 
    label = df['train'].columns[-1]
    X, y = df['train'].drop(label, axis=1).values, df['train'][label].values
    
    
    # scaling option
    if scaling == True:
        sc = StandardScaler()
        sc = sc.fit(df['train'].drop(label, axis=1).values)
        X = sc.transform(X)

    
    # define the GridSearchCV object and fit to the data 
    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv).fit(X, y)

    return gs.best_params_, gs.best_score_



def model_pipeline(df,
                algo,
                estimator,
                param_grid,
                fname,
                model_type,
                feature_selection=True,
                val_ratio=0.2,
                test_ratio=0.2,
                title=None,
                scoring='neg_mean_absolute_percentage_error',
                cv=3,
                scaling=None,
                k_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
                ):
    '''
    This function implements pipeline for optimal hyper-parameter searching, feature selection and model building.

    Args:
         df:                 data for modelling
         algo:               type of algorithm for modelling
         estimator:          model object
         param_grid:         space of parameters
         fname:              string to save the results with
         model_type:         type of model
         feature_selection:  a boolean either to perform feature selection or not
         val_ratio:          ratio of validation set
         test_ratio:         ratio of test set
         title:              plot title
         scoring:            scoring function
         cv:                 cross-validation fold size
         scaling:            a boolean either to scale features or not
         k_list:             a list containing fractions of features to be extracted from features selection process
    
    Returns: best feature selection strategy (in terms of proportion), corresponding parameters, and dataframe of metrics
    '''
    dict_of_opt_params = {}
    metric_list = []

    for k in k_list:

        # generate balanced train, validation and test data
        bl_df = utils_gn.balance_batches(df,
                                        val_ratio,
                                        test_ratio,
                                        feature_selection,
                                        k)

        # search for the best hyper-parameters
        best_param, _ = hyperparameter_tuning(df=bl_df,
                                            estimator=estimator,
                                            param_grid=param_grid,
                                            scoring=scoring,
                                            cv=cv)

        # store the best parameters in the dictionary
        dict_of_opt_params[k] = best_param

        # use the best parameters to build model 
        model, metrics = algo(df=bl_df,
                            scaling=scaling,
                            params=best_param,
                            plot=True,
                            title=title,
                            fname=fname+str(int(k*100)),
                            model_type=model_type)
        
        metric_list.append(list(metrics[0].values()) + list(metrics[1].values()))

    metric_data = pd.DataFrame(data=np.array(metric_list), columns=[data + metric for data in ('Train_', 'Test_') for metric in metrics[0].keys()], index=k_list)
    metric_data.index.name = 'Features used'

    best_k = metric_data['Test_MAPE'].idxmin()

    return best_k, dict_of_opt_params[best_k], metric_data







