import pandas as pd
import numpy as np
import os 
import pickle
import json
import h5py
from scipy.stats import iqr
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SelectFromModel, SequentialFeatureSelector, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
from matplotlib import cm
from utils import utils_models, utils_noah
from datetime import datetime
import importlib
importlib.reload(utils_models)
importlib.reload(utils_noah)


def time_monitor(initial_time=None):
	''' 
    This function monitors time from the start of a process to the end of the process
    '''
	if not initial_time:
		initial_time = datetime.now()
		return initial_time
	elif initial_time:
		thour, temp_sec = divmod((datetime.now() - initial_time).total_seconds(), 3600)
		tmin, tsec = divmod(temp_sec, 60)

		return '%ih %imin and %ss.' % (thour, tmin, round(tsec, 2))


def load_data(filename, batch_num, num_cycles=None):
    ''' 
    This function loads the downloaded matlab file into a dictionary
    
    Args: 
        filename:     string with the path of the data file
        batch_number: index of this batch
        num_cycles:   number of cycles to be loaded
    
    Returns a dictionary with data for each cell in the batch
    '''

    # read the matlab file
    f = h5py.File(filename, 'r')
    batch = f['batch']

    # get the number of cells in this batch
    num_cells = batch['summary'].shape[0]

    # initialize a dictionary to store the result
    batch_dict = {}

    summary_features = ["IR", "QCharge", "QDischarge", "Tavg", "Tmin", 
                     "Tmax", "chargetime", "cycle"]
    cycle_features = ["I", "Qc", "Qd", "Qdlin", "T", "Tdlin", "V", "discharge_dQdV", "t"]

    for i in range(num_cells):

        # decide how many cycles will be loaded
        if num_cycles is None:
            loaded_cycles = f[batch['cycles'][i,0]]['I'].shape[0]
        else:
            loaded_cycles = min(num_cycles, f[batch['cycles'][i,0]]['I'].shape[0])

        if i % 10 == 0:
            print("* {} cells loaded ({} cycles)".format(i, loaded_cycles))

        # initialise a dictionary for this cell
        cell_dict = {}

        # for the data in cycle life    
        if batch_num != 3:
            cell_dict["cycle_life"] = f[batch["cycle_life"][i,0]][()]
        else:
            # There seems to be an offset for batch 3
            cell_dict["cycle_life"] = f[batch["cycle_life"][i,0]][()] + 1

        # for the data in policy
        cell_dict["charge_policy"] = f[batch["policy_readable"][i,0]][()].tobytes()[::2].decode()

        # for the data in summary
        cell_dict["summary"] = {}
        for feature in summary_features:
            cell_dict["summary"][feature] = np.hstack(f[batch['summary'][i,0]][feature][0,:].tolist())

        # for the cycle data
        cell_dict["cycle_dict"] = {}

        for j in range(loaded_cycles):
            cell_dict["cycle_dict"][str(j+1)] = {}
            for feature in cycle_features:
                cell_dict["cycle_dict"][str(j+1)][feature] = np.hstack( (f[f[batch['cycles'][i,0]][feature][j,0]][()]))


        # converge into the batch dictionary
        batch_dict['b{}c{}'.format(batch_num, i)] = cell_dict

    return batch_dict

def load_and_save_dict_data(num_cycles=None, option=1):
    '''
    This function load and save downloaded matlab files as picke files

    Args:
         num_cycles:  number of cycles to load
         option:      1: to load all batches in one pickle file, 2: to load each batch and save it in a pickle file separately
    '''

    # paths for data file with each batch of cells
    mat_filenames = {
        "batch1": os.path.join(".","data","2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
        "batch2": os.path.join(".","data","2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
        "batch3": os.path.join(".","data","2018-04-12_batchdata_updated_struct_errorcorrect.mat")}

    batches = ["batch1", "batch2", "batch3"]

    start = time_monitor()
    print("Loading batch 1 data...")
    batch1 = load_data(mat_filenames["batch1"], 1, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 2 data...")
    batch2 = load_data(mat_filenames["batch2"], 2, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 3 data...")
    batch3 = load_data(mat_filenames["batch3"], 3, num_cycles=num_cycles)
    print(time_monitor(start))

    print("* {} cells loaded in batch 1".format(len(batch1.keys())))
    print("* {} cells loaded in batch 2".format(len(batch2.keys())))
    print("* {} cells loaded in batch 3".format(len(batch3.keys())))

    # there are four cells from batch1 that carried into batch2, we'll remove the data from batch2 and put it with the correct cell from batch1
    b2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    b1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]

    # append data to batch 1
    for i, bk in enumerate(b1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack(
                    (batch1[bk]['summary'][j], 
                    batch2[b2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack(
                    (batch1[bk]['summary'][j], 
                    batch2[b2_keys[i]]['summary'][j]))
                
        last_cycle = len(batch1[bk]['cycle_dict'].keys())

        # useful when all cycles loaded
        if num_cycles is None:
            for j, jk in enumerate(batch2[b2_keys[i]]['cycle_dict'].keys()):
                batch1[bk]['cycle_dict'][str(last_cycle + j)] = batch2[b2_keys[i]]['cycle_dict'][jk]
    '''
    The authors exclude cells that:
        * do not reach 80% capacity (batch 1)
        * were carried into batch2 but belonged to batch 1 (batch 2)
        * noisy channels (batch 3)
    '''

    exc_cells = {}
    exc_cells["batch1"] = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
    exc_cells["batch2"] = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    exc_cells["batch3"] = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c38", "b3c39"]

    for c in exc_cells["batch1"]:
        del batch1[c]
        
    for c in exc_cells["batch2"]:
        del batch2[c]
        
    for c in exc_cells["batch3"]:
        del batch3[c]
    
    # exclude the first cycle from all cells because this data was not part of the first batch of cells
    for batch in [batch1, batch2, batch3]:
       for cell in batch.keys():
          del batch[cell]['cycle_dict']['1']
        
    for batch in [batch1, batch2, batch3]:
        for cell in batch.keys():
            assert '1' not in batch[cell]['cycle_dict'].keys()
            
    for batch in [batch1, batch2, batch3]:
        for cell in batch.keys():
            for feat in batch[cell]['summary'].keys():
                batch[cell]['summary'][feat] = np.delete(batch[cell]['summary'][feat], 0)
    
    if num_cycles is None:
        filename_suffix = 'all.pkl'
    else:
        filename_suffix = str(num_cycles) + 'cycles.pkl'

    
    if option == 1:

        # combine all 3 batches in one dictionary
        data_dict = {**batch1, **batch2, **batch3}
        
        # save the dict as a picle file
        with open(os.path.join("data", "data_" + filename_suffix), "wb") as fp:
            pickle.dump(data_dict, fp)
    
    elif option == 2:
        # save the batch separately
        for i, batch in zip(('1', '2', '3'), (batch1, batch2, batch3)):
            with open(os.path.join("data", "batch" + i + '_' + filename_suffix), "wb") as fp:
                pickle.dump(batch, fp)
    
    
def read_data(fname, folder="data"):
    # Load pickle data
    with open(os.path.join(folder, fname), "rb") as fp:
        df = pickle.load(fp)
    
    return df

def generate_per_cycle_df(batch_data, 
                         cell_identifier,
                         drop_cycle=True
                        ):
    '''
    Create pandas summary (per cycle) dataframe for a given cell in a batch
    '''
    df = pd.DataFrame(batch_data[cell_identifier]['summary'])

    if drop_cycle == True:
        df.drop('cycle', axis=1, inplace=True)
    
    return df

def generate_within_cycle_df(batch_data,
                             cell_identifier,
                             cycle_number):
    '''
    Create pandas cycle (within cycle) dataframe for a given cell in a batch with known measurements
    '''
    columns_needed = [col for col in batch_data[cell_identifier]['cycle_dict'][cycle_number].keys() if col not in ['Qdlin', 'Tdlin', 'discharge_dQdV']]
    dict_needed = {key: batch_data[cell_identifier]['cycle_dict'][cycle_number][key] for key in columns_needed}
    df = pd.DataFrame(dict_needed)

    return df

def gen_percycle_stat_features(batch_data):

    operation_values_list = []
    cell_operation_values = []
    features_considered = ['initial_', 'final_', 'mean_', 'range_', 'intq_range_', 'max_', 'min_']

    cells_in_the_batch = list(batch_data.keys())
    columns_in_the_summary = batch_data[cells_in_the_batch[0]]['summary'].keys()

    list_of_operations = [lambda x: x.values[0], lambda x: x.values[-1], np.mean, np.ptp, iqr, np.max, np.min]

    for cell in cells_in_the_batch:
        df_cell = generate_per_cycle_df(batch_data, cell, drop_cycle=False)

        for operation in list_of_operations:
            operation_values = (df_cell.apply(operation, axis=0)).values.tolist()
            operation_values_list.append(operation_values)
        
        cell_operation_values.append(np.array(operation_values_list).flatten())

        operation_values_list = []
    
    generated_features = [feature + column for feature in features_considered for column in columns_in_the_summary]
    
    generated_features_df = pd.DataFrame(data=cell_operation_values,
                                        columns=generated_features,
                                        index=cells_in_the_batch)

    return  generated_features_df

def plot_variables_for_patterns(batch_data,
                                 independent_variable,
                                 dependent_variable):

    cells_in_the_batch = list(batch_data.keys())
    
    fig, ax = plt.subplots(21, 2, figsize=(15, 100))
    fig_locations = [(i,j) for i in range(21) for j in range(2)]

    for cell, fig_location in zip(cells_in_the_batch, fig_locations):
        df_cell = generate_per_cycle_df(batch_data, cell, drop_cycle=False)

        ax[fig_location].set_title(cell + ': ' + dependent_variable + ' vs ' + independent_variable)
        df_cell.plot(x=independent_variable, y=dependent_variable, kind='scatter', ax=ax[fig_location])
        ax[fig_location].grid('On')

    fig.tight_layout(pad=0.5) # adds space between figures
    plt.show()

def gen_percycle_rate_features(batch_data,
                               model_func_types,
                               dependent_variable,
                               independent_variables,
                               initial_guesses):

    cells_in_the_batch = list(batch_data.keys())
    rate_list_for_a_cell = []
    rate_list_for_all_cells = []

    for cell in cells_in_the_batch:
        df_cell = generate_per_cycle_df(batch_data, cell, drop_cycle=False)

        for independent_variable, model_func, initial_guess in zip(independent_variables, model_func_types, initial_guesses):
            optimized_parameters = utils_models.curve_fitting(model_func, 
                                                            initial_guess,
                                                            df_cell[independent_variable],
                                                            df_cell[dependent_variable],
                                                            plot=False)
            rate_list_for_a_cell.append(optimized_parameters[1])
        
        rate_list_for_all_cells.append(rate_list_for_a_cell)
        rate_list_for_a_cell = []

    generated_features = ['rate' + '_' + dependent_variable + '_' + independent_variable for independent_variable in independent_variables]
    generated_df = pd.DataFrame(data=rate_list_for_all_cells, columns=generated_features, index=cells_in_the_batch)

    return generated_df


def scaler(X):
    '''
    A function that performs standard scaling of an input data.

    Argument:
             X:  the data to be scaled
    Returns: 
            scaled data
    '''
    scaler = StandardScaler()

    return scaler.fit_transform(X)


def feature_importance_ordering(list_of_features, list_of_importance):
    '''
    A function that orders features from highest importance to the least impotant.
    '''
    
    # normalize the list of importance
    list_of_importance = list_of_importance/ list_of_importance.max()

    # get the indices of the sorted impotance 
    sorted_index = np.argsort(list_of_importance)

    # get the corresponding ordering 
    sorted_list_of_features = list_of_features[sorted_index]
    sorted_list_of_importance = list_of_importance[sorted_index]

    return sorted_list_of_features[::-1], sorted_list_of_importance[::-1]
    


def variance_threshold(df, num_of_features_after_ordering, var_threshold=0.01):

    '''
    Implements the variance threshold feature selection.

    Arguments: 
              df:                              the dataframe in consideration
              num_of_features_after_ordering:  number of features you want after ordering  
              var_threshold:                   the threshold of the variance to keep

    Returns:
            the reduced feature values, names and their corresponding variances 
    '''
    
    # get the features in the dataframe
    features_val = df.drop(df.columns[-1], axis=1).values
    
    vt = VarianceThreshold(threshold=var_threshold)
    vt = vt.fit(features_val)

    # ge the indices of the selected features
    #support = vt.get_support()
    
    # transform X 
    feature_val_reduced = vt.transform(features_val)

    print("{} features removed from a total of {}".format(features_val.shape[1] - feature_val_reduced.shape[1], features_val.shape[1]))

    # order the features from highest to least important
    ordered_features, ordered_importance = feature_importance_ordering(df.columns[:-1], vt.variances_)

    return feature_val_reduced, ordered_features[:num_of_features_after_ordering], ordered_importance[:num_of_features_after_ordering]


def univariate_feature_selection(df, num_of_features_after_ordering=100, k=0.8):

    '''
    Implements the univariate feature selectuion
    Arguments:
              df:                              the dataframe under consideration
              num_of_features_after_ordering:  number of features you want after ordering
              k:                               percentage of features to be selected  
    
    Returns:
            the reduced feature values, names and their corresponding scores
    '''

    # get the features and the target
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

    # define and fit the univariate feature selection object
    ufs = SelectKBest(score_func=f_regression, k=int(k*X.shape[1])).fit(X, y)

    # get the support 
    #support = ufs.get_support()

    # transform X 
    X_reduced = ufs.transform(X)

    print("{} features removed from a total of {}".format(X.shape[1] - X_reduced.shape[1], X.shape[1]))

    # order the features from highest to least important
    ordered_features, ordered_importance = feature_importance_ordering(df.columns[:-1], ufs.scores_)

    return X_reduced, ordered_features[:num_of_features_after_ordering], ordered_importance[:num_of_features_after_ordering]


def recursive_feature_selection(df, num_of_features_after_ordering, n_features_to_select=None):

    '''
    This function performs the recursive feeature selection using the Random Forest Regression as estimator.
    Arguments:
              df:                              dataframe containing the features and the target
              num_of_features_after_ordering:  number of features you want after ordering
              n_features_to_select:            the number of features to be selected from the dataframe (this could be intger or float in the range 0 and 1)
    
    Returns:
            the reduced feature values, names and their corresponding rankings.
    '''

    # define the features and the target 
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

    # create the Random Forest Regessor object
    rfr = RandomForestRegressor()

    # create the RFE object and fit to the data 
    rfe = RFE(rfr, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(X,y)

    # get the support of the retained features 
    #support = rfe.support_

    # transform X 
    X_reduced = rfe.transform(X)

    print("{} features removed from a total of {}".format(X.shape[1] - X_reduced.shape[1], X.shape[1]))

    # return the transformed X, the retained feature names and their corresponding rankings 
    return X_reduced, df.columns[:-1][:num_of_features_after_ordering], rfe.ranking_[:num_of_features_after_ordering]


def select_from_model(df,  num_of_features_after_ordering, alpha=10, threshold=None):
    '''
    This function performs feature selection using SelectFromModel from scikit-learn. It makes use of the Lasso regression for the estimator.
    Arguments:
             df:                              dataframe containing the features and the target
             num_of_features_after_ordering:  number of features you want after ordering
             alpha:                           regularization parameter to be used with the Lasso regression
             threshold:                       threshold to be used with the SelectFromModel object. This can be a string ('mean' or 'median') or a float that gives limit on 
                                              the feature importances 
    Returns:
            the reduced feature values, names and their corresponding importances (in terms of the optimized weights of the Lasso regression)
    '''

    # get the features and target lists
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values
    
    # create Lasso regression object 
    ls = Lasso(alpha=alpha)

    # create SelectFromModel object and fit it to the data
    sfm = SelectFromModel(ls, threshold=threshold).fit(X,y)

    # get the support for the retained features 
    #support = sfm.get_support()

    # transform X 
    X_reduced = sfm.transform(X)

    print("{} features removed from a total of {}".format(X.shape[1] - X_reduced.shape[1], X.shape[1])) 
    
    # order the features from highest to least important
    ordered_features, ordered_importance = feature_importance_ordering(df.columns[:-1], np.abs(sfm.estimator_.coef_))

    # return the reatined feature values, names and their corresponding importances 
    return X_reduced, ordered_features[:num_of_features_after_ordering], ordered_importance[:num_of_features_after_ordering]


def sequential_feature_selection(df, num_of_features_after_ordering, n_features_to_select=None):
    '''
    This function performs sequential feature selection with forward option and Random Forest Regressor as estimator.
    Arguments:
              df:                              dataframe in consideration
              num_of_features_after_ordering:  number of features you want after ordering
              n_features_to_select:            number/fraction of features to select, this could be 'warn' to select half of the features,
                                               int to select a definite number of features or float (a number between 0 and 1) to select 
                                               fraction of the features 
     Returns:
            the reduced feature values, names and their corresponding importances.  
    '''

    # get the features and the target from the dataframe
    X, y = df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

    # define the Random Forest Regressor
    rfr = RandomForestRegressor().fit(X,y)

    # define the Sequential Feature Selector
    sfs = SequentialFeatureSelector(rfr, n_features_to_select=n_features_to_select).fit(X,y)

    # get the support 
    #support = sfs.get_support()

    # transform X 
    X_reduced = sfs.transform(X)

    print("{} features removed from a total of {}".format(X.shape[1] - X_reduced.shape[1], X.shape[1]))

    # order the features from highest to least important
    ordered_features, ordered_importance = feature_importance_ordering(df.columns[:-1], sfs.estimator.feature_importances_)

    return X_reduced, ordered_features[:num_of_features_after_ordering], ordered_importance[:num_of_features_after_ordering]


def plot_feature_importance_heatmap(list_of_dataframes, feature_selection_method, num_of_features_after_ordering, string_of_dataframes, label, fname):

    '''
    A function that plots the heatmap of feature importances for a given feature selection method.
    Argument:
             list_of_dataframes:             list of dataframes to be considered 
             feature_selection_method:       feature selection method to be applied   
             num_of_features_after_ordering: number of features you want after the ordering 
             string_of_dataframes:           a list of strings of the names of the dataframes
             label:                          label for the color map
             fname:                          name to save the plot
             figsize:                        a tuple for the figure size
    '''
    
    # create a list of dictionaries: key=feature names, values=importance
    list_of_data_results = []
    for df in list_of_dataframes:
        results = feature_selection_method(df, num_of_features_after_ordering)
        list_of_data_results.append(dict(zip(results[1], results[2])))

    
    intersect_of_features = reduce(np.intersect1d, (list(list_of_data_results[i].keys()) for i in range(len(list_of_data_results))))

    list_of_all_feature_values = []

    for feature in intersect_of_features:
        list_of_feature_values = []

        for dictn in list_of_data_results:
            list_of_feature_values.append(dictn[feature])
    
        list_of_all_feature_values.append(list_of_feature_values)

    fig, ax = plt.subplots(figsize=(len(intersect_of_features)/2, 2))
    ax.set_xticklabels(intersect_of_features, fontsize=12)
    ax.set_yticklabels(string_of_dataframes, fontsize=12)
    sns.heatmap(np.array(list_of_all_feature_values).T, vmin=0, vmax=1, xticklabels=intersect_of_features, yticklabels=string_of_dataframes, cmap='viridis', cbar_kws={'label': label}, ax=ax)
    plt.yticks(rotation=0)

    plt.savefig(fname="plots/"+fname, bbox_inches='tight')
    plt.show()

def feature_importance_barchart(features, importance, importance_tag, fname, title=None):

    df = pd.DataFrame()
    df['Features'] = features 
    df[importance_tag] = importance

    ax = df.plot(x='Features', y=importance_tag, kind='bar', figsize=[15, 2], fontsize=11, legend=False, color='green', ec='black', xlabel='')
    
    '''
    for p in ax.patches:
        ax.annotate(str(np.round(p.get_height(),decimals=2)), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', 
                    va='center', xytext=(0, 10), textcoords='offset points')
    '''
    
    ax.set_ylabel(importance_tag, fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.savefig(fname="plots/"+"importance_"+fname, bbox_inches='tight')
    plt.show()

def dict_of_colours(data_dict):
    '''
    This function returns a dictionry of colors which correspond to the EOL of cells
    '''

    # get the eol of cells and normalize it
    eol = utils_noah.cycle_life(data_dict)['cycle_life']
    eol = (eol-eol.min()) / (eol.max() - eol.min())

    # define the colour map and map it to the normalized eol
    cmap = cm.get_cmap('viridis')
    colours = cmap(eol)

    return dict(zip(data_dict.keys(), colours))

def in_cycle_data_exploration(data_dict, sample_cycle, fname):
    '''
    This function visualizes in-cycle data for a given cycle

    Args:
         data_dict:     dictionary of data
         sample_cycle:  given cycle
         fname:         string to save the plot with
    '''

    # get the  dictionary of colours 
    colour_dict = dict_of_colours(data_dict)

    # create a dictionary of full feature names and units 
    name_unit_dict = {'I': r'Current ($A$)', 'Qc': r'Charge capacity ($Ah$)', 'Qd': r'Discharge capacity ($Ah$)',
                      'T': r'Temperature ($^{\circ}C$)', 'V': r'Voltage (V)', 'Qdlin': r'Interpolated capacity ($Ah$)',
                      'Tdlin': r'Interpolated temperature ($^{\circ}C$)', 'discharge_dQdV': r'dQ/dV ($AhV^{-1}$)'}
    
    fig, ax = plt.subplots(4, 2, figsize=(12, 12))

    i = 0
    for feature in name_unit_dict.keys():
        if feature not in ['Qdlin', 'Tdlin', 'discharge_dQdV']:
            for cell in data_dict.keys():
                ax[i//2, i%2].plot(data_dict[cell]['cycle_dict'][sample_cycle]['t'], data_dict[cell]['cycle_dict'][sample_cycle][feature], color=colour_dict[cell])
            
            ax[i//2, i%2].set_xlabel('Time (min)', fontsize=12)
            ax[i//2, i%2].set_ylabel(name_unit_dict[feature], fontsize=12)

            i += 1

    for feature in ['Qdlin', 'Tdlin', 'discharge_dQdV']:
        for cell in data_dict.keys():
            ax[i//2, i%2].plot(data_dict[cell]['cycle_dict'][sample_cycle][feature], color=colour_dict[cell])   
        
        ax[i//2, i%2].set_xlabel('Index', fontsize=12)
        ax[i//2, i%2].set_ylabel(name_unit_dict[feature], fontsize=12)

        i += 1

    fig.tight_layout(pad=0.5)
    plt.savefig(fname="plots/"+fname, bbox_inches='tight')
    plt.show()

def individual_summary_data_exploration(data_dict, fname):
    '''
    This function visualizes summary data 

    Args:
         data_dict:     dictionary of data
         fname:         string to save the plot with
    '''


    # get the  dictionary of colours 
    colour_dict = dict_of_colours(data_dict)

    # create a dictionary of full feature names and units 
    name_unit_dict = {'IR': r'Internal resistance ($\Omega$)', 'QDischarge': r'Discharge capacity ($Ah$)',
                      'Tavg': r'Average temperature ($^{\circ}C$)', 'Tmin': r'Minimum temperature ($^{\circ}C$)',
                      'Tmax': r'Maximum temperature ($^{\circ}C$)', 'chargetime': r'Charge time (min)'}
    
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    i = 0
    for feature in name_unit_dict.keys():
        if feature not in ['QCharge', 'cycle']:
            for cell in data_dict.keys():
                ax[i//2, i%2].plot(data_dict[cell]['summary'][feature][:500], 'o', linewidth=1, markersize=2, color=colour_dict[cell])
            
            if feature == 'QDischarge':
                ax[i//2, i%2].set_ylim([0.8, 1.1])

            elif feature == 'chargetime':
                ax[i//2, i%2].set_ylim([8.0, 14])

            ax[i//2, i%2].set_xlabel('Cycles', fontsize=12)
            ax[i//2, i%2].set_ylabel(name_unit_dict[feature], fontsize=12)

            i += 1
    
    fig.tight_layout(pad=0.5)
    plt.savefig(fname="plots/"+fname, bbox_inches='tight')
    plt.show()


def plot_eol_cell_frequency():

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    batch_names = ['Batch 1', 'Batch 2', 'Batch 3']
    cell_frequency = [len(read_data(fname).keys()) for fname in ['batch1_100cycles.pkl', 'batch2_100cycles.pkl', 'batch3_100cycles.pkl']]

    ax[0].barh(batch_names, cell_frequency, color='green', ec='black')
    ax[0].set_xlabel('Frequency', fontsize=12)

    ax[1].hist(utils_noah.cycle_life(read_data('data_100cycles.pkl'))['cycle_life'], color='green', ec='black')
    ax[1].set_xlabel('EOL of cells', fontsize=12)
    ax[1].set_ylabel('Frequency', fontsize=12)

    fig.tight_layout(pad=0.5)
    plt.savefig(fname="plots/"+"eol_cell_frequency", bbox_inches='tight')
    plt.show()

    

    



