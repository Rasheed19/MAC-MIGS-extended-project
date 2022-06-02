import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
import scipy.stats
from utils import utils_gn
import importlib
importlib.reload(utils_gn)
import scipy.stats


def generate_severson_features(data_dict, fname):
    '''
    A function that generates Severson et. al features from a given 
    batche(s) of battery data.

    Mainly generate 7 features (with additional 5 without the log):
            * Log mean of Delta Q_{100-10}(V)
            * Log variance of Delta Q_{100-10}(V)
            * Log minimum of  Delta Q_{100-10}(V)
            * Log skewness of Delta Q_{100-10}(V)
            * Log kurtosis of Delta Q_{100-10}(V)
            * Discharge capacity, cycle 2
            * Difference between max discharge capacity and cycle 2
    Additional 5:
            * mean of Delta Q_{100-10}(V)
            * variance of Delta Q_{100-10}(V)
            * minimum of  Delta Q_{100-10}(V)
            * skewness of Delta Q_{100-10}(V)
            * kurtosis of Delta Q_{100-10}(V)

    Argument: 
              data_dict:        A dictionary of batche(s) of data
              fname:            name to which the generated data is saved
    '''

    cells_in_the_batch = data_dict.keys()

    generated_df = pd.DataFrame(index=cells_in_the_batch)
    
    Q_100_10_values = []
    diff_maxqd_qd2 = []

    for cell in cells_in_the_batch:
        # for the log values
        Q_100_10_values.append(data_dict[cell]['cycle_dict']['100']['Qdlin'] - data_dict[cell]['cycle_dict']['10']['Qdlin'])
        
        # for the difference between max discharge capacity and that of cycle 2, from summary data
        diff_maxqd_qd2.append(max(data_dict[cell]['summary']['QDischarge']) - data_dict[cell]['summary']['QDischarge'][1])   
    
    # also consider features before appling log
    considered_delta_features = [
                'mean_Q_100_10', 'log_mean_Q_100_10',
                'var_Q_100_10', 'log_var_Q_100_10',
                'min_Q_100_10', 'log_min_Q_100_10',
                'skew_Q_100_10', 'log_skew_Q_100_10',
                'kurt_Q_100_10', 'log_kurt_Q_100_10'
        ]
    functions_to_be_applied = [
                lambda x : np.mean(x, axis=1), lambda x : np.log10(abs(np.mean(x, axis=1))),
                lambda x : np.var(x, axis=1), lambda x : np.log10(abs(np.var(x, axis=1))),
                lambda x : np.min(x, axis=1), lambda x : np.log10(abs(np.min(x, axis=1))),
                lambda x : scipy.stats.skew(x, axis=1), lambda x : np.log10(abs(scipy.stats.skew(x, axis=1))),
                lambda x : scipy.stats.kurtosis(x, axis=1, fisher=False), lambda x : np.log10(abs(scipy.stats.kurtosis(x, axis=1, fisher=False)))
        ]

    for delta_feature, function in zip(considered_delta_features, functions_to_be_applied):
            generated_df[delta_feature] = function(Q_100_10_values)
    
    # other features 
    # discharge capacity at cycle 2, this is from summary data
    generated_df['Qd_cycle2'] = [data_dict[cell]['summary']['QDischarge'][1] for cell in cells_in_the_batch]

    # difference between max discharge capacity and that of cycle 2
    generated_df['diff_maxqd_qd2'] = diff_maxqd_qd2

    # finally add the cycle life
    generated_df['cycle_life'] = [data_dict[cell]['summary']['cycle'][-1] for cell in data_dict.keys()]

    with open(os.path.join("data", fname), "wb") as fp:
        pickle.dump(generated_df, fp)





    








