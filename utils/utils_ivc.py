import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from functools import reduce
from scipy.interpolate import interp1d
import scipy.stats
import random
from utils import utils_noah
import importlib
importlib.reload(utils_noah)

def current_voltage_relax(data_dict):
    '''
    This function takes a dict of bacthes of data, extract the current and voltage
    values corresponding to the relaxation phase of the discharging part of cycling, 
    and finally dumps the values in a pickle file.
    '''

    # create an empty dict to store values
    relax = {}

    for cell in data_dict.keys():

         # initialize dictionary for each cycle
        this_cycle = {}

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            # get the voltage and curremt values
            V_vals = data_dict[cell]['cycle_dict'][cycle]['V']
            I_vals = data_dict[cell]['cycle_dict'][cycle]['I']
            
            # get the last index of constant current/volatge discharging phase
            #_, end_V = utils_noah.tCF_index(V_vals, 'di')
            _, end_I = utils_noah.tCF_index(I_vals, 'di')

            this_cycle[cycle] = {'V':V_vals[end_I+1:], 'I':I_vals[end_I+1:]}
        
        relax[cell] = this_cycle
    
    with open(os.path.join("data", "relax.pkl"), "wb") as fp:
        pickle.dump(relax, fp)

def capacity_CCV_features(data_dict):

    capacity_CCV_dict = {}
    capacity_CCV_df = pd.DataFrame(columns=['cycle', 'ir', 'min_ccv', 'max_ccv', 'mean_ccv', 'var_ccv', 'skew_ccv', 'kurt_ccv', 'area_ccv', 'capacity'])
   
    for cell in data_dict.keys():
        stat_values = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            # get the discharge values
            i_values = utils_noah.generate_ch_di_values(data_dict, 'I', cell, cycle, 'di')
            v_values = utils_noah.generate_ch_di_values(data_dict, 'V', cell, cycle, 'di')
            t_values = utils_noah.generate_ch_di_values(data_dict, 't', cell, cycle, 'di')
            
            # get the indices of the start and end of CC
            start_I, end_I = utils_noah.tCF_index(i_values, 'di')

            # get the corresponding voltages 
            ccv = v_values[start_I:end_I+1]

            # get the corresponding time 
            cct = t_values[start_I:end_I+1]

            stats = [int(cycle), data_dict[cell]['summary']['IR'][int(cycle)-2], ccv.min(), ccv.max(),
                     ccv.mean(), ccv.var(), scipy.stats.skew(ccv), scipy.stats.kurtosis(ccv, fisher=False), np.trapz(ccv, cct),
                     data_dict[cell]['cycle_dict'][cycle]['Qd'][-1]]


            stat_values.append(stats)
            capacity_CCV_df = pd.concat([capacity_CCV_df, pd.DataFrame(data=np.array([stats]), columns=capacity_CCV_df.columns)], ignore_index=True)
        
        stat_values = np.array(stat_values)
        capacity_CCV_dict[cell] = dict(zip(capacity_CCV_df.columns, [stat_values[:, i] for i in range(len(stat_values[0]))]))
    
    with open(os.path.join("data", "capacity_CCV_dict"), "wb") as fp:
        pickle.dump(capacity_CCV_dict, fp)

    with open(os.path.join("data", "capacity_CCV_df.pkl"), "wb") as fp:
        pickle.dump(capacity_CCV_df, fp)


def CCV_features(data_dict):

    CCV_multi_features = []
    CCV_dict = {}

    for cell in data_dict.keys():
        CCV_features = []

        # initialize a dictionary to store CCV for each cycle
        this_cycle = {}

        for cycle in data_dict[cell]['cycle_dict'].keys():
            # get the discharge values
            i_values = utils_noah.generate_ch_di_values(data_dict, 'I', cell, cycle, 'di')
            v_values = utils_noah.generate_ch_di_values(data_dict, 'V', cell, cycle, 'di')
            t_values = utils_noah.generate_ch_di_values(data_dict, 't', cell, cycle, 'di')
            
            # get the indices of the start and end of CC
            start_I, end_I = utils_noah.tCF_index(i_values, 'di')

            # get the corresponding voltages 
            ccv = v_values[start_I:end_I+1]

            # get the corresponding time 
            cct = t_values[start_I:end_I+1]
            
            CCV_features.append([ccv.min(), ccv.max(), ccv.mean(), ccv.var(), scipy.stats.skew(ccv), scipy.stats.kurtosis(ccv, fisher=False),
                                np.trapz(ccv, cct)])
            this_cycle[cycle] = ccv
        
        # get the multicycle features
        #CCV_multi_features.append(reduce(np.union1d, (utils_noah.multi_cycle_features(np.array(CCV_features)[:, i]) for i in range(len(CCV_features[0])))))
        CCV_features = np.array(CCV_features)
        CCV_multi_features.append(utils_noah.multi_cycle_features(CCV_features[:,0])\
                                + utils_noah.multi_cycle_features(CCV_features[:,1])\
                                + utils_noah.multi_cycle_features(CCV_features[:,2])\
                                + utils_noah.multi_cycle_features(CCV_features[:,3])\
                                + utils_noah.multi_cycle_features(CCV_features[:,4])\
                                + utils_noah.multi_cycle_features(CCV_features[:,5])\
                                + utils_noah.multi_cycle_features(CCV_features[:,6])
        )

        CCV_dict[cell] = this_cycle
    
    CCV_df = pd.DataFrame(data=np.array(CCV_multi_features),
                          columns=[ft + item for ft in ('min_', 'max_', 'mean_', 'var_', 'skew_', 'kurt_', 'area_') for item in utils_noah.strings_multi_cycfeatures()],
                          index=data_dict.keys())
    
    # add the end of life of cells
    CCV_df['end_of_life'] = utils_noah.cycle_life(data_dict)['cycle_life'].values
    
    # dump the files in pickle
    with open(os.path.join("data", "CCV_dict.pkl"), "wb") as fp:
        pickle.dump(CCV_dict, fp)

    with open(os.path.join("data", "CCV_df.pkl"), "wb") as fp:
        pickle.dump(CCV_df, fp)


def plot_CCV_features(data_dict, ylabel=None, ylim=None, sample_cells=None, option=1):
    
    if option == 1:
        # get the cells belonging to the same batch
        b1 = [cell for cell in data_dict.keys() if cell[:2]=='b1']
        b2 = [cell for cell in data_dict.keys() if cell[:2]=='b2']
        b3 = [cell for cell in data_dict.keys() if cell[:2]=='b3']

        x_labels = dict(zip(data_dict['b1c0'].keys(), ['Cycles', r'Internal resistance ($\Omega$)', 'Min of CCV (V)', 'Max of CCV (V)',
                                                    'Mean of CCV (V)', 'Variance of CCV (V)', 'Skewness of CCV', 'Kurtosis of CCV',
                                                    'Area under CC Voltage Curve','Capacity (Ah)']))
        
        for batch in [b1, b2, b3]:
            fig, ax = plt.subplots(3, 3, figsize=(20, 15))
            i = 0
            for feature in data_dict['b1c0'].keys():
                if feature not in [ylabel]:
                    for cell in batch:
                        ax[i//3, i%3].plot(data_dict[cell][ylabel], data_dict[cell][feature], 'o', linewidth=1, markersize=2)
                        ax[i//3, i%3].set_ylabel(x_labels[feature], fontsize=14)
                        ax[i//3, i%3].set_xlabel(x_labels[ylabel], fontsize=14)
                        ax[i//3, i%3].set_ylim(ylim)
                    i += 1
            i = 0

            #handles, labels = ax.get_legend_handles_labels()
            #fig.legend(handles, labels)

            plt.show()

    elif option == 2:

        for i, cell in enumerate(sample_cells):

            fig, ax = plt.subplots(figsize=(6, 6))
            cycles = [int(cyc) for cyc in data_dict[cell].keys()]
            cmap = plt.get_cmap('viridis', len(cycles))

            for cycle in data_dict[cell].keys():
                ax.plot(data_dict[cell][cycle], c=cmap(int(cycle)), linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_ylabel('CC Voltage (V)', fontsize=14)
            ax.set_title(cell, fontsize=16)

            ## Normalizer
            norm = mpl.colors.Normalize(vmin=1, vmax=len(cycles))
            
            # creating ScalarMappable
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            cbar = fig.colorbar(sm, orientation="horizontal")
            cbar.set_label('Cycles', fontsize=14)

            plt.show()




