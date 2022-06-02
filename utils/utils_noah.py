from operator import index
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
from utils import utils_gn
import importlib
importlib.reload(utils_gn)
import scipy.stats
from scipy.interpolate import splev, splrep, interp1d
from scipy.signal import find_peaks

def strings_multi_cycfeatures():
    return ('y_0', 'y_50', 'y_100', 'y_100m0', 'y_diff')

def multi_cycle_features(feature_values_list):

    # y_0, y_50, y_100, y_100m0, y_diff as defined in Noah et. al paper
    y_0 = np.median(feature_values_list[:10])
    y_50 = np.median(feature_values_list[44:54])
    y_100 = np.median(feature_values_list[90:100])
    y_100m0 = y_100 - y_0
    y_diff = (y_100 - y_50) - (y_50 - y_0)

    return [y_0, y_50, y_100, y_100m0, y_diff]

def generate_ch_di_values(data_dict, col_name, cell, cycle, option):

    # Here, I saw an outlier in b1c2 at cycle 2176, so I think this 
    # measurement is in seconds and thus divide it by 60
    if cell=='b1c2' and cycle=='2176':
        summary_charge_time = data_dict[cell]['summary']['chargetime'][int(cycle) - 2] / 60
    else:
        summary_charge_time = data_dict[cell]['summary']['chargetime'][int(cycle) - 2]

    values = data_dict[cell]['cycle_dict'][cycle][col_name]

    if option=='ch':
        return np.array(values[data_dict[cell]['cycle_dict'][cycle]['t'] - summary_charge_time <= 1e-10])
    if option=='di':
        return np.array(values[data_dict[cell]['cycle_dict'][cycle]['t'] - summary_charge_time > 1e-10])


def tCF_index(feature, option):
    '''
    This function generates indices corresponding to the start and the end of constant values of a given feature.

    Arguments:
             feature:     a list of considered feature, e.g current, voltage
             option:      a string to provide option for charge ('ch') and discharge ('di') indices 
    '''

    i = 1
    constant_feature_list  = []
    constant_feature_index  = []

    while i < len(feature):
        if abs(feature[i-1] - feature[i]) <= 1e-2:
            constant_feature_list.append(feature[i-1])
            constant_feature_index.append(i-1)
        i+=1
    
    opt_list = []

    if option=='ch':
        det_value = np.max(constant_feature_list)
        opt_list = [index for index, element in zip(constant_feature_index, constant_feature_list) if np.round(det_value - element, 2) <= 1e-2]

        return opt_list[0], opt_list[-1]

    if option=='di':
        det_value = np.min(constant_feature_list)
        opt_list = [index for index, element in zip(constant_feature_index, constant_feature_list) if np.round(element - det_value, 2) <= 1e-2]
    
        return opt_list[0], opt_list[-1]

def get_peak_area_dQdV(I_list, Q_list, V_list, option, plot=False):
    '''
    A function that calculates the maximum/minimum peak/valley for the charging/discharging phases of the dQdV vs V curve.
    It is noticed that the curve predominantly shows a peak for charging phase but valley for discharging phase.
    '''
    # get the indices corresponding to the beginning and end of constant current
    init_index, last_index = tCF_index(I_list, option)

    # get the Q and V corresponding to the constant current phase
    const_Q, const_V = Q_list[init_index:last_index+1], V_list[init_index:last_index+1]

    # remove the repeated values in V, use resulting indices to sort Q
    const_V, sorted_index = np.unique(const_V, return_index=True)
    const_Q = const_Q[sorted_index]

    
    # get the dQdV values
    dQdV, corres_V = [], []

    for i in range(1, len(const_Q)):

        if abs(const_V[i] - const_V[i-1]) <= 1e-6:
            continue

        dQdV.append((const_Q[i] - const_Q[i-1]) / (const_V[i] - const_V[i-1]))
        corres_V.append(const_V[i-1])
    
    dQdV, corres_V = np.array(dQdV), np.array(corres_V)
    #dQdV = splev(corres_V, splrep(corres_V, dQdV, s=10, k=2))
    

    if plot == True:
        plt.plot(corres_V, dQdV, 'r-', alpha=0.2, label='data')
        plt.xlabel(r'Voltage $(V)$')
        plt.ylabel(r'$dQ/dV \; (Ah V^{-1})$')
        plt.legend()
        plt.show()
    
    # find the peaks/valleys
    if option == 'ch':
        peak, _ = find_peaks(dQdV, prominence=0.1)
        
        if len(peak) > 0:
            return corres_V[peak][np.argmax(dQdV[peak])], dQdV[peak].max(), np.trapz(corres_V, dQdV)
        else:
            return np.nan, np.nan, np.nan 

        
    if option == 'di':
        peak, _ = find_peaks(-1*dQdV, prominence=0.1)   # here peaks is interpreted to be valleys

        if len(peak) > 0:
            return corres_V[peak][np.argmin(dQdV[peak])], dQdV[peak].min(), np.trapz(corres_V, dQdV)
        else:
          return np.nan, np.nan, np.nan 



def get_peak_area_dVdQ(I_list, Q_list, V_list, option, plot=False):
    '''
    A function that calculates the peak and area of the dVdQ vs Q curve for both charging and discharging phase. 
    It is noticed that the curve predominantly has peak but not significant valleys.
    '''
    init_index, last_index = tCF_index(I_list, option)

    # get the Q and V corresponding to the constant current phase
    const_Q, const_V = Q_list[init_index:last_index+1], V_list[init_index:last_index+1]
    
    # remove the repeated values in Q, use resulting indices to sort V
    const_Q, sorted_index = np.unique(const_Q, return_index=True)
    const_V = const_V[sorted_index]

    # get the dVdQ values
    #dVdQ = np.gradient(const_V, const_Q)
    dVdQ, corres_Q = [], []

    for i in range(1, len(const_Q)):

        if abs(const_Q[i] - const_Q[i-1]) <= 1e-6:
            continue

        dVdQ.append((const_V[i] - const_V[i-1]) / (const_Q[i] - const_Q[i-1]))
        corres_Q.append(const_Q[i-1])
    
    dVdQ, corres_Q = np.array(dVdQ), np.array(corres_Q)


    if plot == True:
        plt.plot(const_Q, dVdQ, 'b-', linewidth=2, label='smooth')
        plt.xlabel(r'Capacity $(Ah)$')
        plt.ylabel(r'$dV/dQ \; (VAh^{-1})$')
        plt.legend()
        plt.show()
    
    # find the peaks
    peak, _ = find_peaks(dVdQ, prominence=0.1)

    if len(peak) > 0:
        return const_Q[peak][np.argmax(dVdQ[peak])], dVdQ.max(), np.trapz(corres_Q, dVdQ)
    else:
        return np.nan, np.nan, np.nan


def cycle_life(data_dict) :
    cycle_life = [data_dict[cell]['summary']['cycle'][-1] for cell in data_dict.keys()]

    return pd.DataFrame(data=cycle_life, columns=['cycle_life'], index=data_dict.keys())


def Imed_state(data_dict):
    '''
    This function generates features for Imed, median current
    '''
    cells_in_the_batch = data_dict.keys()

    Imed_multi_cycle_values = []

    for cell in cells_in_the_batch:

        Imed_ch_for_each_cycle = []
        Imed_di_for_each_cycle = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            Imed_ch_for_each_cycle.append(np.median(abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'ch'))))
            Imed_di_for_each_cycle.append(np.median(abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'di'))))

        Imed_multi_cycle_values.append(multi_cycle_features(Imed_ch_for_each_cycle) + multi_cycle_features(Imed_di_for_each_cycle))
    
    return pd.DataFrame(data=np.array(Imed_multi_cycle_values),
                        columns=['Imed_ch_'+ item for item in strings_multi_cycfeatures()] + ['Imed_di_'+ item for item in strings_multi_cycfeatures()],
                        index=cells_in_the_batch)

def Vavg_state(data_dict):
    '''
    This function generates features for Vavg, average voltage.
    '''
    cells_in_the_batch = data_dict.keys()


    Vavg_multi_cycle_values = []

    for cell in cells_in_the_batch:

        Vavg_ch_for_each_cycle = []
        Vavg_di_for_each_cycle = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            Vavg_ch_for_each_cycle.append(np.mean(generate_ch_di_values(data_dict, 'V', cell, cycle, 'ch')))
            Vavg_di_for_each_cycle.append(np.mean(generate_ch_di_values(data_dict, 'V', cell, cycle, 'di')))

        Vavg_multi_cycle_values.append(multi_cycle_features(Vavg_ch_for_each_cycle) + multi_cycle_features(Vavg_di_for_each_cycle))
    
    return pd.DataFrame(data=np.array(Vavg_multi_cycle_values),
                        columns=['Vavg_ch_'+ item for item in strings_multi_cycfeatures()] + ['Vavg_di_'+ item for item in strings_multi_cycfeatures()],
                        index=cells_in_the_batch)

def Q_state(data_dict):
    '''
    This function generates features for Q_sate, cummulative capacity for both
    charge and discharge columns.
    '''
    cells_in_the_batch = data_dict.keys()

    Q_multi_cycle_values = []

    for cell in cells_in_the_batch:

        Qc_for_each_cycle = []
        Qd_for_each_cycle = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            Qc_for_each_cycle.append(np.sum(data_dict[cell]['cycle_dict'][cycle]['Qc']))   # maybe you might need to reconsider
            Qd_for_each_cycle.append(np.sum(data_dict[cell]['cycle_dict'][cycle]['Qd']))   # maybe you might need to reconsider

        Q_multi_cycle_values.append(multi_cycle_features(Qc_for_each_cycle) + multi_cycle_features(Qd_for_each_cycle))
    
    return pd.DataFrame(data=np.array(Q_multi_cycle_values),
                        columns=['Qc_'+ item for item in strings_multi_cycfeatures()] + ['Qd_'+ item for item in strings_multi_cycfeatures()],
                        index=cells_in_the_batch)

def E_state(data_dict):
    '''
    This function generates features for E_sate, cummulative energy.
    I expressed E as the product of current, voltage and time (= IVt)
    '''
    cells_in_the_batch = data_dict.keys()

    E_multi_cycle_values = []

    for cell in cells_in_the_batch:

        Ec_for_each_cycle = []
        Ed_for_each_cycle = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            Ec_for_each_cycle.append(np.trapz(
                abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'ch')) * generate_ch_di_values(data_dict, 'V', cell, cycle, 'ch'), generate_ch_di_values(data_dict, 't', cell, cycle, 'ch')*60))

            Ed_for_each_cycle.append(np.trapz(
                abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'di')) * generate_ch_di_values(data_dict, 'V', cell, cycle, 'di'), generate_ch_di_values(data_dict, 't', cell, cycle, 'di')*60))

        E_multi_cycle_values.append(multi_cycle_features(Ec_for_each_cycle) + multi_cycle_features(Ed_for_each_cycle))
    
    return  pd.DataFrame(data=np.array(E_multi_cycle_values),
                         columns=['E_ch_'+ item for item in strings_multi_cycfeatures()] + ['E_di_'+ item for item in strings_multi_cycfeatures()],
                         index=cells_in_the_batch)

def Qeff_state(data_dict):
    '''
    This function generates features for Qeff, coulombic efficiency.
    I expressed Qeff as the ratio of the total charge 
    extracted from the battery to the total charge put into the battery
    over a full cycle: QDischarge / QCharge.
    Here, I used the summary data since the definition says over a full cycle.
    '''
    cells_in_the_batch = data_dict.keys()

    Qeff_multi_values = []

    for cell in cells_in_the_batch:
        Qeff_multi_values.append(multi_cycle_features(data_dict[cell]['summary']['QDischarge'] / data_dict[cell]['summary']['QCharge']))
    
    return pd.DataFrame(data=np.array(Qeff_multi_values),
                        columns=['Qeff_'+ item for item in strings_multi_cycfeatures()],
                        index=cells_in_the_batch)

def Eeff_state(data_dict):
    '''
    This function generates features for Eeff, Energy efficiency.
    I expressed Eeff as the ratio of the total energy 
    extracted from the battery to the total energy put into the battery
    over a full cycle: output energy / input energy.
    '''
    cells_in_the_batch = data_dict.keys()

    Eeff_multi_values = []

    for cell in cells_in_the_batch:

        efficiency_for_cell = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
     
            efficiency_for_cell.append(
                (np.trapz(abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'di')) * generate_ch_di_values(data_dict, 'V', cell, cycle, 'di'), generate_ch_di_values(data_dict, 't', cell, cycle, 'di')))/ \
                (np.trapz(abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'ch')) * generate_ch_di_values(data_dict, 'V', cell, cycle, 'ch'), generate_ch_di_values(data_dict, 't', cell, cycle, 'ch'))))

    
        Eeff_multi_values.append(multi_cycle_features(efficiency_for_cell))
    
    return pd.DataFrame(data=np.array(Eeff_multi_values),
                        columns=['Eeff_'+ item for item in strings_multi_cycfeatures()],
                        index=cells_in_the_batch)

def Inorm_state(data_dict):
    '''
    This function generates median current normalized by lifetime median current
    '''
    cells_in_the_batch = data_dict.keys()

    Inorm_multi_cycle_values = []

    for cell in cells_in_the_batch:

        Inorm_ch_for_each_cycle = []
        Inorm_di_for_each_cycle = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            Inorm_ch_for_each_cycle.append(np.median(abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'ch'))) / np.median(abs(data_dict[cell]['cycle_dict'][cycle]['I'])))
            Inorm_di_for_each_cycle.append(np.median(abs(generate_ch_di_values(data_dict, 'I', cell, cycle, 'di'))) / np.median(abs(data_dict[cell]['cycle_dict'][cycle]['I'])))

        Inorm_multi_cycle_values.append(multi_cycle_features(Inorm_ch_for_each_cycle) + multi_cycle_features(Inorm_di_for_each_cycle))
    
    return pd.DataFrame(data=np.array(Inorm_multi_cycle_values),
                        columns=['Inorm_ch_'+ item for item in strings_multi_cycfeatures()] + ['Inorm_di_'+ item for item in strings_multi_cycfeatures()],
                        index=cells_in_the_batch)


def SOH(data_dict):
    '''
    This function calculates SOH, state of health, expressed as capacity as a ratio of the initial capacity
    '''
    cells_in_the_batch = data_dict.keys()

    SOH_multi_cycle_values = []

    for cell in cells_in_the_batch:
        SOH_multi_cycle_values.append(multi_cycle_features(data_dict[cell]['summary']['QDischarge'] / data_dict[cell]['summary']['QDischarge'][0]))
    
    return pd.DataFrame(data=np.array(SOH_multi_cycle_values), columns=['SOH_'+ item for item in strings_multi_cycfeatures()], index=cells_in_the_batch)


def t_state(data_dict):
    '''
    This function creates features from charge and discharge times
    '''
    t_ch = []
    t_di = []

    for cell in data_dict.keys():
        t_ch.append(multi_cycle_features(data_dict[cell]['summary']['chargetime']))
        t_di.append(
            multi_cycle_features(
            [generate_ch_di_values(data_dict, 't', cell, cycle, 'di')[-1] for cycle in data_dict[cell]['cycle_dict'].keys()]))
    
    return pd.DataFrame(data=np.array([tl1 + tl2 for tl1, tl2 in zip(t_ch, t_di)]),
                        columns=['t_ch_'+ item for item in strings_multi_cycfeatures()]+['t_di_'+ item for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())


def tCC_state(data_dict):
    '''
    This function generates features for tCC, constant current charge or discharge time
    '''

    tCC_ch_di_multi_values = []

    for cell in data_dict.keys():

        tCC_ch = []
        tCC_di = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            cycle_current = data_dict[cell]['cycle_dict'][cycle]['I']
            CC_ch_index, CC_di_index = tCF_index(cycle_current, 'ch'), tCF_index(cycle_current, 'di')
            
            tCC_ch.append(data_dict[cell]['cycle_dict'][cycle]['t'][CC_ch_index[1]] - data_dict[cell]['cycle_dict'][cycle]['t'][CC_ch_index[0]])
            tCC_di.append(data_dict[cell]['cycle_dict'][cycle]['t'][CC_di_index[1]] - data_dict[cell]['cycle_dict'][cycle]['t'][CC_di_index[0]])
        
        tCC_ch_di_multi_values.append(multi_cycle_features(tCC_ch) + multi_cycle_features(tCC_di))

    return pd.DataFrame(data=np.array(tCC_ch_di_multi_values),
                        columns=['tCC_ch_'+ item for item in strings_multi_cycfeatures()]+['tCC_di_'+ item for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def tCV_state(data_dict):
    '''
    This function generates features for tCV, constant voltage charge or discharge time
    '''

    tCV_ch_di_multi_values = []

    for cell in data_dict.keys():

        tCV_ch = []
        tCV_di = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            cycle_voltage = data_dict[cell]['cycle_dict'][cycle]['V']
            CV_ch_index, CV_di_index = tCF_index(cycle_voltage, 'ch'), tCF_index(cycle_voltage, 'di')
            
            tCV_ch.append(data_dict[cell]['cycle_dict'][cycle]['t'][CV_ch_index[1]] - data_dict[cell]['cycle_dict'][cycle]['t'][CV_ch_index[0]])
            tCV_di.append(data_dict[cell]['cycle_dict'][cycle]['t'][CV_di_index[1]] - data_dict[cell]['cycle_dict'][cycle]['t'][CV_di_index[0]])
        
        tCV_ch_di_multi_values.append(multi_cycle_features(tCV_ch) + multi_cycle_features(tCV_di))

    return pd.DataFrame(data=np.array(tCV_ch_di_multi_values),
                        columns=[ftname + ext for ftname in ('tCV_ch_', 'tCV_di_') for ext in strings_multi_cycfeatures()],
                        index=data_dict.keys())


def tCCvsCVfrac_state(data_dict):
    '''
    This function generates features for tCCvsCVfrac, tCC divided by total time
    '''

    tCCvsCVfrac_multi_values = []

    for cell in data_dict.keys():

        tCCvsCVfrac_ch = []
        tCCvsCVfrac_di = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            cycle_current = data_dict[cell]['cycle_dict'][cycle]['I']
            CC_ch_index, CC_di_index = tCF_index(cycle_current, 'ch'), tCF_index(cycle_current, 'di')
            
            # get the discharge time list, to use for calculating the total time of discharge process
            discharge_time_list  = generate_ch_di_values(data_dict, 't', cell, cycle, 'di')
            
            tCCvsCVfrac_ch.append((data_dict[cell]['cycle_dict'][cycle]['t'][CC_ch_index[1]] - data_dict[cell]['cycle_dict'][cycle]['t'][CC_ch_index[0]]) \
                / generate_ch_di_values(data_dict, 't', cell, cycle, 'ch')[-1])

            tCCvsCVfrac_di.append((data_dict[cell]['cycle_dict'][cycle]['t'][CC_di_index[1]] - data_dict[cell]['cycle_dict'][cycle]['t'][CC_di_index[0]]) \
                / (discharge_time_list[-1] - discharge_time_list[0]))
        
        tCCvsCVfrac_multi_values.append(multi_cycle_features(tCCvsCVfrac_ch) + multi_cycle_features(tCCvsCVfrac_di))

    return pd.DataFrame(data=np.array(tCCvsCVfrac_multi_values),
                        columns=[ftname + ext for ftname in ('tCCvsCVfrac_ch_', 'tCCvsCVfrac_di_') for ext in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def TIEVC_state(data_dict):
    '''
    This function generates the features corresponding to TEIVC, time interval during equal voltage change
    '''

    TIEVC_multi_values = []

    for cell in data_dict.keys():

        TIEVC_ch = []
        TIEVC_di = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            charging_voltage = generate_ch_di_values(data_dict, 'V', cell, cycle, 'ch')
            discharging_voltage = generate_ch_di_values(data_dict, 'V', cell, cycle, 'di')

            bool_ch = (charging_voltage >= 3.4).tolist() and (charging_voltage <= 3.6).tolist()
            bool_di = (discharging_voltage >= 3.0).tolist() and (discharging_voltage <= 3.2).tolist()

            chargetimes_in_the_interval = generate_ch_di_values(data_dict, 't', cell, cycle, 'ch')[bool_ch]
            dischargetimes_in_the_interval = generate_ch_di_values(data_dict, 't', cell, cycle, 'di')[bool_di]

            TIEVC_ch.append(chargetimes_in_the_interval[-1] - chargetimes_in_the_interval[0])
            TIEVC_di.append(dischargetimes_in_the_interval[-1] - dischargetimes_in_the_interval[0])
        
        TIEVC_multi_values.append((multi_cycle_features(TIEVC_ch) + multi_cycle_features(TIEVC_di)))
    
    return pd.DataFrame(data=np.array(TIEVC_multi_values),
                        columns=[ftname + ext for ftname in ('TIEVC_ch_', 'TIEVC_di_') for ext in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def VstartVend(data_dict):

    '''
    This function calculates features for the Vstart and Vend, the voltages at the start and end of segment
    '''

    VstartVend_multi_values = []

    for cell in data_dict.keys():

        Vstart_ch_list = []
        Vend_ch_list = []

        Vstart_di_list = []
        Vend_di_list = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            charge_values = generate_ch_di_values(data_dict, 'V', cell, cycle, 'ch')
            Vstart_ch_list.append(charge_values[0])
            Vend_ch_list.append(charge_values[-1])

            discharge_values = generate_ch_di_values(data_dict, 'V', cell, cycle, 'di')
            Vstart_di_list.append(discharge_values[0])
            Vend_di_list.append(discharge_values[-1])

        VstartVend_multi_values.append(multi_cycle_features(Vstart_ch_list) + multi_cycle_features(Vstart_di_list) \
                                      + multi_cycle_features(Vend_ch_list) + multi_cycle_features(Vend_di_list))
    
    return pd.DataFrame(data=np.array(VstartVend_multi_values), 
                        columns=[ftname + ext for ftname in ('Vstart_ch_', 'Vstart_di_', 'Vend_ch_', 'Vend_di_') for ext in strings_multi_cycfeatures()],
                        index=data_dict.keys())


def dVdtStartEnd(data_dict):
    '''
    This function generates the  features for dVdt at the start and end of segment
    '''
    dVdtStartEnd_multi_values = []

    for cell in data_dict.keys():

        dVdtStart_ch_list = []
        dVdtEnd_ch_list = []

        dVdtStart_di_list = []
        dVdtEnd_di_list = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            charge_values_V = generate_ch_di_values(data_dict, 'V', cell, cycle, 'ch')
            charge_values_t = generate_ch_di_values(data_dict, 't', cell, cycle, 'ch')
            dVdtStart_ch_list.append((charge_values_V[1] - charge_values_V[0]) / (60*(charge_values_t[1] - charge_values_t[0])))
            dVdtEnd_ch_list.append((charge_values_V[-1] - charge_values_V[-2]) / (60*(charge_values_t[-1] - charge_values_t[-2])))

            discharge_values_V = generate_ch_di_values(data_dict, 'V', cell, cycle, 'di')
            discharge_values_t = generate_ch_di_values(data_dict, 't', cell, cycle, 'di')
            dVdtStart_di_list.append((discharge_values_V[1] - discharge_values_V[0]) / (60*(discharge_values_t[1] - discharge_values_t[0])))
            dVdtEnd_di_list.append((discharge_values_V[-1] - discharge_values_V[-2]) / (60*(discharge_values_t[-1] - discharge_values_t[-2])))

        dVdtStartEnd_multi_values.append(multi_cycle_features(dVdtStart_ch_list) + multi_cycle_features(dVdtEnd_ch_list) + multi_cycle_features(dVdtStart_di_list) + multi_cycle_features(dVdtEnd_di_list))
    
    return pd.DataFrame(data=np.array(dVdtStartEnd_multi_values), 
                        columns=[ftname + ext for ftname in ('dVdtStart_ch_', 'dVdtEnd_ch_', 'dVdtStart_di_', 'dVdtEnd_di_') for ext in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def VoltsAt80pctSOCcharge(data_dict):
    '''
    This function generates features for voltage at 80 percent SOC for charging phase
    '''

    VoltsAt80pct_multi_values = []

    for cell in data_dict.keys():
        
        VoltsAt80pct_list = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            V_list = data_dict[cell]['cycle_dict'][cycle]['V']
            t_list = data_dict[cell]['cycle_dict'][cycle]['t']

            VoltsAt80pct_list.append(V_list[t_list - 13.3 <= 1e-10][-1])
        
        VoltsAt80pct_multi_values.append(multi_cycle_features(VoltsAt80pct_list))
    
    return pd.DataFrame(data=np.array(VoltsAt80pct_multi_values), 
                        columns=['VoltsAt80pctSOC_ch_' + item for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def dVdtAt80pctSOC(data_dict):
    '''
    This function generates features for the derivative of V wrt t at 80% SOC
    '''

    dVdtat80pctSOC_multi_values = []

    for cell in data_dict.keys():

        dVdtat80pctSOC_list = []

        for cycle in data_dict[cell]['cycle_dict'].keys():

            V_list = data_dict[cell]['cycle_dict'][cycle]['V']
            t_list = data_dict[cell]['cycle_dict'][cycle]['t']

            index_of_80pctV = len(V_list[t_list - 13.3 <= 1e-10] - 1)

            dVdtat80pctSOC_list.append((V_list[index_of_80pctV + 1] - V_list[index_of_80pctV]) / (60*(t_list[index_of_80pctV + 1] - t_list[index_of_80pctV])))

        dVdtat80pctSOC_multi_values.append(multi_cycle_features(dVdtat80pctSOC_list))
    
    return pd.DataFrame(data=np.array(dVdtat80pctSOC_multi_values), 
                        columns=['dVdtat80pctSOC_ch_' + item for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def VCETchargetime(data_dict):
    '''
    This function generates features for voltage change between time t=0 and chargetime. This is somehow equivalent to the features generated for 
    VCET<time>s_<state> in the Noah et. al paper (which depicts voltage change during equal time between t=0 and a specified time). The specified 
    time here is the chargetime. 
    '''

    VCETchargetime_multi_values = []

    for cell in data_dict.keys():

        VCETchargetime_list = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            # to handle the outlier in b1c2 and cycle 2176
            if cell=='b1c2' and cycle=='2176':
                chargetime_index = np.where(data_dict[cell]['cycle_dict'][cycle]['t'] - (data_dict[cell]['summary']['chargetime'][int(cycle)-2]/60) <= 1e-10)[0][-1]
            else:
                chargetime_index = np.where(data_dict[cell]['cycle_dict'][cycle]['t'] - (data_dict[cell]['summary']['chargetime'][int(cycle)-2]) <= 1e-10)[0][-1]

            VCETchargetime_list.append(data_dict[cell]['cycle_dict'][cycle]['V'][chargetime_index] - data_dict[cell]['cycle_dict'][cycle]['V'][0])
        
        VCETchargetime_multi_values.append(multi_cycle_features(VCETchargetime_list))
    
    return pd.DataFrame(data=np.array(VCETchargetime_multi_values), 
                        columns=['VCETchargetime_' + item for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())

def dQdVfeatures(data_dict):
    '''
    This function generates dQdV features for 

        dQdVpeak_maxloc_ch:   the location of the maximum value of the dQdV vs V curve (charge)
        dQdVpeak_maxmag_ch:   the amplitude of the maximum value of the dQdV vs V curve (charge)
        dQdVpeak_maxarea_ch:  the area under the dQdV vs V curve (charge)

        dQdVvalley_minloc_di:   the location of the minimum value of the dQdV vs V curve (discharge)
        dQdVvalley_minmag_di:   the amplitude of the minimum value of the dQdV vs V curve (discharge)
        dQdVvalley_minarea_di:  the area under the dQdV vs V curve (discharge)
    '''
    dQdV_multi_values = []

    for cell in data_dict.keys():

        dQdVpeak_ch_values = []
        dQdVvalley_di_values = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            dQdVpeak_ch_values.append(get_peak_area_dQdV(data_dict[cell]['cycle_dict'][cycle]['I'],
                                                         data_dict[cell]['cycle_dict'][cycle]['Qc'],
                                                         data_dict[cell]['cycle_dict'][cycle]['V'], 'ch'))

            dQdVvalley_di_values.append(get_peak_area_dQdV(data_dict[cell]['cycle_dict'][cycle]['I'],
                                                           data_dict[cell]['cycle_dict'][cycle]['Qd'],
                                                           data_dict[cell]['cycle_dict'][cycle]['V'], 'di'))

        dQdVpeak_ch_values, dQdVvalley_di_values = np.array(dQdVpeak_ch_values), np.array(dQdVvalley_di_values)
        dQdV_multi_values.append(multi_cycle_features(dQdVpeak_ch_values[:,0]) + multi_cycle_features(dQdVpeak_ch_values[:,1]) + multi_cycle_features(dQdVpeak_ch_values[:,2])\
                                 + multi_cycle_features(dQdVvalley_di_values[:,0]) + multi_cycle_features(dQdVvalley_di_values[:,1]) + multi_cycle_features(dQdVvalley_di_values[:,2]))
        


    return pd.DataFrame(data=np.array(dQdV_multi_values), 
                        columns=[ft + item for ft in ('dQdVpeak_maxloc_ch_', 'dQdVpeak_maxmag_ch_', 'dQdVpeak_maxarea_ch_', 'dQdVvalley_minloc_di_', 'dQdVvalley_minmag_di_', 'dQdVvalley_minarea_di_') \
                                              for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())


def dVdQfeatures(data_dict):
    '''
    This function generates dVdQ features for 

        dVdQpeak_maxloc_ch:   the location of the maximum value of the dVdQ vs Q curve (charge)
        dQdVpeak_maxmag_ch:   the amplitude of the maximum value of the dVdQ vs Q curve (charge)
        dQdVpeak_maxarea_ch:  the area under the dVdQ vs Q curve (charge)

        dVdQpeak_maxloc_di:   the location of the maximum value of the dVdQ vs Q curve (discharge)
        dQdVpeak_maxmag_di:   the amplitude of the maximum value of the dVdQ vs Q curve (discharge)
        dQdVpeak_maxarea_di:  the area under the dVdQ vs Q curve (discharge)
    '''
    dVdQ_multi_values = []

    for cell in data_dict.keys():

        dVdQpeak_ch_values = []
        dVdQpeak_di_values = []

        for cycle in data_dict[cell]['cycle_dict'].keys():
            
            dVdQpeak_ch_values.append(get_peak_area_dVdQ(data_dict[cell]['cycle_dict'][cycle]['I'],
                                                         data_dict[cell]['cycle_dict'][cycle]['Qc'],
                                                         data_dict[cell]['cycle_dict'][cycle]['V'], 'ch'))

            dVdQpeak_di_values.append(get_peak_area_dVdQ(data_dict[cell]['cycle_dict'][cycle]['I'],
                                                           data_dict[cell]['cycle_dict'][cycle]['Qd'],
                                                           data_dict[cell]['cycle_dict'][cycle]['V'], 'di'))

        dVdQpeak_ch_values, dVdQpeak_di_values = np.array(dVdQpeak_ch_values), np.array(dVdQpeak_di_values)
        dVdQ_multi_values.append(multi_cycle_features(dVdQpeak_ch_values[:,0]) + multi_cycle_features(dVdQpeak_ch_values[:,1]) + multi_cycle_features(dVdQpeak_ch_values[:,2])\
                                 + multi_cycle_features(dVdQpeak_di_values[:,0]) + multi_cycle_features(dVdQpeak_di_values[:,1]) + multi_cycle_features(dVdQpeak_di_values[:,2]))
        


    return pd.DataFrame(data=np.array(dVdQ_multi_values), 
                        columns=[ft + item for ft in ('dVdQpeak_maxloc_ch_', 'dVdQpeak_maxmag_ch_', 'dVdQpeak_maxarea_ch_', 'dVdQpeak_maxloc_di_', 'dVdQpeak_maxmag_di_', 'dVdQpeak_maxarea_di_') \
                                              for item in strings_multi_cycfeatures()],
                        index=data_dict.keys())
    
def dSOHdCycCyc(data_dict):

    '''
    This function calculates change in SOH with respect to cycle for given cycles in the range 1 to 100.
    '''

    dSOHdCyc_values = []

    for cell in data_dict.keys():

        SOH_values = data_dict[cell]['summary']['QDischarge'][:100] / data_dict[cell]['summary']['QDischarge'][0]

        SOH_interp = interp1d(np.linspace(1, 100, 100), SOH_values)
        SOH_interp_values = SOH_interp(np.linspace(1, 100, 1000))
        dSOHdCyc = np.gradient(SOH_interp_values, np.linspace(1, 100, 1000))

        index = int(0.1 * len(dSOHdCyc))
        dSOHdCyc_values.append([np.median(dSOHdCyc[:index+1]), np.median(dSOHdCyc[-index:])])
    

    return pd.DataFrame(data=np.array(dSOHdCyc_values), 
                        columns=['dSOHdCycCyc1', 'dSOHdCycCyc100'],
                        index=data_dict.keys())


def getNoahData(data_dict, fname, dropna=False, fillna=True):

    generated_df = Imed_state(data_dict)

    list_of_generators = [Vavg_state, Q_state, E_state,  Qeff_state,
                          Eeff_state, Inorm_state, SOH, t_state, tCC_state, tCV_state, tCCvsCVfrac_state,
                          TIEVC_state, VstartVend, dVdtStartEnd, VoltsAt80pctSOCcharge, dVdtAt80pctSOC, 
                          VCETchargetime, dQdVfeatures, dVdQfeatures, dSOHdCycCyc, cycle_life]

    df_remainder = [gen(data_dict) for gen in list_of_generators]
    generated_df = generated_df.join(df_remainder)
    
    if dropna==True:
        # drop columns that contain nan values
        generated_df.dropna(axis=1, inplace=True)
    
    if fillna==True:
        for col in generated_df.columns[generated_df.isna().any()]:
            generated_df.fillna(value=generated_df[col].mean(), inplace=True)
    
    with open(os.path.join("data", fname), "wb") as fp:
        pickle.dump(generated_df, fp)
    

















        



