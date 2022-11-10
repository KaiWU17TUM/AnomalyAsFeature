import matplotlib.pyplot as plt
from matplotlib import gridspec

import pandas as pd
import os

from utils.common import *




def plot_data(gs, data, data_item):
    for iid in data_item:

        ax = plt.subplot(gs)
        if item_dict[item_dict['itemid']==iid]['category'].item() == 'Labs':
            ax.scatter(data[data['itemid']==iid]['charttime'], data[data['itemid']==iid]['valuenum'],
                     label=data_item[iid])

        elif item_dict[item_dict['itemid']==iid]['category'].item() == 'Treatments':
            for d in data[data['itemid']==iid].iterrows():
                    ax.axvline(d[1]['charttime'], color='y')


        elif item_dict[item_dict['itemid']==iid]['category'].item() == 'Pain/Sedation':
            for d in data[data['itemid']==iid].iterrows():
                if d[1]['value'] == 'Yes':
                    ax.axvline(d[1]['charttime'], color='r')

        else:
            if item_dict[item_dict['itemid']==iid]['param_type'].item()=='Text':
                continue
            ax.plot(data[data['itemid']==iid]['charttime'], data[data['itemid']==iid]['valuenum'],
                     label=data_item[iid])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_vital_and_ref(gs, adm, vital_source, df_chart):
    # plot HR or BP data with reference orthostatic measurements
    ax = plt.subplot(gs)
    plt.xticks(rotation=30)
    data = df_chart[df_chart['hadm_id']==adm]
    colors_ref = ['dodgerblue', 'cornflowerblue', 'lightsteelblue']
    colors = ['maroon', 'firebrick', 'lightcoral']
    if vital_source == 'HR':
        iids = [220045] # HR
        iids_ref = [223764, 223765, 224647] # HR lying, HR sitting, HR standing
        ref_labels = ['HR lying', 'HR sitting', 'HR standing']
        for i, iid in enumerate(iids):
            ct = pd.to_datetime(data[data['itemid']==iid]['charttime'])
            val = data[data['itemid']==iid]['valuenum']
            ax.plot(
                ct,
                val,
                c=colors[i],
                label='HR',
            )
        for i, iid in enumerate(iids_ref):
            ct_last = pd.to_datetime(data['charttime'].max())
            val_last = data[data['itemid']==iid].sort_values('charttime')['valuenum'].iloc[-1]
            ct = pd.to_datetime(data[data['itemid']==iid]['charttime']).to_list() + [ct_last]
            val = data[data['itemid']==iid]['valuenum'].to_list() + [val_last]
            ax.plot(
                ct,
                val,
                drawstyle='steps-post',
                c=colors_ref[i],
                linestyle='dashed',
                label=ref_labels[i],
            )
    elif vital_source == 'NBP':
        iids = [220179, 220180, 220181] # NBPs, NBPd, NBPm
        iids_ref = [224645, 224646, 223766, # BPs lying, BPs sitting, BPs standing
                    226092, 226094, 226096] # BPd lying, BPd sitting, BPd standing
        data_labels = ['BPs', 'BPd', 'BPm']
        ref_labels = ['BPs lying', 'BPs sitting', 'BPs standing',
                      'BPd lying', 'BPd sitting', 'BPd standing',]
        for i, iid in enumerate(iids):
            ct = pd.to_datetime(data[data['itemid']==iid]['charttime'])
            val = data[data['itemid']==iid]['valuenum']
            ax.plot(
                ct,
                val,
                c=colors[i],
                label=data_labels[i],
            )
        for i, iid in enumerate(iids_ref):
            ct_last = pd.to_datetime(data['charttime'].max())
            val_last = data[data['itemid']==iid].sort_values('charttime')['valuenum'].iloc[-1]
            ct = pd.to_datetime(data[data['itemid']==iid]['charttime']).to_list() + [ct_last]
            val = data[data['itemid']==iid]['valuenum'].to_list() + [val_last]
            ax.plot(
                ct,
                val,
                drawstyle='steps-post',
                c=colors_ref[i%3],
                linestyle='dashed',
                label=ref_labels[i],
            )
    else:
        raise AssertionError('Only support vital references for "HR" and "NBP".')