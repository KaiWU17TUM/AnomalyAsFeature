import os, sys
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *
from utils.common import *

import tqdm
import pandas as pd

if __name__ == "__main__":
    data_path = os.path.join(os.path.abspath(''), '..', 'data_all')
    data_raw_path = os.path.join(data_path, 'raw')
    data_processed_path = os.path.join(data_path, 'preprocessed')

    # df_check = pickle.load(open(os.path.join(data_path, 'preprocessed', '26892283.pkl'), 'rb')

    zero_imp = {}
    for col in zero_imp_numeric:
        zero_imp[col] = zero_imp_numeric[col]
        zero_imp[col+'_mean'] = zero_imp[col]
        zero_imp[col+'_var'] = norm_param_numeric[col]['std']**2
    zero_imp = dict(zero_imp, **zero_imp_info)


    training_data_raw = []
    training_data_encoded = []

    death_num = 0
    for adm in tqdm.tqdm(adm_cohort):
        data = pickle.load(open(os.path.join(data_processed_path, f'{adm}.pkl'), 'rb'))
        info = pd.read_csv(os.path.join(data_raw_path, f'{adm}_info.csv'), header=[0])
        admittime = pd.to_datetime(info['admittime']).item()
        dischtime = pd.to_datetime(info['dischtime']).item()

        if not pd.isnull(pd.to_datetime(info['deathtime']).item()):
            mortality = True
            deathtime = pd.to_datetime(info['deathtime']).item()
            death_num += 1
            if deathtime != dischtime:
                print(f'death time != discharge time: {adm} --- {deathtime}, {dischtime}')
        else:
            mortality = False

        for i in range(data.shape[0]//48):
            data_seg = data.iloc[48*i:48*(i+1)]
            ct_min = data_seg['charttime'].min()
            ct_max = data_seg['charttime'].max()
            if ct_min < admittime or ct_max > dischtime:
                continue
            patient_data = data_seg[col_chart]
            na_col = patient_data[patient_data.columns[pd.isna(patient_data).all(axis=0)]]
            if (pd.isna(patient_data).sum().sum() / patient_data.size > .9)\
                    or (pd.isna(patient_data).all(axis=0).sum() / len(col_chart) > .5):
                continue

            sample_raw = data_seg[['LOS'] + col_info + col_chart].reset_index(drop=True)
            sample_encoded = data_seg[['LOS'] + col_info_numeric + col_text + col_numeric_encoded].reset_index(drop=True)
            sample_raw.fillna(value=zero_imp, inplace=True)
            sample_encoded.fillna(value=zero_imp, inplace=True)

            if mortality:
                sample_raw['mortality'] = 1
                sample_encoded['mortality'] = 1
            else:
                sample_raw['mortality'] = 0
                sample_encoded['mortality'] = 0

            if pd.isnull(sample_raw[col_info+col_chart_numeric]).sum().sum() > 0:
                print(adm)
            # if pd.isnull(sample_encoded[[col_info+col_chart_numeric]]).sum().sum() > 0:
            #     print(adm)

            training_data_raw.append(sample_raw)
            training_data_encoded.append(sample_encoded)




        # print(adm)
    print(f'num_samples: {len(training_data_raw)}')
    print(f'mortality samples: {death_num}')
    # pickle.dump(training_data_raw, open(os.path.join(data_path, 'training_data_raw.pkl'), 'wb'))
    # pickle.dump(training_data_encoded, open(os.path.join(data_path, 'training_data_encoded.pkl'), 'wb'))
    # pickle.dump(training_data_raw, open(os.path.join(data_path, 'training_data_raw_0imp_with_mortality.pkl'), 'wb'))
    # pickle.dump(training_data_encoded, open(os.path.join(data_path, 'training_data_encoded_0imp_with_mortality.pkl'), 'wb'))
    pickle.dump(training_data_raw, open(os.path.join(data_path, 'training_data_raw_0imp_without_mortality.pkl'), 'wb'))
    pickle.dump(training_data_encoded, open(os.path.join(data_path, 'training_data_encoded_0imp_without_mortality.pkl'), 'wb'))

