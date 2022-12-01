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

    training_data_raw = []
    training_data_encoded = []

    for adm in tqdm.tqdm(adm_cohort):
        data = pickle.load(open(os.path.join(data_processed_path, f'{adm}.pkl'), 'rb'))
        info = pd.read_csv(os.path.join(data_raw_path, f'{adm}_info.csv'), header=[0])
        admittime = pd.to_datetime(info['admittime']).item()
        dischtime = pd.to_datetime(info['dischtime']).item()

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

            sample_raw = data_seg[['LOS'] + col_info + col_chart].reset_index()
            sample_encoded = data_seg[['LOS'] + col_info_numeric + col_text + col_numeric_encoded].reset_index()
            training_data_raw.append(sample_raw)
            training_data_encoded.append(sample_encoded)

        print(adm)
    print(f'num_samples: {len(training_data_raw)}')
    pickle.dump(training_data_raw, open(os.path.join(data_path, 'training_data_raw.pkl'), 'wb'))
    pickle.dump(training_data_encoded, open(os.path.join(data_path, 'training_data_encoded.pkl'), 'wb'))

