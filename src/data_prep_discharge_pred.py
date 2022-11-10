import os, sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *
from utils.common import *

import pandas as pd
np.random.seed(0)

if __name__ == '__main__':
    # load data
    df_info, df_chart, vital_ref_dict, selected_items = load_data()
    df_info['admittime'] = pd.to_datetime(df_info['admittime'])
    df_info['dischtime'] = pd.to_datetime(df_info['dischtime'])
    item_vital = selected_items['vital']
    item_resp = selected_items['resp']
    item_lab = selected_items['lab']
    item_treat = selected_items['treatment']
    item_pain = selected_items['pain']

    item_selected = pd.concat((item_dict[(item_dict['itemid'].isin(item_vital)) & (item_dict['param_type']=='Numeric')],
                               item_dict[(item_dict['itemid'].isin(item_resp)) & (item_dict['param_type']=='Numeric')]))
    item_selected = item_selected.drop(item_selected.index[-1])
    vital_cols = item_selected['abbreviation'].tolist()

    df_info_norm, norm_param_info = normalize_df(df_info, type='info')
    df_chart_norm, norm_param_chart = normalize_df(df_chart, type='chart',
                                                   selected_itemids=item_selected['itemid'].values, item_dict=item_dict)

    columns = ['hadm_id', 'charttime'] +\
              ['age', 'admission_weight', 'admission_type', 'admission_location'] +\
              ['Orthostatic HR lying', 'Orthostatic HR sitting', 'Orthostatic HR standing'] +\
              ['Orthostatic BPs lying', 'Orthostatic BPs sitting', 'Orthostatic BPs standing'] +\
              ['Orthostatic BPd lying', 'Orthostatic BPd sitting', 'Orthostatic BPd standing'] + vital_cols + ['LOS']
    df_model = pd.DataFrame(columns=columns)

    for adm in df_info['hadm_id'].unique():
        ct_disch = df_info.loc[df_info['hadm_id']==adm, 'dischtime'].item()

        info = df_info_norm[df_info_norm['hadm_id']==adm].copy()
        ref = df_chart_norm[(df_chart_norm['hadm_id']==adm) & (df_chart_norm['itemid'].isin(ref_itemids))].copy()

        data = df_chart_norm[(df_chart_norm['hadm_id']==adm) & (df_chart_norm['itemid'].isin(item_selected['itemid']))].copy()
        ct_start = data['charttime'].min()
        ct_start = ct_start.replace(hour=ct_start.hour, minute=0, second=0, microsecond=0)
        ct_end = data['charttime'].max()
        data = interpolate_charttime_df(data, freq='1H', method='fillin', begin=ct_start, end=ct_end)

        df_ = pd.DataFrame(columns=columns)
        df_['charttime'] = pd.date_range(ct_start, ct_end, freq='1H')
        los = ct_disch - df_['charttime']
        df_['LOS'] = [dt.days*24 + dt.seconds//3600 for dt in los]
        df_['hadm_id'] = adm
        for iid, col in item_selected[['itemid', 'abbreviation']].to_numpy():
            d = data.loc[data['itemid']==iid, 'valuenum'].values
            df_.loc[:, col] = d

        for iid, col in vital_ref_dict[['itemid', 'abbreviation']].to_numpy():
            ref_i = ref[ref['itemid']==iid].sort_values('charttime')
            if ref_i.shape[0] == 1:
                df_.loc[:, col] = ref_i['valuenum'].item()
                continue
            for i in range(ref_i.shape[0]):
                ct_curr = ref_i.iloc[i]['charttime']
                if i == 0:
                    ct_next = ref_i.iloc[i+1]['charttime']
                    df_.loc[df_['charttime']<ct_next, col] = ref_i.iloc[i]['valuenum']
                elif i == ref_i.shape[0] - 1:
                    df_.loc[df_['charttime']>=ct_curr, col] = ref_i.iloc[i]['valuenum']
                else:
                    ct_next = ref_i.iloc[i+1]['charttime']
                    df_.loc[(df_['charttime']>=ct_curr) & (df_['charttime']<ct_next), col] = ref_i.iloc[i]['valuenum']

        for col in ['age', 'admission_weight']:
            df_[col] = info[col].item()
        for col in ['admission_type', 'admission_location']:
            df_[col] = info[col].cat.codes.item()


        df_model = pd.concat((df_model, df_), ignore_index=True)


    df_model.to_pickle(os.path.abspath('../data/data_los_pred.pkl'))


    print(11111)




