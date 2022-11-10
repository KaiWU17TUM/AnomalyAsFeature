import os, sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *

import pandas as pd
np.random.seed(0)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import gridspec


if __name__ == '__main__':
    # load data
    df_info, df_chart, vital_ref_dict, selected_items = load_data()
    item_vital = selected_items['vital']
    item_resp = selected_items['resp']
    item_lab = selected_items['lab']
    item_treat = selected_items['treatment']
    item_pain = selected_items['pain']


    # select pain events with pain level >= 6:
    # 'Moderate to Severe', 'Moderate to Severe.', 'Severe', 'Severe to Worse', 'Worst'
    pain_event = df_chart[(df_chart['itemid']==223791) & (df_chart['valuenum']>=6)]
    item_selected = pd.concat((item_dict[(item_dict['itemid'].isin(item_vital)) & (item_dict['param_type']=='Numeric')],
                               item_dict[(item_dict['itemid'].isin(item_resp)) & (item_dict['param_type']=='Numeric')]))
    item_selected = item_selected.drop(item_selected.index[-1])
    vital_cols = item_selected['abbreviation'].tolist()

    toi_prev = 6
    toi_post = 0
    len_seg = toi_prev + toi_post

    # Patient data near the pain event
    if os.path.isfile(os.path.abspath('../data/df_pain.csv')):
        cvt = {k: lambda x: [float(d) for d in x.replace('[','').replace(']','').split(',')]
               for k in vital_cols}
        df_pain = pd.read_csv(os.path.abspath('../data/df_pain.csv'), header=0, index_col=[0], converters=cvt)
    else:
        df_pain = pd.DataFrame(columns=['hadm_id', 'charttime', 'pain_level'] + vital_cols)
        for i, (_, event) in enumerate(pain_event.iterrows()):
            adm = event['hadm_id']
            ct = event['charttime']
            df_pain.loc[i, 'hadm_id'] = adm
            df_pain.loc[i, 'charttime'] = ct
            df_pain.loc[i, 'pain_level'] = event['valuenum']
            for _, vital in item_selected.iterrows():
                data_ = df_chart[(df_chart['hadm_id']==adm) & (df_chart['itemid']==vital['itemid'])]
                df_ = get_data_near_event(data_, ct, toi_prev, toi_post, fillna=None)
                df_pain.loc[i, vital['abbreviation']] = df_['valuenum'].tolist()
        df_pain.to_csv(os.path.abspath('../data/df_pain.csv'))

    # Patient data at non-pain status
    if os.path.isfile(os.path.abspath('../data/adm_ts_no_pain.csv')):
        adm_ts_no_pain = pd.read_csv(os.path.abspath('../data/adm_ts_no_pain.csv'), header=0, index_col=[0])
    else:
        adm_ts_no_pain = []
        for adm in df_info['hadm_id'].unique():
            data = df_chart[(df_chart['hadm_id']==adm) & (df_chart['itemid'].isin(item_selected['itemid']))].copy()
            data = interpolate_charttime_df(data)

            for ct_pain in pain_event[pain_event['hadm_id']==adm]['charttime']:
                delta_t = pd.Timedelta(hours = len_seg)
                data.drop(data[(data['charttime']>=ct_pain-delta_t) & (data['charttime']<=ct_pain+delta_t)].index,
                          inplace=True)
            list_of_data = [d for _, d in data.groupby(['itemid', data.index - np.arange(len(data))])]
            list_of_data = [d for d in list_of_data if d.shape[0] >= len_seg]
            list_of_data = [d.drop(d.index[-len_seg+1:]) for d in list_of_data]

            ts_list = [list(d['charttime']) for d in list_of_data]
            ts_list = [ts for l in ts_list for ts in l]
            ts_list = list(set(ts_list))
            ts_count = {ts: data[data['charttime']==ts].shape[0]-pd.isnull(data[data['charttime']==ts]['valuenum']).sum() for ts in ts_list}

            for ts in ts_list:
                if ts_count[ts] < 5: # valid signal sources
                    del ts_count[ts]
            adm_ts_no_pain += [(adm, ts) for ts in ts_count]
        adm_ts_no_pain = pd.DataFrame(adm_ts_no_pain, columns=['hadm_id', 'charttime'])
        adm_ts_no_pain.to_csv(os.path.abspath('../data/adm_ts_no_pain.csv'))

    df_normal = pd.DataFrame(columns=df_pain.columns)
    # idx_normal = np.random.choice(range(adm_ts_no_pain.shape[0]), pain_event.shape[0], replace=False)

    # for i, idx in enumerate(idx_normal):
    for i in range(len(adm_ts_no_pain)):
        try:
            adm, ct = adm_ts_no_pain.iloc[i]
            df_normal.loc[i, 'hadm_id'] = adm
            df_normal.loc[i, 'charttime'] = ct
            df_normal.loc[i, 'pain_level'] = 0
            for _, vital in item_selected.iterrows():
                data_ = df_chart[(df_chart['hadm_id']==adm) & (df_chart['itemid']==vital['itemid'])]
                df_ = get_data_near_event(data_, ct, 0, len_seg, fillna=None)
                df_normal.loc[i, vital['abbreviation']] = df_['valuenum'].tolist()
        except:
            print(11111)

    # hr_pain = [d for l in df_pain['HR'].tolist() for d in l]
    # hr_no_pain = [d for l in df_normal['HR'].tolist() for d in l]
    # rr_pain = [d for l in df_pain['RR'].tolist() for d in l]
    # rr_no_pain = [d for l in df_normal['RR'].tolist() for d in l]

    df_pain['label'] = 1
    df_normal['label'] = 0

    df_all = pd.concat((df_pain, df_normal))
    X = np.array([df_all.iloc[i][vital_cols].to_list() for i in range(df_all.shape[0])])
    y = df_all['label'].to_numpy().reshape(-1, 1)

    df_model_data = pd.DataFrame(columns=['X', 'y'])
    df_model_data['X'] = [x for x in X]
    df_model_data['y'] = df_all['pain_level']
    df_model_data['y_label'] = y

    df_model_data.to_pickle(os.path.abspath('../data/data_pain_detector.pkl'))


    # df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=0)
    #
    # X_train = np.array([df_train.iloc[i][vital_cols].to_list() for i in range(df_train.shape[0])])
    # y_train = df_train['label'].to_numpy().reshape(-1, 1)
    # X_test = np.array([df_test.iloc[i][vital_cols].to_list() for i in range(df_test.shape[0])])
    # y_test = df_test['label'].to_numpy().reshape(-1, 1)
    #
    # np.save(
    #     os.path.abspath('../data/data_pain_detector'),
    #     {
    #         'X_train': X_train,
    #         'y_train': y_train,
    #         'X_test': X_test,
    #         'y_test': y_test,
    #     },
    #     allow_pickle=True
    # )



    print(111)
