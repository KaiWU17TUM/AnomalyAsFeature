import os, sys
import pickle
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *
from utils.common import *

import tqdm
import pandas as pd
np.random.seed(0)

import GPy as gpy


def one_hot_encoding(text, text_dict):
    ohk = [0] * len(text_dict)
    ohk[text_dict[text]] = 1
    return ohk

def  one_hot_encoding_arr(arr, text_dict):
    ohk_arr = []
    for text in arr:
        ohk = one_hot_encoding(text, text_dict)
        ohk_arr.append(ohk)
    return np.array(ohk)

if __name__ == '__main__':
    data_path = os.path.join(os.path.abspath(''), '..', 'data_all')

    adm_cohort = pickle.load(open(os.path.join(data_path, 'admission_cohort.pkl'), 'rb'))
    selected_chart_items = pickle.load(
        open(os.path.join(data_path, 'selected_chart_items.pkl'), 'rb')
    )
    selected_vital_iids = selected_chart_items['vital']
    selected_lab_iids = selected_chart_items['lab']
    del selected_chart_items
    item_text_vals = pickle.load(
        open(os.path.join(data_path, 'text_chart_onehotkey.pkl'), 'rb')
    )

    selected_numeric_iid = item_dict.loc[
        (item_dict['itemid'].isin(selected_vital_iids + selected_lab_iids)) &
        (item_dict['param_type'].isin(['Numeric', 'Numeric with tag'])),
        'itemid'
    ].to_list()
    selected_text_iid = item_dict.loc[
        (item_dict['itemid'].isin(selected_vital_iids + selected_lab_iids)) &
        (item_dict['param_type']=='Text'),
        'itemid'
    ].to_list()

    # get normalization parameters
    if os.path.isfile(os.path.join(data_path, 'norm_param_chart.pkl')):
        norm_param_numeric = pickle.load(open(os.path.join(data_path, 'norm_param_chart.pkl'), 'rb'))
        norm_param_info = pickle.load(open(os.path.join(data_path, 'norm_param_info.pkl'), 'rb'))
        text_ohk_chart = pickle.load(open(os.path.join(data_path, 'text_chart_onehotkey.pkl'), 'rb'))
        text_ohk_info = pickle.load(open(os.path.join(data_path, 'text_info_onehotkey.pkl'), 'rb'))
        print(1)
    else:
        info_vals = {
            'age':[],
            'admission_weight':[],
            'gender':[],
            'admission_type':[],
        }
        norm_param_numeric = {
            iid:{} for iid in selected_numeric_iid
        }
        item_vals = {
            iid:[] for iid in selected_numeric_iid
        }
        for adm in tqdm.tqdm(adm_cohort):
            info = pd.read_csv(os.path.join(data_path, 'raw', f'{adm}_info.csv'))
            data = pd.read_csv(os.path.join(data_path, 'raw', f'{adm}.csv'))
            for key in info_vals:
                info_vals[key] += info[key].to_list()
            for iid in selected_numeric_iid:
                val = data.loc[data['itemid']==iid, 'valuenum'].astype(float).to_list()
                item_vals[iid] += val
        for iid in norm_param_numeric:
            d = np.array(item_vals[iid])
            lower = np.percentile(d, 1)
            upper = np.percentile(d, 99)
            d_1_99 = d[np.argwhere((d >= lower) & (d <= upper))]

            norm_param_numeric[iid]['max'] = np.nanmax(d_1_99)
            norm_param_numeric[iid]['min'] = np.nanmin(d_1_99)
            norm_param_numeric[iid]['mean'] = np.nanmean(d_1_99)
            norm_param_numeric[iid]['std'] = np.nanstd(d_1_99)

        info_text_vals = {
            'gender': {'F':0, 'M':1},
            'admission_type': {
                k: i for i, k in enumerate(set(info_vals['admission_type']))
            }
        }
        norm_param_info = {k:{} for k in ['age', 'admission_weight']}
        for k in ['age', 'admission_weight']:
            d = np.array(info_vals[k])
            d = d[~np.isnan(d)]
            lower = np.percentile(d, 1)
            upper = np.percentile(d, 99)
            d_1_99 = d[np.argwhere((d >= lower) & (d <= upper))]
            norm_param_info[k]['max'] = np.nanmax(d_1_99)
            norm_param_info[k]['min'] = np.nanmin(d_1_99)
            norm_param_info[k]['mean'] = np.nanmean(d_1_99)
            norm_param_info[k]['std'] = np.nanstd(d_1_99)

        pickle.dump(norm_param_numeric, open(os.path.join(data_path, 'norm_param_chart.pkl'), 'wb'))
        pickle.dump(
            {
                'info': info_vals,
                'chart_numeric': item_vals,
            },
            open(os.path.join(data_path, 'info_and_numeric_vals.pkl'), 'wb')
        )
        pickle.dump(norm_param_info, open(os.path.join(data_path, 'norm_param_info.pkl'), 'wb'))
        pickle.dump(info_text_vals, open(os.path.join(data_path, 'text_info_onehotkey.pkl'), 'wb'))


    # normalize + one hot key + interpolation
    save_path = os.path.join(data_path, 'preprocessed')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    columns = ['hadm_id', 'charttime'] +\
              ['age', 'admission_weight', 'gender', 'admission_type'] +\
              [item_dict[item_dict['itemid']==iid]['abbreviation'].item() for iid in selected_numeric_iid + selected_text_iid] +\
              ['LOS']
    for adm in tqdm.tqdm(adm_cohort):
        try:
            data = pd.read_csv(os.path.join(data_path, 'raw', f'{adm}.csv'))
            info = pd.read_csv(os.path.join(data_path, 'raw', f'{adm}_info.csv'))
            data_ = data.copy()
            data_['charttime'] = pd.to_datetime(data_['charttime'])
            # normalize + one hot encode
            for iid in selected_numeric_iid:
                d = data[data['itemid']==iid]['valuenum']
                d = (d - norm_param_numeric[iid]['mean']) /  norm_param_numeric[iid]['std']
                data_.loc[data_['itemid']==iid, 'valuenum'] = d
            for iid in selected_text_iid:
                d = data[data['itemid']==iid]['value']
                # d = one_hot_encoding_arr(d, text_ohk_chart[iid])
                data_.loc[data_['itemid']==iid, 'valuenum'] = [text_ohk_chart[iid][di] for di in d.values]

            ct_disch = pd.to_datetime(info['dischtime']).item()
            ct_start = data_['charttime'].min()
            ct_start = ct_start.replace(hour=ct_start.hour, minute=0, second=0, microsecond=0)
            ct_end = data_['charttime'].max()
            data_ = interpolate_charttime_df(data_, freq='1H', method='fillin', begin=ct_start, end=ct_end)

            df_data = pd.DataFrame(columns=columns)
            df_data['charttime'] = pd.date_range(ct_start, ct_end, freq='1H')
            los = ct_disch - df_data['charttime']
            df_data['LOS'] = [dt.days*24 + dt.seconds//3600 for dt in los]
            df_data['hadm_id'] = adm
            for col in ['age', 'admission_weight']:
                df_data[col] = info[col].item()
            for col in ['gender', 'admission_type']:
                df_data[col] = text_ohk_info[col][info[col].item()]
            for iid in selected_numeric_iid + selected_text_iid:
                try:
                    col = item_dict[item_dict['itemid']==iid]['abbreviation'].item()
                    df_data[col] = data_.loc[data_['itemid']==iid, 'valuenum'].values
                except:
                    pass

            # trend + seasonality
            df_data = df_data.reindex(columns = df_data.columns.tolist() + [item_dict[item_dict['itemid']==iid]['abbreviation'].item() + '_mean' for iid in selected_numeric_iid])
            df_data = df_data.reindex(columns = df_data.columns.tolist() + [item_dict[item_dict['itemid']==iid]['abbreviation'].item() + '_var' for iid in selected_numeric_iid])
            df_data = df_data.reindex(columns = df_data.columns.tolist() + [item_dict[item_dict['itemid']==iid]['abbreviation'].item() + f'_encoded{i}' for iid in selected_numeric_iid for i in range(1, 5)])

            for iid in selected_numeric_iid:
                col = item_dict[item_dict['itemid']==iid]['abbreviation'].item()
                d_encoded = np.zeros((df_data.shape[0], 4))

                y = df_data[col].to_numpy().astype(float)
                x = np.linspace(0, df_data.shape[0]-1, df_data.shape[0])
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                if x.size == 0:
                    # df_data[col+'_mean'] = 0
                    # df_data[col+'_var'] = 0
                    df_data[df_data.columns[df_data.columns.str.startswith(col+'_encoded')]] = 0
                    continue

                anomaly = detect_sudden_change_anomaly(x, y, margin=.2, plot=False)
                x_ = np.delete(x, anomaly['peaks'])
                y_ = np.delete(y, anomaly['peaks'])

                kern_trend = gpy.kern.RBF(input_dim=1, variance=1., lengthscale=24)
                kern_trend.lengthscale.constrain_bounded(lower=24, upper=48, warning=False)
                kern_periodic_std = gpy.kern.StdPeriodic(input_dim=1, variance=1., lengthscale=3., period=24)
                kern_periodic_std.period.constrain_bounded(22, 26, warning=False)
                kern_periodic_std.lengthscale.constrain_bounded(.5, 10, warning=False)

                model_trend = gpy.models.GPRegression(x_.reshape(-1, 1), y_.reshape(-1, 1), kern_trend)
                model_trend.optimize()

                trend_mean, trend_cov = model_trend.predict(x_.reshape(-1, 1), full_cov=True)
                y_detrend = y_ - trend_mean.ravel()

                model_season_ = gpy.models.GPRegression(x_.reshape(-1, 1), y_detrend.reshape(-1, 1), kern_periodic_std)
                model_season_.optimize()


                trend_mean, trend_var = model_trend.predict(np.arange(0, df_data.shape[0]+1).reshape(-1, 1), full_cov=False)
                season_mean, season_var = model_season_.predict(np.arange(0, df_data.shape[0]+1).reshape(-1, 1), full_cov=False)
                trend_diff = trend_mean[1:] - trend_mean[:-1]

                mean = trend_mean + season_mean
                var = trend_var + season_var
                # likelihood = likelihoods.Gaussian(variance=1.)
                # quantiles = likelihood.predictive_quantiles(mean, var, (2.5, 97.5))

                lower = mean - 2 * var
                upper = mean + 2 * var

                for i, idx in enumerate(x_.astype(int)):
                    if y_[i] < lower[idx]:
                        if trend_diff[idx] < 0:
                            d_encoded[idx, 0] = y_[i] - trend_diff[idx]
                        elif trend_diff[idx] >= 0:
                            d_encoded[idx, 1] = y_[i] - trend_diff[idx]
                    elif y_[i] > upper[idx]:
                        if trend_diff[idx] < 0:
                            d_encoded[idx, 2] = y_[i] - trend_diff[idx]
                        elif trend_diff[idx] >= 0:
                            d_encoded[idx, 3] = y_[i] - trend_diff[idx]

                df_data[col+'_mean'] = mean[:-1]
                df_data[col+'_var'] = var[:-1]
                df_data[df_data.columns[df_data.columns.str.startswith(col+'_encoded')]] = d_encoded



                # trend_mean, trend_var = model_trend.predict(np.arange(-50, 200).reshape(-1, 1), full_cov=False)
                # season_mean, season_var = model_season_.predict(np.arange(-50, 200).reshape(-1, 1), full_cov=False)
                # mean = trend_mean + season_mean
                # var = trend_var + season_var
                #
                # fig, ax = plt.subplots(3, 1, figsize=(12, 8))
                # model_trend.plot_f(ax=ax[0])
                # ax[0].scatter(x_, y_, c='black', marker='x')
                # ax[0].scatter(x[anomaly['peaks']], y[anomaly['peaks']], c='r', marker='^')
                #
                # model_season_.plot_f(ax=ax[1])
                # ax[1].scatter(x_, y_detrend, c='black', marker='x')
                #
                # ax[2].plot(np.arange(-50, 200), mean, label='mean')
                # ax[2].fill_between(np.arange(-50, 200), (mean-var).ravel(), (mean+var).ravel(), label='confidence', alpha=.3, color='b')
                # ax[2].scatter(x_, y_, c='black', marker='x')
                #
                # plt.show()
                # print(11)

            pickle.dump(df_data, open(os.path.join(save_path, f'{adm}.pkl'), 'wb'))
        except:
            print(adm)

    print(111)

