import os, sys
import pickle

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *
from utils.gp import *
from utils.common import *

import pandas as pd

random_seed = 0
np.random.seed(random_seed)

import tqdm
import GPy as gpy
from GPy.core.gp import likelihoods


if __name__ == '__main__':

    if os.path.isfile(os.path.abspath('../data/data_los_pred_encoded.pkl')):
        df_data = pd.read_pickle(os.path.abspath('../data/data_los_pred_encoded.pkl'))
    else:
        df_data = pd.read_pickle(os.path.abspath('../data/data_los_pred.pkl'))
        df_data = df_data.reindex(columns = df_data.columns.tolist() + [col + '_mean' for col in col_vital])
        df_data = df_data.reindex(columns = df_data.columns.tolist() + [col + '_var' for col in col_vital])
        df_data = df_data.reindex(columns = df_data.columns.tolist() + [col + f'_encoded{i}' for col in col_vital for i in range(1, 5)])
        for adm in tqdm.tqdm(df_data['hadm_id'].unique()):
            data = df_data[df_data['hadm_id']==adm].copy()

            for col in col_vital:
                d_encoded = np.zeros((data.shape[0], 4))

                y = data[col].to_numpy().astype(float)
                x = np.linspace(0, data.shape[0]-1, data.shape[0])
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
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

                trend_mean, trend_var = model_trend.predict(np.arange(0, data.shape[0]+1).reshape(-1, 1), full_cov=False)
                season_mean, season_var = model_season_.predict(np.arange(0, data.shape[0]+1).reshape(-1, 1), full_cov=False)
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

                # plt.plot(np.arange(0, x_.max()+2), mean, label='mean')
                # plt.fill_between(np.arange(0, x_.max()+2), lower.ravel(), upper.ravel(), label='confidence', alpha=.3, color='b')
                # plt.scatter(x_, y_, c='black', marker='x')
                # plt.show()

                df_data.loc[df_data['hadm_id']==adm, df_data.columns.str.startswith(col+'_encoded')]= d_encoded
                df_data.loc[df_data['hadm_id']==adm, df_data.columns.str.startswith(col+'_mean')]= mean[:-1]
                df_data.loc[df_data['hadm_id']==adm, df_data.columns.str.startswith(col+'_var')]= var[:-1]

        df_data.to_pickle(os.path.abspath('../data/data_los_pred_encoded.pkl'))



    df_48h = pd.DataFrame(columns = ['slice_id'] + df_data.columns.tolist())
    slice_id = 0
    for adm in df_data['hadm_id'].unique():
        data = df_data[df_data['hadm_id']==adm].copy()
        for i in range(0, data.shape[0]-48):
            data_slice = data.iloc[i:i+48]
            data_slice['slice_id'] = slice_id
            df_48h = pd.concat((df_48h, data_slice), ignore_index=True)
            slice_id += 1

    df_48h.to_pickle(os.path.abspath('../data/data_los_pred_encoded_48h.pkl'))



    print(111)