import os, sys
import pickle

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *

import pandas as pd

random_seed = 0
np.random.seed(random_seed)

from matplotlib.ticker import MaxNLocator

from sklearn import linear_model, neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import tqdm
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


if __name__ == '__main__':
    df_data = pd.read_pickle(os.path.abspath('../data/data_los_pred.pkl'))

    # for adm in df_data['hadm_id']:
    #     data = df_data[df_data['hadm_id']==adm]
    #     if np.logical_and(data['LOS'].diff() != -1, ~pd.isnull(data['LOS'].diff())).any():
    #         print(11)
    #     los = data['LOS'].to_numpy()
    #     if los[0]-los[-1]+1 != len(los):
    #         print(los[0], los[-1], los[0]-los[-1]+1, len(los))

    vital_cols = df_data.columns[-8:-1].to_list()
    # for adm in df_data['hadm_id'].unique():
    #     data = df_data[df_data['hadm_id']==adm]
    #     fig, ax = plt.subplots(2, 7, figsize=(24,8))
    #     plt.subplots_adjust(hspace=0.3)
    #     for i, col in enumerate(vital_cols):
    #         ax[0][i].set_title(col)
    #
    #         if 'Temperature' in col:
    #             ax[0][i].scatter(data['charttime'], data[col], label=col, c='firebrick')
    #         else:
    #             ax[0][i].plot(data['charttime'], data[col], label=col, c='firebrick')
    #         if col in ['HR', 'NBPs', 'NBPd']:
    #             colors_ref = ['dodgerblue', 'cornflowerblue', 'lightsteelblue']
    #             for ia, act in enumerate(['lying', 'sitting', 'standing']):
    #                 col_ref = 'Orthostatic ' + col.replace('N', '') + ' ' + act
    #                 ax[0][i].plot(data['charttime'], data[col_ref], label=col_ref, c=colors_ref[ia])
    #         # ax[0][i].legend()
    #         ax[0][i].xaxis.set_major_locator(MaxNLocator(5))
    #         plt.setp(ax[0][i].xaxis.get_majorticklabels(), rotation=30, ha='right')
    #         ax[1][i].hist(data[col])
    #     # plt.show()
    #
    #     folder_path = os.path.abspath(f'../plots/norm_data_per_patient/{adm}.png')
    #     plt.savefig(folder_path, dpi=300)

    # training_samples = pd.DataFrame(columns=['sample_id'] + df_data.columns.to_list())
    # sample_id = 0
    # adm_count = 0
    # training_sample_count = 0
    # invalid_count = 0
    # toi = 48
    # los_range = [2, 15]
    # for adm in df_data['hadm_id'].unique():
    #     data = df_data[df_data['hadm_id']==adm].copy()
    #     if data.shape[0] < los_range[0] * 24 or data.shape[0] > los_range[1] * 24:
    #         continue
    #     # if pd.isnull(data).sum().sum() > 0.2 * (data.shape[0] * data.shape[1]):
    #     #     continue
    #     adm_count += 1
    #     for begin in range(data.shape[0]-toi):
    #         sample = data.iloc[begin:begin+toi].copy()
    #         if pd.isnull(sample).sum().sum() > 0.2 * (sample.shape[0] * sample.shape[1]):
    #             invalid_count += 1
    #             continue
    #         training_sample_count += 1
    #         sample['sample_id'] = sample_id
    #         sample_id += 1
    #         training_samples = pd.concat((training_samples, sample), ignore_index=True)
    # print(adm_count, training_sample_count, invalid_count)
    # training_samples.to_pickle(os.path.abspath(f'../data/los_pred_{los_range[0]}-{los_range[1]}d_{toi}h.pkl'))


    # toi = 24
    # los_range = [1, 15]
    toi = 48
    los_range = [2, 15]
    training_samples = pd.read_pickle(os.path.abspath(f'../data/los_pred_{los_range[0]}-{los_range[1]}d_{toi}h.pkl'))

    if os.path.isfile(os.path.abspath('../data/data_los_pred_2-15d_48h_reformat.pkl')):
        data_dict = pickle.load(open(os.path.abspath('../data/data_los_pred_2-15d_48h_reformat.pkl'), 'rb'))
        X = data_dict['X']
        y = data_dict['y']
        del data_dict
    else:
        X = []
        y = []
        for sample_id in training_samples['sample_id'].unique():
            sample = training_samples[training_samples['sample_id']==sample_id]
            # x_info = sample[col_static].iloc[-1].to_list()
            # x_ref = sample[col_refs].iloc[-1].to_list()
            # x_ts = sample[vital_cols].to_list()
            # x_ = sample[col_static + col_refs + vital_cols].fillna(0).to_numpy()
            x_ = sample[col_static + col_refs + vital_cols].to_numpy()
            y_ = sample['LOS'].iloc[-1]

            X.append(x_)
            y.append(y_)

        X = np.array(X)
        y = np.array(y)
        pickle.dump(
            {'X': X, 'y': y},
            open(os.path.abspath('../data/data_los_pred_2-15d_48h_reformat.pkl'), 'wb')
        )
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, shuffle=True)

    train_adm, test_adm = train_test_split(training_samples['hadm_id'].unique(),
                                           test_size=.2, random_state=random_seed, shuffle=True)
    if os.path.isfile(os.path.abspath('../data/data_los_pred_2-15d_48h_hadm_split.pkl')):
        data_dict = pickle.load(open(os.path.abspath('../data/data_los_pred_2-15d_48h_hadm_split.pkl'), 'rb'))
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        del data_dict
    else:
        X_train, y_train, X_test, y_test = [], [], [], []
        for adm_list, X, y in zip([train_adm, test_adm], [X_train, X_test], [y_train, y_test]):
            for adm in adm_list:
                for sample_id in training_samples.loc[training_samples['hadm_id']==adm, 'sample_id'].unique():
                    sample = training_samples[training_samples['sample_id']==sample_id]
                    # x_ = sample[col_static + col_refs + vital_cols].fillna(0).to_numpy()
                    x_ = sample[col_static + col_refs + vital_cols].to_numpy()
                    y_ = sample['LOS'].iloc[-1]

                    X.append(x_)
                    y.append(y_)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        pickle.dump(
            {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test},
            open(os.path.abspath('../data/data_los_pred_2-15d_48h_hadm_split.pkl'), 'wb')
        )


# test regression models with raw data
    # test linear regression with raw data
    metrics = {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
    }

    X_train = np.nan_to_num(X_train.astype(float), nan=0.0)
    X_test = np.nan_to_num(X_test.astype(float), nan=0.0)
    y_train = np.nan_to_num(y_train.astype(float), nan=0.0)
    y_test = np.nan_to_num(y_test.astype(float), nan=0.0)

    model_lr = linear_model.LinearRegression()
    model_lr.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_lr = model_lr.predict(X_test.reshape(X_test.shape[0], -1))

    lr_score = {
        k: metrics[k](y_test, y_lr) for k in metrics
    }
    print(111)

    # test MLP regression with raw data
    # model_mlp = neural_network.MLPRegressor(
    #     hidden_layer_sizes=(1024, 2048, 512, 128, 64),
    #     activation='relu',
    #     solver='adam',
    #     batch_size=256,
    #     learning_rate_init=1e-3,
    #     max_iter=3000,
    #     shuffle=True,
    #     early_stopping=True,
    #     validation_fraction=.1,
    #     random_state=random_seed
    # )
    # model_mlp.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    # pickle.dump(model_mlp, open(os.path.abspath('../models/mlp_2-15d_48h_adm_split.pkl'), 'wb'))
    model_mlp = pickle.load(open(os.path.abspath('../models/mlp_2-15d_48h_adm_split.pkl'), 'rb'))
    y_mlp = model_mlp.predict(X_test.reshape(X_test.shape[0], -1))
    mlp_score = {
        k: metrics[k](y_test, y_mlp) for k in metrics
    }

    for adm in test_adm:
        if adm not in training_samples['hadm_id'].values:
            continue
        data_vis = df_data[df_data['hadm_id']==adm]
        samples = []
        los = []
        ct_los = []
        for sample_id in training_samples.loc[training_samples['hadm_id'] == adm, 'sample_id'].unique():
            sample = training_samples[training_samples['sample_id'] == sample_id]
            # x_info = sample[col_static].iloc[-1].to_list()
            # x_ref = sample[col_refs].iloc[-1].to_list()
            # x_ts = sample[vital_cols].to_list()
            x_ = sample[col_static + col_refs + vital_cols].fillna(0).to_numpy()
            y_ = sample['LOS'].iloc[-1]
            if pd.isnull(sample).sum().sum() > .15*48*24:
                print(pd.isnull(sample).sum().sum() / 48 / 24)
            if y_ < 5:
                print(222)

            ct_los.append(sample['charttime'].iloc[-1])
            samples.append(x_)
            los.append(y_)
        samples = np.array(samples)
        los = np.array(los)

        los_pred_lr = model_lr.predict(samples.reshape(samples.shape[0], -1))
        los_pred_mlp = model_mlp.predict(samples.reshape(samples.shape[0], -1))

        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax_los = ax.twinx()
        for col in vital_cols:
            ax.plot(data_vis['charttime'], data_vis[col], label=col)
        plt.legend()
        ax_los.plot(ct_los, los, 'ro', label='LOS_gt')
        ax_los.plot(ct_los, los_pred_lr, 'go', label='LOS_lr')
        ax_los.plot(ct_los, los_pred_mlp, 'bo', label='LOS_mlp')
        plt.legend()
        plt.show()
        print(1111)


# test baysian regression model with gaussian process input data






    sample = training_samples[training_samples['sample_id'] == 5]
    data = data = sample[['charttime', 'HR']].set_index('charttime')
    result = seasonal_decompose(data, period=24, model='additive')
    fig = result.plot()
    plt.show()

    seasonal_loess1 = calculate_smoothed_seasonal(result, method='loess', loess_deg=1, frac=.25)
    seasonal_loess2 = calculate_smoothed_seasonal(result, method='loess', loess_deg=2, frac=.25)
    seasonal_lowess = calculate_smoothed_seasonal(result, method='lowess', frac=.25)
    fig = plt.figure()
    plt.plot(result.seasonal.values, label='raw')
    plt.plot(seasonal_loess1, label='loess1deg')
    plt.plot(seasonal_loess2, label='loess2deg')
    plt.plot(seasonal_lowess, label='lowess')
    plt.legend()
    plt.show()





    print(11111)