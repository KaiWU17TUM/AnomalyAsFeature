import os, sys
import pickle

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *
from utils.gp import *

import pandas as pd

random_seed = 0
np.random.seed(random_seed)

import tqdm
import GPy as gpy


if __name__ == '__main__':
    df_data = pd.read_pickle(os.path.abspath('../data/data_los_pred.pkl'))

    toi = 48
    los_range = [2, 15]
    training_samples = pd.read_pickle(os.path.abspath(f'../data/los_pred_{los_range[0]}-{los_range[1]}d_{toi}h.pkl'))

    data_dict = pickle.load(open(os.path.abspath('../data/data_los_pred_2-15d_48h_reformat.pkl'), 'rb'))
    X = data_dict['X']
    Y = data_dict['y']
    del data_dict

    for adm in df_data['hadm_id'].unique()[4:]:

        sample = df_data[df_data['hadm_id']==adm]
        hr = sample['HR'].to_numpy().astype(float)
        nbps = sample['NBPs'].to_numpy().astype(float)
        temp = sample['Temperature F'].to_numpy().astype(float)
        rr = sample['RR'].to_numpy().astype(float)

        for y in [hr]: #[hr, nbps, temp, rr]:
            y = np.array(y)
            x = np.linspace(0, sample.shape[0]-1, sample.shape[0])
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            anomaly = detect_sudden_change_anomaly(x, y, margin=.3, plot=False)
            x_ = np.delete(x, anomaly['peaks'])
            y_ = np.delete(y, anomaly['peaks'])


            rbf_kern = gpy.kern.RBF(input_dim=1, variance=1., lengthscale=1)
            trend_kern = gpy.kern.RBF(input_dim=1, variance=1., lengthscale=24)
            trend_kern.lengthscale.constrain_bounded(lower=24, upper=48)
            noise_kern = gpy.kern.White(input_dim=1, variance=.5)
            periodic_std_kern = gpy.kern.StdPeriodic(input_dim=1, variance=1., lengthscale=1., period=24)
            periodic_std_kern.period.constrain_bounded(22, 26)
            periodic_exp_kern = gpy.kern.PeriodicExponential(input_dim=1, variance=1., lengthscale=1., period=24)
            periodic_m32_kern = gpy.kern.PeriodicMatern32(input_dim=1, variance=1., lengthscale=1., period=24)
            periodic_m32_kern.period.constrain_bounded(22, 26)
            periodic_m52_kern = gpy.kern.PeriodicMatern52(input_dim=1, variance=1., lengthscale=1., period=24)
            mlp_kern = gpy.kern.MLP(1) + gpy.kern.Bias(1)


            # model = gpy.models.GPHeteroscedasticRegression(x.reshape(-1, 1), y.reshape(-1, 1), periodic_std_kern)
            model_trend = gpy.models.GPRegression(x_.reshape(-1, 1), y_.reshape(-1, 1), trend_kern)
            model_trend.optimize()
            # x_detrend = x_ - model_trend.predict()
            model_season = gpy.models.GPRegression(x_.reshape(-1, 1), y_.reshape(-1, 1), periodic_std_kern)
            model_season.optimize()

            # model.plot_f(ax=ax[0])
            # ax[0].scatter(x_, y_, c='black', marker='x')
            # ax[0].scatter(x[anomaly['peaks']], y[anomaly['peaks']], c='r', marker='^')
            # model.plot_data(ax=ax[0])
            # model.optimize()
            # model.optimize_restarts(10, parallel=False, messages=True)

            trend_mean, trend_cov = model_trend.predict(x_.reshape(-1, 1), full_cov=True)
            y_detrend = y_ - trend_mean.ravel()
            model_season_ = gpy.models.GPRegression(x_.reshape(-1, 1), y_detrend.reshape(-1, 1), periodic_std_kern)
            model_season_.optimize()

            fig, ax = plt.subplots(3, 1, figsize=(12,6))
            model_trend.plot_f(ax=ax[0])
            ax[0].scatter(x_, y_, c='black', marker='x')
            ax[0].scatter(x[anomaly['peaks']], y[anomaly['peaks']], c='r', marker='^')

            model_season.plot_f(ax=ax[1])
            ax[1].scatter(x_, y_, c='black', marker='x')
            ax[1].scatter(x[anomaly['peaks']], y[anomaly['peaks']], c='r', marker='^')
            # model.plot_errorbars_trainset(ax=ax[1], alpha=1)

            model_season_.plot_f(ax=ax[2])
            ax[2].scatter(x_, y_detrend, c='black', marker='x')
            # ax[2].scatter(x[anomaly['peaks']], y[anomaly['peaks']], c='r', marker='^')

            fig.tight_layout()
            plt.show()
            print(111)




            # for kern in [periodic_std_kern, periodic_exp_kern, periodic_m32_kern, periodic_m52_kern]:
            #     model = gpy.models.GPRegression(x.reshape(-1, 1), y.reshape(-1, 1), kern)
            #     model.optimize()
            #     fig, ax = plt.subplots(1,1,figsize=(13,5))
            #     model.plot_f(ax=ax)
            #     model.plot_data(ax=ax)
            #     model.plot_errorbars_trainset(ax=ax, alpha=1)
            #     fig.tight_layout()
            #     plt.show()
            #     print(111)



            # x_ = np.linspace(0, sample.shape[0]-1, sample.shape[0])
            # with torch.no_grad():
            #     observed_pred = likelihood(model_gp(torch.Tensor(x_)))
            #     lower, upper = observed_pred.confidence_region()
            #
            #     f, ax = plt.subplots(1, 1, figsize=(8, 3))
            #     ax.scatter(x, y, c='gray')
            #     ax.plot(x_, observed_pred.mean.numpy(), 'b')
            #     ax.fill_between(x_, lower.numpy(), upper.numpy(), alpha=.5)
            #     plt.show()

            print(111)
