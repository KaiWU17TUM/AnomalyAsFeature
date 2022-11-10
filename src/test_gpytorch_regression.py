import os, sys
import pickle

import gpytorch.constraints

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *
from utils.gp import *

import pandas as pd

random_seed = 0
np.random.seed(random_seed)

import tqdm

import torch



def train(x, y, model, likelihood, num_epochs=50, lr=1e-2):
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    if isinstance(model, gpytorch.models.ExactGP):
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr=lr)

        # mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            if i % 50 == 0:
                print(f'Epoch {i}\t---\tLOSS{loss}')
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            x = x.cpu()
            y = y.cpu()
            model = model.cpu()
            likelihood = likelihood.cpu()

    else:
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr)

        # mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=y.size(0))

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            if i % 50 == 0:
                print(f'Epoch {i}\t---\tLOSS{loss}')
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            x = x.cpu()
            y = y.cpu()
            model = model.cpu()
            likelihood = likelihood.cpu()


if __name__ == '__main__':
    df_data = pd.read_pickle(os.path.abspath('../data/data_los_pred.pkl'))

    toi = 48
    los_range = [2, 15]
    training_samples = pd.read_pickle(os.path.abspath(f'../data/los_pred_{los_range[0]}-{los_range[1]}d_{toi}h.pkl'))

    data_dict = pickle.load(open(os.path.abspath('../data/data_los_pred_2-15d_48h_reformat.pkl'), 'rb'))
    X = data_dict['X']
    Y = data_dict['y']
    del data_dict

    for adm in df_data['hadm_id'].unique():
        sample = df_data[df_data['hadm_id']==adm]
        hr = sample['HR'].to_numpy().astype(float)
        nbps = sample['NBPs'].to_numpy().astype(float)
        temp = sample['Temperature F'].to_numpy().astype(float)
        rr = sample['RR'].to_numpy().astype(float)

        for y in [hr, nbps, temp, rr]:
            y = np.array(y)
            x = np.linspace(0, sample.shape[0]-1, sample.shape[0])
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # likelihood = gpytorch.likelihoods.LaplaceLikelihood()
            model_gp = VariationalGPModel(inducing_points=y[np.random.choice(range(len(y)), len(y), replace=False)],
                                          kernal='locally_periodic')
            # model_gp = ExactGPModel(x, y, likelihood, kernal='locally_periodic')


            likelihood.train()
            model_gp.train()
            train(x, y, model_gp, likelihood, num_epochs=1000, lr=5e-2)

            model_gp.eval()
            likelihood.eval()
            x_ = np.linspace(0, sample.shape[0]-1, sample.shape[0])
            with torch.no_grad():
                observed_pred = likelihood(model_gp(torch.Tensor(x_)))
                lower, upper = observed_pred.confidence_region()

                f, ax = plt.subplots(1, 1, figsize=(8, 3))
                ax.scatter(x, y, c='gray')
                ax.plot(x_, observed_pred.mean.numpy(), 'b')
                ax.fill_between(x_, lower.numpy(), upper.numpy(), alpha=.5)
                plt.show()

            print(111)
