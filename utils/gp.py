import math

import torch

import gpytorch
from gpytorch.kernels import PeriodicKernel
from gpytorch.functions import RBFCovariance
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class LocallyPeriodicKernal(PeriodicKernel):
    has_lengthscale = True

    def postprocess_rbf(self, dist_mat):
        return dist_mat.div_(-2).exp_()

    def forward(self, x1, x2, diag=False, **params):
        # PERIODIC COVARIANCE
        # Pop this argument so that we can manually sum over dimensions
        last_dim_is_batch = params.pop("last_dim_is_batch", False)
        # Get lengthscale
        lengthscale = self.lengthscale

        x1_ = x1.div(self.period_length / math.pi)
        x2_ = x2.div(self.period_length / math.pi)
        # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
        diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        if diag:
            lengthscale = lengthscale[..., 0, :, None]
        else:
            lengthscale = lengthscale[..., 0, :, None, None]
        exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)

        if not last_dim_is_batch:
            exp_term = exp_term.sum(dim=(-2 if diag else -3))
        periodic_cov = exp_term.exp()

        # RBF COVARIANCE
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        rbf_cov = self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=self.postprocess_rbf, postprocess=True, **params
        )

        local_periodic_cov = rbf_cov * periodic_cov

        return local_periodic_cov


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernal=None):
        super(ExactGPModel, self).__init__(torch.Tensor(train_x), torch.Tensor(train_y), likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        lengthscale_prior = gpytorch.priors.GammaPrior(5.0, 3.0)
        periodic_prior = gpytorch.priors.GammaPrior(24.0, 5.0)
        if kernal == 'locally_periodic':
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)) \
            #                 + gpytorch.kernels.ScaleKernel(LocallyPeriodicKernal(lengthscale_prior=periodic_prior))
            rq = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            linear = gpytorch.kernels.LinearKernel()
            periodic = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(period_length_prior=periodic_prior))
            self.covar_module = gpytorch.kernels.AdditiveKernel(rq, periodic, linear)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPModel(ApproximateGP):
    def __init__(self, inducing_points, kernal=None):
        inducing_points = torch.Tensor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        lengthscale_prior = gpytorch.priors.GammaPrior(5.0, 3.0)
        periodic_prior = gpytorch.priors.GammaPrior(24.0, 5.0)
        # periodic_constraint = gpytorch.constraints.Interval(20, 30)
        if kernal == 'locally_periodic':
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)) \
            #                 + gpytorch.kernels.ScaleKernel(LocallyPeriodicKernal(lengthscale_prior=periodic_prior))
            rq = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            linear = gpytorch.kernels.LinearKernel()
            periodic = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(period_length_prior=periodic_prior,))
                                                                                    # period_length_constraint=periodic_constraint))
            self.covar_module = gpytorch.kernels.AdditiveKernel(rq, periodic)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
