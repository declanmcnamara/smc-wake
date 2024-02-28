import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import torch.distributions as D


def generate_data(n, k, A, sigma, tau, prior, n_obs=25):
    zs = prior.sample((n_obs,))
    means = A @ zs.T
    xs = D.MultivariateNormal(means.T, tau**2 * torch.eye(n)).sample()
    return zs, xs
