import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys

sys.path.append("../")
import math
import random
from os.path import exists

import hydra
import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from generate import generate_data
from hydra import compose, initialize
from modules import Encoder, FullRankEncoder
from omegaconf import DictConfig, OmegaConf
from utils import exact_posteriors

from smc.smc_sampler import (
    EmpiricalDistribution,
    LikelihoodTemperedSMC,
    MetaImportanceSampler,
    Sampler,
)

"""
Our SMC sampler object currently takes in three functions, which allow for greater
flexibility. The three functions are, briefly:

1) log_target: computes log p(x | z)
2) log_prior: computes log p(z)
3) proposal: prescribes how to propose new zs from current zs

We describe each of these functions in more detail below. The first two are problem-
specific, while the last is a design choice made by the user.
"""


def log_target(params, context, **kwargs):
    """
    Given a 'params' object of size N x c x ...,
    where N is the number of particles, c is current number
    of SMC steps, ... is remaining dimension.

    Returns log p(x | z_c), for likelihood tempered SMC. In other
    words, we ignore z_1, ..., z_{c-1} and only use the most recent
    stage of the params input.
    """
    A = kwargs["A"]
    tau = kwargs["tau"]
    n = kwargs["n"]
    N = params.shape[0]
    z_to_use = params
    means = torch.bmm(A.repeat(N, 1, 1), z_to_use.unsqueeze(-1))
    temp = (means - context.reshape(-1, 1)).squeeze(-1)
    mags = torch.sum(torch.square(temp), -1) * (-1 / 2) * (1 / tau**2)
    log_probs = (
        -1 * n * torch.log(torch.tensor(tau))
        + -1 * n * torch.log(torch.tensor((2 * math.pi)))
        + mags
    )
    # distr = D.MultivariateNormal(means.squeeze(-1), tau**2*torch.eye(n))
    # log_probs = distr.log_prob(context) # shape N x len(true_x)
    return log_probs


def log_prior(params, **kwargs):
    """
    Given a 'params' object of size N x c x ...,
    where N is the number of particles, c is current number
    of SMC steps, ... is remaining dimension.

    Returns log p(z_c), for likelihood tempered SMC. In other
    words, we ignore z_1, ..., z_{c-1} and only use the most recent
    stage of the params input.
    """
    prior = kwargs["prior"]
    z_to_use = params
    lps = prior.log_prob(z_to_use)  # for our problem, N x len(true_x)
    return lps


def mh_step(params, context, target_fcn, **kwargs):
    z_to_use = params
    proposed_particles = D.Normal(z_to_use, 0.01).rsample()
    lps_curr = target_fcn(z_to_use)
    lps_new = target_fcn(proposed_particles)
    lp_ratios = lps_new - lps_curr
    lp_ratios = torch.exp(lp_ratios).clamp(min=0.0, max=1.0)
    flips = D.Bernoulli(lp_ratios).sample()
    indices_old = torch.arange(len(flips))[flips == 0]
    indices_new = torch.arange(len(flips))[flips == 1]
    new = torch.empty(proposed_particles.shape)
    new[indices_new] = proposed_particles[indices_new]
    new[indices_old] = z_to_use[indices_old]
    return new


def proposal(params, context, target_fcn, **kwargs):
    """
    Given a 'params' object of size N x c x ...,
    where N is the number of particles, c is current number
    of SMC steps, ... is remaining dimension.

    Returns propoal object q(z_{c+1} | z_{1:c})

    We propose using the current encoder q_\phi(z \mid x)

    We propose using most recent step z_{c-1} by random walk, i.e.
    q(z_{c+1} | z_{1:c}) = N(z_c, \sigma)
    """
    new = params
    for _ in range(100):
        new = mh_step(new, context, target_fcn, **kwargs)
    return new


def setup(cfg):
    # Problem Setup - Simple
    device = cfg.training.device
    k = cfg.data.k
    n = cfg.data.n
    A = torch.randn((n, k)) + torch.cat(
        [1e-2 * torch.eye(k), torch.zeros((n - k, k))], 0
    )
    sigma = cfg.data.sigma
    tau = cfg.data.tau
    prior = D.MultivariateNormal(torch.zeros(k), sigma**2 * torch.eye(k))
    n_pts = cfg.data.n_pts
    refresh_time = cfg.training.refresh_time

    zs, xs = generate_data(n, k, A, sigma, tau, prior, n_pts)
    posterior_means, posterior_cov = exact_posteriors(zs, xs, A, sigma, tau)
    posterior_means = posterior_means.to(device)
    posterior_cov = posterior_cov.to(device)

    log_string = "loss={},init={},rs={},k={},n={},lr={},sigma={},tau={},npts={},K={},refresh={},epochs={}".format(
        cfg.training.loss,
        cfg.smc.num_init,
        cfg.smc.n_resample,
        k,
        n,
        cfg.training.lr,
        sigma,
        tau,
        n_pts,
        cfg.smc.K,
        refresh_time,
        cfg.training.epochs,
    )

    # Instantiate Training
    encoder = FullRankEncoder(n, k, device).to(device)
    epochs = cfg.training.epochs
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)
    K = cfg.smc.K
    kwargs = {
        "A": A,
        "k": k,
        "n": n,
        "sigma": sigma,
        "n_pts": n_pts,
        "K": K,
        "tau": tau,
        "enc": encoder,
        "prior": prior,
        "device": device,
    }

    mb_size = cfg.training.mb_size

    # Select loss function
    loss_name = cfg.training.loss
    kwargs["loss_name"] = loss_name
    smc_meta_samplers = []

    if ("smc" in loss_name) and (cfg.smc.run):
        for j in range(len(xs)):
            init_samplers = []
            for n_samp in range(cfg.smc.num_init):
                # Set up SMC
                particles = prior.sample((K,))
                if loss_name == "vsmc":
                    particles.requires_grad_(True)
                particles = particles.unsqueeze(1)
                init_log_weights = torch.zeros((K, 1))
                init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
                final_target_fcn = lambda z: log_prior(z, **kwargs) + log_target(
                    z, xs[j], **kwargs
                )

                SMC = LikelihoodTemperedSMC(
                    particles,
                    init_weights,
                    init_log_weights,
                    final_target_fcn,
                    prior,
                    log_prior,
                    log_target,
                    proposal,
                    max_mc_steps=100,
                    context=xs[j],
                    kwargs=kwargs,
                )

                # Run SMC
                SMC.run()
                init_samplers.append(SMC)
            keep_all = (
                True
                if (
                    (cfg.training.loss == "smcwake_og")
                    or (cfg.training.loss == "smcwake1")
                    or (cfg.training.loss == "smcwake3")
                    or (cfg.training.loss == "smcwake_pimh")
                )
                else False
            )
            this_point_meta_IS = MetaImportanceSampler(
                init_samplers, xs, j, keep_all=keep_all, n_resample=cfg.smc.n_resample
            )
            smc_meta_samplers.append(this_point_meta_IS)

    # Set up Markov Chains for MSC
    old_particles = encoder.get_q(
        rearrange(xs, "n_obs d -> n_obs 1 d").to(device)
    ).sample()
    old_particles = rearrange(old_particles, "n_obs 1 k -> n_obs k")
    kwargs["old_particles"] = old_particles

    return (
        zs,
        xs,
        posterior_means,
        posterior_cov,
        log_string,
        encoder,
        epochs,
        optimizer,
        mb_size,
        smc_meta_samplers,
        refresh_time,
        device,
        kwargs,
    )
