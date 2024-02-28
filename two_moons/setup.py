import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import sys

sys.path.append("../")
import math
import random

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from generate import generate_data, generate_data_favi
from hydra import compose, initialize
from omegaconf import DictConfig

from cde.nsf import build_nsf
from smc.smc_sampler import LikelihoodTemperedSMC, MetaImportanceSampler


def log_target(params, context, **kwargs):
    """
    Given a 'params' object of size N x c x ...,
    where N is the number of particles, c is current number
    of SMC steps, ... is remaining dimension.

    Returns log p(x | z_c), for likelihood tempered SMC. In other
    words, we ignore z_1, ..., z_{c-1} and only use the most recent
    stage of the params input.
    """
    r_dist = kwargs["r_dist"]
    theta = params
    new1 = -1 * torch.sum(theta, 1).abs() / math.sqrt(2)
    new2 = (-1 * theta[:, 0] + theta[:, 1]) / math.sqrt(2)
    new = torch.stack([new1, new2]).T
    p = context - new
    u = p[:, 0] - 0.25
    v = p[:, 1]
    r = torch.sqrt(u**2 + v**2)  # note the angle distribution is uniform
    to_adjust = r_dist.log_prob(r)
    adjusted = torch.where(u < 0.0, -torch.inf, to_adjust.double())

    return adjusted


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
    theta = params
    lps = prior.log_prob(theta)  # for our problem, N x len(true_x)
    return lps.sum(-1)


def mh_step(params, context, target_fcn, **kwargs):
    z_to_use = params
    proposed_particles = (
        D.Normal(z_to_use, 0.1)
        .sample()
        .clamp(min=-1 * kwargs["val"], max=kwargs["val"])
    )
    lps_curr = torch.nan_to_num(target_fcn(z_to_use), -torch.inf)
    lps_new = torch.nan_to_num(target_fcn(proposed_particles), -torch.inf)
    lp_ratios = torch.nan_to_num(lps_new - lps_curr, -torch.inf)
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
    for _ in range(5):
        new = mh_step(new, context, target_fcn, **kwargs)
    return new


def setup(cfg: DictConfig):

    K = cfg.smc.K
    refresh_every = cfg.smc.refresh_rate
    broad = cfg.data.bigger_range
    small = cfg.data.smaller_range
    if cfg.data.prior_broad:
        prior = D.Uniform(torch.tensor([-broad, -broad]), torch.tensor([broad, broad]))
        val = broad
    else:
        prior = D.Uniform(torch.tensor([-small, -small]), torch.tensor([small, small]))
        val = small

    a_dist = D.Uniform(-math.pi / 2, math.pi / 2)
    r_dist = D.Normal(0.1, 0.01)

    kwargs = {
        "K": K,
        "prior": prior,
        "a_dist": a_dist,
        "r_dist": r_dist,
        "val": val,
        "wake_range": cfg.plots.wake_range,
        "num_panels": cfg.plots.num_panels,
    }

    epochs = cfg.training.epochs
    device = cfg.training.device
    mb_size = cfg.training.mb_size

    kwargs["mb_size"] = mb_size
    kwargs["device"] = device

    theta, x = generate_data_favi(cfg.data.n_pts, return_theta=True, **kwargs)

    # EXAMPLE BATCH FOR SHAPES
    z_dim = prior.sample().shape[-1]
    x_dim = x.shape[-1]
    fake_zs = torch.randn((K * mb_size, z_dim))
    fake_xs = torch.randn((K * mb_size, x_dim))
    encoder = build_nsf(fake_zs, fake_xs, z_score_x="none", z_score_y="none")

    kwargs["encoder"] = encoder
    name = "flow" if cfg.encoder.type == "flow" else "mdn"
    if name == "flow":
        mdn = False
        flow = True
    else:
        mdn = True
        flow = False
    kwargs["mdn"] = mdn
    kwargs["flow"] = flow
    logger_string = "{},{},{},{},{},{}".format(
        cfg.training.loss, name, cfg.training.lr, K, refresh_every, val
    )
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)

    # Select loss function
    loss_name = cfg.training.loss
    kwargs["loss"] = loss_name
    smc_meta_samplers = []

    if ("smc" in loss_name) and (cfg.smc.run):
        for j in range(len(x)):
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
                    z, x[j].cpu(), **kwargs
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
                    context=x[j].cpu(),
                    kwargs=kwargs,
                )

                # Run SMC
                SMC.run()
                init_samplers.append(SMC)
            keep_all = (
                True
                if (
                    (cfg.training.loss == "smcwake_naive")
                    or (cfg.training.loss == "smcwake_a")
                    or (cfg.training.loss == "smcwake_c")
                )
                else False
            )
            this_point_meta_IS = MetaImportanceSampler(
                init_samplers, x, j, keep_all=keep_all, n_resample=cfg.smc.n_resample
            )
            smc_meta_samplers.append(this_point_meta_IS)

    return (
        theta,
        x,
        K,
        prior,
        a_dist,
        r_dist,
        smc_meta_samplers,
        epochs,
        device,
        mb_size,
        encoder,
        name,
        mdn,
        flow,
        logger_string,
        encoder,
        optimizer,
        refresh_every,
        kwargs,
    )
