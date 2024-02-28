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

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from hydra import compose, initialize
from modules import SBN, Encoder
from omegaconf import DictConfig
from scipy.stats.kde import gaussian_kde

from cde.nsf import build_nsf
from smc.smc_sampler import LikelihoodTemperedSMC, MetaImportanceSampler


def log_target(params, context, **kwargs):
    model = kwargs["model"]
    device = kwargs["device"]
    kwargs["model"] = kwargs["model"].to("cpu")
    image_part, label_part = context[..., :-1], context[..., -1:]
    to_return = model.log_prob(
        params.cpu(), label_part.long().cpu(), image_part.cpu()
    ).detach()
    kwargs["model"] = kwargs["model"].to(device)
    return to_return.detach()  # .cpu()


def log_prior(params, **kwargs):
    prior = kwargs["prior"]
    device = kwargs["device"]
    return prior.log_prob(params.to(device)).sum(-1).cpu()


def mh_step(params, context, target_fcn, **kwargs):
    z_to_use = params
    proposed_particles = D.Normal(z_to_use, 0.1).sample()
    lps_curr = torch.nan_to_num(target_fcn(z_to_use), -torch.inf)
    lps_new = torch.nan_to_num(target_fcn(proposed_particles), -torch.inf)
    lp_ratios = torch.nan_to_num(lps_new - lps_curr, -torch.inf)
    lp_ratios = torch.exp(lp_ratios).clamp(min=0.0, max=1.0)
    flips = D.Bernoulli(lp_ratios).sample()
    indices_old = torch.arange(len(flips))[flips == 0]
    indices_new = torch.arange(len(flips))[flips == 1]
    new = torch.empty(proposed_particles.shape)  # .to(kwargs['device'])
    new[indices_new] = proposed_particles[indices_new]
    new[indices_old] = z_to_use[indices_old]
    return new


def proposal(params, context, target_fcn, **kwargs):
    new = params
    for _ in range(5):
        new = mh_step(new, context, target_fcn, **kwargs)
    return new


def setup(cfg: DictConfig):
    latent_dim = cfg.data.latent_dim
    device = cfg.training.device
    true_x = torch.load("./normalized_mnist/data/continuous_data_BIG.pt")
    true_x = torch.flatten(true_x, start_dim=1)
    true_digits = torch.load("./normalized_mnist/data/labels_BIG.pt")
    true_data = torch.cat([true_x, true_digits.unsqueeze(-1)], -1)
    data_dim = true_data.shape[-1]

    K = cfg.smc.K
    loss_name = cfg.training.loss

    true_x = true_x.to(device)
    prior = D.Normal(torch.zeros(latent_dim).to(device), cfg.data.prior_dispersion)

    kwargs = {
        "K": K,
        "loss_name": loss_name,
        "latent_dim": latent_dim,
        "data_dim": data_dim,
        "prior": prior,
    }

    epochs = cfg.training.epochs
    device = cfg.training.device
    mb_size = cfg.training.mb_size

    kwargs["mb_size"] = mb_size
    kwargs["device"] = device

    encoder = Encoder(data_dim, latent_dim, cfg.training.hidden_dim)

    kwargs["encoder"] = encoder
    logger_string = "{},{},{},{},{},{},{}_digits_".format(
        cfg.training.loss,
        cfg.training.lr,
        K,
        latent_dim,
        cfg.training.prior_refresh,
        mb_size,
        cfg.data.prior_dispersion,
    )
    encoder.to(device)
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)

    model = SBN(latent_dim, true_x.shape[-1])
    model.to(device)
    kwargs["model"] = model
    optimizer_model = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    smc_meta_samplers = []
    if "smc" in loss_name:
        for j in range(len(true_x)):
            init_samplers = []
            print("On point j={}".format(j))
            for n_samp in range(cfg.smc.num_init):
                # Set up SMC
                particles = prior.sample((K,)).cpu()
                particles = particles.unsqueeze(1)
                init_log_weights = torch.zeros((K, 1))
                init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
                final_target_fcn = lambda z: log_prior(z, **kwargs).cpu() + log_target(
                    z, true_data[j].cpu(), **kwargs
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
                    context=true_data[j].cpu(),
                    kwargs=kwargs,
                )
                SMC.run()
                init_samplers.append(SMC)
            keep_all = False
            this_point_meta_IS = MetaImportanceSampler(
                init_samplers,
                true_data,
                j,
                keep_all=keep_all,
                n_resample=cfg.smc.n_resample,
            )
            smc_meta_samplers.append(this_point_meta_IS)

    return (
        true_x,
        true_data,
        true_digits,
        K,
        prior,
        smc_meta_samplers,
        epochs,
        device,
        mb_size,
        encoder,
        optimizer_encoder,
        logger_string,
        model,
        optimizer_model,
        kwargs,
    )
