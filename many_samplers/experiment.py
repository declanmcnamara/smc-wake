import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys

sys.path.append("../")
sys.path.append("../high_dim_gaussian")
import math
import random
from os.path import exists

import hydra
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from generate import generate_data
from hydra import compose, initialize
from modules import Encoder, FullRankEncoder
from omegaconf import DictConfig
from setup import log_prior, log_target
from utils import exact_posteriors

from smc.smc_sampler import LikelihoodTemperedSMC, MetaImportanceSampler


def mh_step(params, context, target_fcn, **kwargs):
    z_to_use = params
    proposed_particles = D.Normal(z_to_use, 0.1).rsample()
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
    for _ in range(10):
        new = mh_step(new, context, target_fcn, **kwargs)
    return new


def setup(cfg):
    # Problem Setup - Simple
    device = cfg.training.device
    k = cfg.data.k
    n = cfg.data.n
    A = 10 * torch.randn((n, k)) + torch.cat(
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
                )
                else False
            )
            this_point_meta_IS = MetaImportanceSampler(
                init_samplers, xs, j, keep_all=keep_all, n_resample=cfg.smc.n_resample
            )
            smc_meta_samplers.append(this_point_meta_IS)

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


def my_smc_wake_big(xs, smc_samplers, **kwargs):
    # Choose data points
    device = kwargs["device"]
    mb_size = xs.shape[0]

    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = rearrange(xs[indices].to(device), "b d -> b 1 d")
    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    all_part_weights = [smc._get_recent() for smc in these_meta_samplers]
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K d -> K b d").to(device)
    weights = rearrange(all_weights, "b K -> b K").to(device)

    encoder = kwargs["enc"]

    # Compute Forward KL Loss
    lps = encoder.get_q(pts).log_prob(rearrange(particles, "K b k -> K b 1 k"))
    lps = rearrange(lps, "K b 1 -> K b")
    return -1 * torch.diag(weights @ lps).mean()


def my_smc_wake_mini(xs, smc_samplers, **kwargs):
    # Choose data points
    device = kwargs["device"]
    mb_size = xs.shape[0]
    enc = kwargs["enc"]

    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = rearrange(xs[indices].to(device), "b d -> b 1 d")
    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    # Sample and select particles, weights
    all_part_weights = [
        smc._get_random_subset_weights_particles() for smc in these_meta_samplers
    ]
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K r d -> K r b d")
    weights = rearrange(all_weights, "b r K -> b r K").to(device)

    n_samplers_all = particles.shape[1]
    rpts = repeat(pts, "b 1 dim -> b nsamp dim", nsamp=n_samplers_all)
    to_eval = rearrange(particles, "K nsamp b k -> K b nsamp k").to(device)
    lps = rearrange(enc.get_q(rpts).log_prob(to_eval), "K b r -> b K r")
    mms = torch.bmm(weights, lps)
    diags = torch.diagonal(mms, dim1=-2, dim2=-1)

    return -1 * diags.mean()


@hydra.main(config_path="../conf", config_name="config_many")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="config_many")
    # cfg = compose(config_name="config_many")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (
        zs,
        xs,
        posterior_means,
        posterior_cov,
        log_string,
        encoder,
        epochs,
        optimizer,
        mb_size,
        smc_samplers,
        refresh_time,
        device,
        kwargs,
    ) = setup(cfg)

    l1s = []
    fkls = []
    rkls = []
    losses = []
    exact_posts = D.MultivariateNormal(posterior_means, posterior_cov)

    suffix = "fullrank" if cfg.training.full_rank else ""
    log_string = log_string + "refreshtime={}{}".format(refresh_time, suffix)

    biglittle = "little" if cfg.smc.num_init > 1 else "big"

    for j in range(epochs):
        # Take a gradient step
        optimizer.zero_grad()
        loss = (
            my_smc_wake_mini(xs, smc_samplers, **kwargs)
            if cfg.smc.num_init > 1
            else my_smc_wake_big(xs, smc_samplers, **kwargs)
        )
        loss.backward(retain_graph=True)
        optimizer.step()

        # Print update
        print("On iteration {}, loss is {}".format(j, loss.item()))
        approx_posts = encoder.get_q(xs.unsqueeze(1).to(device))
        loc, cov = reduce(approx_posts.loc, "b r k -> b k", "mean"), reduce(
            approx_posts.covariance_matrix, "b r k1 k2 -> b k1 k2", "mean"
        )
        approx_posts = D.MultivariateNormal(loc, cov)

        l1s.append(
            torch.sum(torch.abs((approx_posts.loc - posterior_means.to(device))), 1)
            .mean()
            .item()
        )
        fkls.append(D.kl.kl_divergence(exact_posts, approx_posts).mean().item())
        rkls.append(D.kl.kl_divergence(approx_posts, exact_posts).mean().item())
        losses.append(loss.item())

        del loss
        del approx_posts

        if (j + 1) % 5000 == 0:
            torch.save(
                kwargs["enc"].state_dict(),
                "./many_samplers/logs/{}{}_weights.pth".format(biglittle, log_string),
            )
            np.save(
                "./many_samplers/logs/{}{}_l1srw.npy".format(biglittle, log_string),
                np.array(l1s),
            )
            np.save(
                "./many_samplers/logs/{}{}_fklsrw.npy".format(biglittle, log_string),
                np.array(fkls),
            )
            np.save(
                "./many_samplers/logs/{}{}_rklsrw.npy".format(biglittle, log_string),
                np.array(rkls),
            )
            np.save(
                "./many_samplers/logs/{}{}_lossesrw.npy".format(biglittle, log_string),
                np.array(losses),
            )


if __name__ == "__main__":
    main()
