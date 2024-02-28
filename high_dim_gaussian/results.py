import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import itertools
import json
import random

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from hydra import compose, initialize
from losses import get_imp_weights
from omegaconf import DictConfig, OmegaConf
from setup import log_prior, log_target, proposal, setup
from utils import exact_posteriors

from smc.smc_sampler import LikelihoodTemperedSMC

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 20})


@hydra.main(config_path="../conf", config_name="config_hdg")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config_hdg")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    # cfg.smc.run = False
    # cfg.smc.K = 100
    # cfg.training.epochs = 500000
    # cfg.training.full_rank = True
    # cfg.data.n_pts = 50
    # cfg.smc.num_init = 1
    # cfg.data.k = 50
    # cfg.data.n = 100
    # cfg.data.tau = 1
    # cfg.training.device = "cpu"

    (
        _,
        xs,
        posterior_means,
        posterior_cov,
        _,
        encoder,
        _,
        _,
        _,
        _,
        _,
        device,
        _,
    ) = setup(cfg)

    strings = [
        "loss=msc,init=1,rs=10,k=50,n=100,lr=0.0001,sigma=1,tau=1,npts=50,K=100,refresh=1,epochs=500000refreshtime=1fullrank",
        "loss=smcwake_pimh,init=1,rs=10,k=50,n=100,lr=0.0001,sigma=1,tau=1,npts=50,K=100,refresh=1,epochs=40000refreshtime=1fullrank",
    ]

    names = ["MSC", "SMC-PIMH"]

    # Results for FKL, RKL, L1 at specified iteration.
    iterations_to_use = [500000, 40000]
    exact_posts = D.MultivariateNormal(posterior_means, posterior_cov)
    results = {}
    for j in range(len(strings)):
        s = strings[j]
        iteration = iterations_to_use[j]
        try:
            encoder.load_state_dict(
                torch.load(
                    "./high_dim_gaussian/logs/{}/weights_{}_seed_{}.pth".format(
                        s, iteration, cfg.seed
                    )
                )
            )
        except FileNotFoundError:
            continue

        approx_posts = encoder.get_q(xs.unsqueeze(1).to(device))
        loc, cov = rearrange(approx_posts.loc, "b 1 k -> b k"), rearrange(
            approx_posts.covariance_matrix, "b 1 k1 k2 -> b k1 k2"
        )
        approx_posts = D.MultivariateNormal(loc, cov)
        l1 = (
            torch.sum(torch.abs((approx_posts.loc - posterior_means.to(device))), 1)
            .mean()
            .item()
        )
        fkl = D.kl.kl_divergence(exact_posts, approx_posts).mean().item()
        rkl = D.kl.kl_divergence(approx_posts, exact_posts).mean().item()
        skl = fkl + rkl
        results[names[j]] = [l1, fkl, rkl, skl]

    pd.options.display.float_format = "{:,.4f}".format
    results = pd.DataFrame(results)
    results.to_latex("./high_dim_gaussian/figs/results_{}.tex".format(cfg.seed))

    # Plot training curves using saved weights, every 5k iterations.
    l1s = {}
    fkls = {}
    for s in strings:
        l1s[s] = []
        fkls[s] = []
    for j in range(len(strings)):
        s = strings[j]
        iteration = iterations_to_use[j]
        for iter in range(5000, iteration + 1, 5000):
            try:
                encoder.load_state_dict(
                    torch.load(
                        "./high_dim_gaussian/logs/{}/weights_{}_seed_{}.pth".format(
                            s, iter, cfg.seed
                        )
                    )
                )
            except FileNotFoundError:
                continue

            approx_posts = encoder.get_q(xs.unsqueeze(1).to(device))
            loc, cov = rearrange(approx_posts.loc, "b 1 k -> b k"), rearrange(
                approx_posts.covariance_matrix, "b 1 k1 k2 -> b k1 k2"
            )
            approx_posts = D.MultivariateNormal(loc, cov)
            l1 = (
                torch.sum(torch.abs((approx_posts.loc - posterior_means.to(device))), 1)
                .mean()
                .item()
            )
            fkl = D.kl.kl_divergence(exact_posts, approx_posts).mean().item()
            rkl = D.kl.kl_divergence(approx_posts, exact_posts).mean().item()
            l1s[s].append(l1)
            fkls[s].append(fkl)

    fig, ax = plt.subplots(figsize=(20, 10))
    for j in range(len(strings[:1])):
        ax.plot(
            [5000 * x for x in range(len(l1s[strings[j]]))],
            l1s[strings[j]],
            label=names[j],
        )
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("L1 Distance")
    plt.tight_layout()
    plt.savefig("./high_dim_gaussian/figs/l1s_{}.png".format(cfg.seed), dpi=300)

    fig, ax = plt.subplots(figsize=(20, 10))
    for j in range(len(strings[:1])):
        ax.plot(
            [5000 * x for x in range(len(l1s[strings[j]]))],
            fkls[strings[j]],
            label=names[j],
        )
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Forward KL")
    plt.axhline(y=2000, color="r", linestyle="-")
    plt.tight_layout()
    plt.savefig("./high_dim_gaussian/figs/fkls_{}.png".format(cfg.seed), dpi=300)


if __name__ == "__main__":
    main()
