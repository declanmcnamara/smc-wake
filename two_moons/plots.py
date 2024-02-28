import math
import os
import random
from operator import add

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import torch
import torch.distributions as D
import torch.nn as nn
from generate import generate_data
from hydra import compose, initialize
from omegaconf import DictConfig
from setup import setup
from utils import log_post

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 100})


def wake_panels(cfg, j, x, **kwargs):
    device = kwargs["device"]
    K = kwargs["K"]
    encoder = kwargs["encoder"]
    encoder = encoder.to(device)

    num_panels = kwargs["num_panels"]
    side = int(math.sqrt(num_panels))
    wake_range = kwargs["wake_range"]
    every = wake_range // num_panels

    logger_string = "{},{},{},{},{},{}".format(
        "wake", "flow", cfg.training.lr, K, cfg.smc.refresh_rate, kwargs["val"]
    )

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    for row in range(side):
        for col in range(side):
            index = (side * row) + col
            suffix = (index + 1) * every
            to_load = "./two_moons/logs/{}/encoder{}.pth".format(logger_string, suffix)
            encoder.load_state_dict(torch.load(to_load))
            n_samp_empirical = 10000
            particles, log_denoms = encoder.sample_and_log_prob(
                num_samples=K, context=x[j].view(1, -1).to(device)
            )
            particles = particles.view(K, -1).clamp(-1.0, 1.0)
            log_denoms = log_denoms.view(-1)
            log_nums = log_post(x[j].to(device), particles, **kwargs)
            log_weights = log_nums - log_denoms
            weights = nn.Softmax(0)(log_weights)
            weights = weights.detach()
            indices = torch.multinomial(weights, n_samp_empirical, replacement=True)
            samples = particles[indices]
            ax[row, col].hist2d(
                samples[:, 0].detach().cpu().numpy(),
                samples[:, 1].detach().cpu().numpy(),
                bins=100,
                range=[[-1.0, 1.0], [-1.0, 1.0]],
                density=True,
            )
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            # ax[row,col].set_title('IS Approx. K={}'.format(K))
    plt.tight_layout()
    plt.savefig("./two_moons/figs/panels{}.png".format(logger_string))

    return fig


def plot_comparison(cfg, index, x, **kwargs):
    losses = cfg.plots.losses
    mapper = {
        "smcwake_naive": "SMC-Wake (Naive)",
        "smcwake_a": "SMC-Wake(a)",
        "smcwake_b": "SMC-Wake(b)",
        "smcwake_c": "SMC-Wake(c)",
        "wake": "Wake",
        "prior_wake": "P-Wake",
        "defensive_wake": "Defensive Wake",
    }
    encoder = kwargs["encoder"]
    names = list(map(lambda name: mapper[name], losses))
    device = kwargs["device"]
    K = kwargs["K"]

    ncol = 3 if len(losses) > 3 else 2
    figsize = (50, 30) if len(losses) > 3 else (50, 50)

    nrows_to_use = math.ceil(len(losses) / ncol)
    fig, ax = plt.subplots(nrows=nrows_to_use, ncols=ncol, figsize=figsize)

    # Plot the exact posterior
    val = kwargs["val"]
    vals = torch.arange(-val, val, 0.01)
    eval_pts = torch.cartesian_prod(vals, vals)
    lps = log_post(x[index], eval_pts, **kwargs)
    X, Y = torch.meshgrid(vals, vals)
    Z = lps.view(X.shape)
    ax[nrows_to_use - 1, ncol - 1].pcolormesh(
        X.numpy(), Y.numpy(), Z.exp().numpy(), vmax=50.0
    )
    ax[nrows_to_use - 1, ncol - 1].set_title("Exact", fontsize=75)

    iteration = 50000
    for j in range(len(losses)):
        loss = losses[j]
        print("Plotting some images, loss is {}".format(loss))
        logger_string = "{},{},{},{},{},{}".format(
            loss, "flow", cfg.training.lr, K, cfg.smc.refresh_rate, kwargs["val"]
        )
        encoder.load_state_dict(
            torch.load(
                "./two_moons/logs/{}/encoder{}.pth".format(logger_string, iteration),
                map_location=device,
            )
        )
        encoder = encoder.to(device)
        kwargs["encoder"] = encoder

        if loss == "wake":
            vals = torch.arange(-val, val, 0.03)
            eval_pts = torch.cartesian_prod(vals, vals)
            lps = encoder.log_prob(
                eval_pts.to(device),
                x[index].view(1, -1).repeat(eval_pts.shape[0], 1).to(device),
            ).detach()
            X, Y = torch.meshgrid(vals, vals)
            Z = lps.view(X.shape)
            row, col = j // ncol, j % ncol
            ax[row, col].pcolormesh(
                X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().exp().numpy()
            )
            ax[row, col].set_title("{}".format(names[j]), fontsize=75)
            ax[row, col].set_xticklabels([])
            ax[row, col].set_yticklabels([])
        else:
            vals = torch.arange(-val, val, 0.01)
            eval_pts = torch.cartesian_prod(vals, vals)
            lps = encoder.log_prob(
                eval_pts.to(device),
                x[index].view(1, -1).repeat(eval_pts.shape[0], 1).to(device),
            ).detach()
            X, Y = torch.meshgrid(vals, vals)
            Z = lps.view(X.shape)
            row, col = j // ncol, j % ncol
            ax[row, col].pcolormesh(
                X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().exp().numpy(), vmax=50.0
            )
            ax[row, col].set_title("{}".format(names[j]), fontsize=75)
            ax[row, col].set_xticklabels([])
            ax[row, col].set_yticklabels([])

    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        "./two_moons/figs/{}_comparison{},{},{},{}ncol={}.png".format(
            index, cfg.training.lr, K, cfg.smc.refresh_rate, kwargs["val"], ncol
        ),
        bbox_inches="tight",
    )
    return


@hydra.main(version_base=None, config_path="../conf", config_name="config_two_moons")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config2")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (
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
    ) = setup(cfg)

    kwargs["device"] = "cuda:0"

    wake_panels(cfg, 0, x, **kwargs)
    for index in cfg.plots.indices:
        plot_comparison(cfg, index, x, **kwargs)


if __name__ == "__main__":
    main()
