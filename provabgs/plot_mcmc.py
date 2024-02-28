import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1
import sys

sys.path.append("../")
import random

import hydra

# -- plotting --
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from setup import log_t_prior, log_target, mh_step, setup
from utils import construct_one_smc_sampler, prior_t_sample, transform_thetas

plt.rcParams.update({"font.size": 52})


def mcmc_mh(
    thetas,
    seds,
    index,
    logger_string,
    encoder,
    truth_start=False,
    burn_in=10000,
    n_walkers=100,
    n_steps=30000,
    **kwargs
):
    # Run MCMC, Metropolis Hastings
    if truth_start:
        init_particles = thetas[index].to("cpu").repeat(n_walkers, 1)  # at ground truth
    else:
        init_particles = prior_t_sample(100, **kwargs)

    my_target_fcn = lambda z_to_use: log_t_prior(z_to_use, **kwargs) + log_target(
        z_to_use, seds[index].float().to("cpu"), **kwargs
    )

    all_results = init_particles.unsqueeze(0)  # 1 step x 100 walker x dim
    start = torch.clone(all_results)
    for iteration in range(n_steps + burn_in):
        if iteration % 25 == 0:
            print("On iteration {}.".format(iteration))
        next_step = mh_step(
            start.squeeze(0), seds[index].float().to("cpu"), my_target_fcn, **kwargs
        )
        start = torch.clone(next_step)
        all_results = torch.cat([all_results, next_step.unsqueeze(0)], dim=0)

    all_results = all_results[burn_in:]
    flattened_chain = all_results.flatten(end_dim=1)
    flattened_true_scale = (
        transform_thetas(flattened_chain, **kwargs).cpu().detach().numpy()
    )

    return flattened_true_scale


def plot_true_scale(
    log_name,
    flattened_true_scale,
    thetas,
    seds,
    index,
    logger_string,
    encoder,
    truth_start=False,
    burn_in=10000,
    n_walkers=100,
    n_steps=30000,
    **kwargs
):

    samples, lps = encoder.sample_and_log_prob(
        100000, seds[index].view(1, -1).float().to(kwargs["device"])
    )
    samples = samples.squeeze(0)
    tthetas = transform_thetas(samples, **kwargs).detach().numpy()
    truth = (
        transform_thetas(thetas[index].view(1, -1), **kwargs)
        .view(-1)
        .cpu()
        .detach()
        .numpy()
    )

    MCMC = pd.DataFrame(flattened_true_scale)
    FLOW = pd.DataFrame(tthetas)
    MCMC["Type"] = "MCMC"
    FLOW["Type"] = "Flow"
    big_df = pd.concat([MCMC, FLOW], ignore_index=True)

    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(50, 100))
    for i in range(6):
        for j in range(2):
            if (i == 5) and (j == 1):
                break
            this_index = 2 * i + j
            sns.kdeplot(
                big_df[[this_index, "Type"]],
                x=this_index,
                hue="Type",
                bw_adjust=0.2,
                ax=ax[i, j],
                common_norm=False,
                palette={"MCMC": sns.xkcd_rgb["bright orange"], "Flow": "blue"},
                legend=False,
            )
            ax[i, j].axvline(x=truth[this_index], color="r")

    ax[5, 1].remove()

    plt.savefig(
        "./provabgs/figs/{}mcmc_{}_{}_truth={}.png".format(
            log_name, index, logger_string, truth_start
        ),
        bbox_inches="tight",
    )


def plot_true_scale4smcwake(
    log_name,
    flattened_true_scale,
    thetas,
    seds,
    index,
    logger_string,
    encoder,
    string1,
    iteration,
    truth_start=False,
    **kwargs
):

    encoder.load_state_dict(
        torch.load(
            "./provabgs/logs/{}/encoder{}.pth".format(string1, iteration),
            map_location=kwargs["device"],
        )
    )
    samples, lps = encoder.sample_and_log_prob(
        100000, seds[index].view(1, -1).float().to(kwargs["device"])
    )
    samples = samples.squeeze(0)
    tthetas = transform_thetas(samples, **kwargs).detach().numpy()
    truth = (
        transform_thetas(thetas[index].view(1, -1), **kwargs)
        .view(-1)
        .cpu()
        .detach()
        .numpy()
    )

    MCMC = pd.DataFrame(flattened_true_scale)
    FLOW = pd.DataFrame(tthetas)
    MCMC["Type"] = "MCMC"
    FLOW["Type"] = "SMC-Wake"
    big_df = pd.concat([MCMC, FLOW], ignore_index=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(50, 50))
    indices_to_plot = [0, 5, 9, 10]
    for subfig in range(4):
        i = subfig // 2
        j = subfig % 2
        this_index = indices_to_plot[subfig]
        sns.kdeplot(
            big_df[[this_index, "Type"]],
            x=this_index,
            hue="Type",
            bw_adjust=0.2,
            ax=ax[i, j],
            common_norm=False,
            palette={"MCMC": sns.xkcd_rgb["green"], "SMC-Wake": sns.xkcd_rgb["blue"]},
            legend=False,
        )
        ax[i, j].axvline(x=truth[this_index], color=sns.xkcd_rgb["red"])

    plt.savefig(
        "./provabgs/figs/{}mcmc4_{}_{}_truth={}.png".format(
            log_name, index, logger_string, truth_start
        ),
        bbox_inches="tight",
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config_provabgs")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config3")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (
        thetas,
        seds,
        epochs,
        device,
        mb_size,
        encoder,
        smc_meta_samplers,
        mdn,
        flow,
        logger_string,
        optimizer_encoder,
        refresh_every,
        kwargs,
    ) = setup(cfg)

    # Load state dict
    iteration = 25000
    device = torch.device("cpu")

    kwargs["device"] = "cpu"
    kwargs["model"] = kwargs["model"].to("cpu")
    kwargs.pop("encoder")
    encoder.eval()

    burn_in = 10000
    n_steps = 10000

    strings = [
        "smcwake_a,flow,0.0001,10,mult=0.1,smooth=False,refresh=50",
    ]

    names = [""]

    for string_index in range(len(strings)):
        s = strings[string_index]
        encoder.load_state_dict(
            torch.load(
                "./provabgs/logs/{}/encoder{}.pth".format(s, iteration),
                map_location=device,
            )
        )
        encoder = encoder.to("cpu")
        name = names[string_index]
        for j in cfg.plots.flow_and_mcmc.points:
            try:
                flat_chain = mcmc_mh(
                    thetas,
                    seds,
                    j,
                    logger_string,
                    encoder,
                    truth_start=False,
                    burn_in=burn_in,
                    n_steps=n_steps,
                    **kwargs
                )
                plot_true_scale(
                    name,
                    flat_chain,
                    thetas,
                    seds,
                    j,
                    logger_string,
                    encoder,
                    truth_start=False,
                    burn_in=burn_in,
                    n_steps=n_steps,
                    **kwargs
                )
                plot_true_scale4smcwake(
                    "smcwake_a_4panel",
                    flat_chain,
                    thetas,
                    seds,
                    j,
                    logger_string,
                    encoder,
                    strings[0],
                    iteration,
                    truth_start=False,
                    **kwargs
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
