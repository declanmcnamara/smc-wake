import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1
import sys

sys.path.append("../")
# -- plotting --
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["axes.xmargin"] = 1
mpl.rcParams["xtick.labelsize"] = "x-large"
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["xtick.major.width"] = 1.5
mpl.rcParams["ytick.labelsize"] = "x-large"
mpl.rcParams["ytick.major.size"] = 5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["legend.frameon"] = False
import random
from os.path import exists

import hydra
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.nn as nn
from hydra import compose, initialize
from losses import smc_wake_loss_a
from omegaconf import DictConfig
from setup import log_target, proposal, setup
from utils import log_t_prior, prior_t_sample

from smc.smc_sampler import LikelihoodTemperedSMC


def refresh_samplers(seds, curr_samplers, **kwargs):
    K = kwargs["K"]
    n_pts = seds.shape[0]
    random_index = torch.randint(low=0, high=n_pts, size=(1,))[0].item()
    print("Replacing index {}".format(random_index))

    particles = prior_t_sample(K, **kwargs)
    particles = particles.unsqueeze(1)
    init_log_weights = torch.zeros((K, 1))
    init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
    final_target_fcn = lambda z: log_t_prior(z, **kwargs) + log_target(
        z, seds[random_index].cpu(), **kwargs
    )

    SMC = LikelihoodTemperedSMC(
        particles,
        init_weights,
        init_log_weights,
        final_target_fcn,
        None,
        log_t_prior,
        log_target,
        proposal,
        max_mc_steps=100,
        context=seds[random_index].cpu(),
        z_min=kwargs["z_min"],
        z_max=kwargs["z_max"],
        kwargs=kwargs,
    )
    SMC.run()

    curr_samplers[random_index]._append(SMC, random_index)

    return curr_samplers


def loss_choice(seds, smc_samplers, **kwargs):
    loss_name = kwargs["loss"]
    if loss_name == "smcwake_a":
        return smc_wake_loss_a(seds, smc_samplers, **kwargs)
    else:
        raise ValueError("Specify an appropriate loss name string.")


@hydra.main(version_base=None, config_path="../conf", config_name="config_provabgs")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config_provabgs")
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
        smc_samplers,
        mdn,
        flow,
        logger_string,
        optimizer_encoder,
        refresh_every,
        kwargs,
    ) = setup(cfg)

    if not exists("./provabgs/logs/{}".format(logger_string)):
        os.mkdir("./provabgs/logs/{}".format(logger_string))

    for j in range(epochs):
        if j % 1000 == 0:
            print("On iteration {}".format(j))

        # Update encoder
        optimizer_encoder.zero_grad()
        try:
            loss = loss_choice(seds, smc_samplers, **kwargs)
            print("Encoder Loss iter {} is {}".format(j, loss.item()))
        except Exception:
            continue
        if torch.isnan(loss).any():
            del loss
            continue
        loss.backward()
        optimizer_encoder.step()
        del loss

        # Update SMC samplers
        if ("smc" in kwargs["loss"]) and ((j + 1) % refresh_every == 0):
            smc_samplers = refresh_samplers(seds, smc_samplers, **kwargs)

        save_every_all = 5000
        if (j + 1) % save_every_all == 0:
            torch.save(
                encoder.state_dict(),
                "./provabgs/logs/{}/encoder{}.pth".format(logger_string, j + 1),
            )


if __name__ == "__main__":
    main()
