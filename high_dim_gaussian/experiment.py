import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys

sys.path.append("../")
import json
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
from losses import markovian_score_climbing_loss, smc_wake_loss
from modules import Encoder, FullRankEncoder
from omegaconf import DictConfig
from scipy.linalg import eig
from setup import log_prior, log_target, proposal, setup
from utils import exact_posteriors

from smc.smc_sampler import LikelihoodTemperedSMC


def refresh_samplers(xs, samplers, **kwargs):
    n_obs = xs.shape[0]
    prior = kwargs["prior"]
    K = kwargs["K"]
    random_index = torch.randint(low=0, high=n_obs, size=(1,))[0].item()
    particles = prior.sample((K,))
    particles = particles.unsqueeze(1)
    init_log_weights = torch.zeros((K, 1))
    init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
    final_target_fcn = lambda z: log_prior(z, **kwargs) + log_target(
        z, xs[random_index], **kwargs
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
        context=xs[random_index],
        kwargs=kwargs,
    )

    # Run SMC
    SMC.run()

    if kwargs["loss_name"] == "smcwake_pimh":
        samplers[random_index]._append_pimh(SMC, random_index)
    else:
        samplers[random_index]._append(SMC, random_index)
    print("replaced index {}".format(random_index))
    return samplers, len(SMC.tau_list)


def loss_choice(j, xs, mb_size, samplers, **kwargs):
    loss_name = kwargs["loss_name"]
    if loss_name == "smcwake_pimh":
        return smc_wake_loss(j, xs, mb_size, samplers, **kwargs)
    elif loss_name == "msc":
        loss, sel_particles, sel_indices = markovian_score_climbing_loss(
            j, xs, mb_size, samplers, **kwargs
        )
        kwargs["old_particles"][sel_indices] = sel_particles
        return loss
    else:
        raise ValueError("Specify an appropriate loss name string.")


@hydra.main(config_path="../conf", config_name="config_hdg")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config_hdg")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    dir = cfg.dir
    os.chdir(dir)

    # cfg.training.loss='msc'
    # cfg.training.refresh_time=1
    # cfg.smc.K=1000
    # cfg.training.epochs=500000
    # cfg.training.full_rank=True
    # cfg.data.n_pts=50
    # cfg.smc.num_init=1
    # cfg.data.k=50
    # cfg.data.n=100
    # cfg.data.tau=1
    # cfg.training.device='cuda:6'

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

    suffix = "fullrank" if cfg.training.full_rank else ""
    log_string = log_string + "refreshtime={}{}".format(refresh_time, suffix)

    if not exists("./high_dim_gaussian/logs/{}".format(log_string)):
        os.mkdir("./high_dim_gaussian/logs/{}".format(log_string))

    for j in range(epochs):
        # Take a gradient step
        optimizer.zero_grad()
        loss = loss_choice(j, xs, mb_size, smc_samplers, **kwargs)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Print update
        print("On iteration {}, loss is {}".format(j, loss.item()))
        del loss

        if ("smc" in kwargs["loss_name"]) and ((j + 1) % refresh_time) == 0:
            smc_samplers, temp_steps = refresh_samplers(xs, smc_samplers, **kwargs)

        if (j + 1) % 5000 == 0:
            torch.save(
                kwargs["enc"].state_dict(),
                "./high_dim_gaussian/logs/{}/weights_{}_seed_{}.pth".format(
                    log_string, j + 1, seed
                ),
            )


if __name__ == "__main__":
    main()
