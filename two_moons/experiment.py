import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys

sys.path.append("../")
import random
from os.path import exists

import hydra
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from hydra import compose, initialize
from losses import (
    smc_wake_loss_a,
    smc_wake_loss_b,
    smc_wake_loss_c,
    smc_wake_loss_naive,
    wake_loss,
)
from omegaconf import DictConfig
from setup import log_prior, log_target, proposal, setup
from torch.utils.tensorboard import SummaryWriter

from smc.smc_sampler import LikelihoodTemperedSMC


def refresh_samplers(x, curr_samplers, **kwargs):
    prior = kwargs["prior"]
    K = kwargs["K"]
    n_pts = x.shape[0]
    random_index = torch.randint(low=0, high=n_pts, size=(1,))[0].item()
    print("Replacing index {}".format(random_index))

    particles = prior.sample((K,))
    particles = particles.unsqueeze(1)
    init_log_weights = torch.zeros((K, 1))
    init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
    final_target_fcn = lambda z: log_prior(z, **kwargs) + log_target(
        z, x[random_index].cpu(), **kwargs
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
        context=x[random_index].cpu(),
        kwargs=kwargs,
    )
    SMC.run()

    curr_samplers[random_index]._append(SMC, random_index)

    return curr_samplers


def loss_choice(x, smc_samplers, **kwargs):
    loss_name = kwargs["loss"]
    if loss_name == "smcwake_naive":
        return smc_wake_loss_naive(x, smc_samplers, **kwargs)
    elif loss_name == "smcwake_a":
        return smc_wake_loss_a(x, smc_samplers, **kwargs)
    elif loss_name == "smcwake_b":
        return smc_wake_loss_b(x, smc_samplers, **kwargs)
    elif loss_name == "smcwake_c":
        return smc_wake_loss_c(x, smc_samplers, **kwargs)
    elif loss_name == "wake":
        return wake_loss(x, **kwargs)
    elif loss_name == "defensive_wake":
        return wake_loss(x, prop_prior=0.5, **kwargs)
    else:
        raise ValueError("Specify an appropriate loss name string.")


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
        smc_samplers,
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

    loss_name = kwargs["loss"]
    losses = []

    if not exists("./two_moons/logs/{}".format(logger_string)):
        os.mkdir("./two_moons/logs/{}".format(logger_string))

    for j in range(epochs):
        if j % 1000 == 0:
            print("On iteration {}".format(j))

        optimizer.zero_grad()
        loss = loss_choice(x, smc_samplers, **kwargs)
        print("Loss iter {} is {}".format(j, loss))
        if torch.isnan(loss).any():
            continue
        loss.backward()
        optimizer.step()
        # scheduler.step()

        losses.append(loss.item())

        # Update SMC samplers
        if ("smc" in kwargs["loss"]) and ((j + 1) % refresh_every == 0):
            smc_samplers = refresh_samplers(x, smc_samplers, **kwargs)

        # Save weights for wake variants to look at later
        save_every = kwargs["wake_range"] // kwargs["num_panels"]
        max_save = kwargs["wake_range"]
        if (
            ((loss_name == "wake") or (loss_name == "defensive_wake"))
            and (((j + 1) % save_every) == 0)
            and ((j + 1) < max_save)
        ):
            torch.save(
                encoder.state_dict(),
                "./two_moons/logs/{}/encoder{}.pth".format(logger_string, (j + 1)),
            )

        save_every_all = 5000
        if (j + 1) % save_every_all == 0:
            torch.save(
                encoder.state_dict(),
                "./two_moons/logs/{}/encoder{}.pth".format(logger_string, j + 1),
            )


if __name__ == "__main__":
    main()
