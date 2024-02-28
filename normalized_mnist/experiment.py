import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys
from os.path import exists

sys.path.append("../")
import random

import hydra
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from hydra import compose, initialize
from losses import iwbo_loss, smc_wake_loss2, wake_loss
from omegaconf import DictConfig
from setup import log_prior, log_target, proposal, setup

from smc.smc_sampler import LikelihoodTemperedSMC


def refresh_samplers_prior(x, curr_samplers, **kwargs):
    prior = kwargs["prior"]
    K = kwargs["K"]
    mb_size = kwargs["mb_size"]

    n_pts = x.shape[0]
    random_indices = torch.randint(low=0, high=n_pts, size=(mb_size,))
    for k in range(len(random_indices)):
        print("Updating {} of {} meta samplers".format(k + 1, mb_size))
        random_index = random_indices[k].item()
        particles = prior.sample((K,)).cpu()
        particles = particles.unsqueeze(1)
        init_log_weights = torch.zeros((K, 1))
        init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
        final_target_fcn = lambda z: log_prior(z, **kwargs).cpu() + log_target(
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
            context=x[random_index],
            kwargs=kwargs,
        )
        SMC.run()
        curr_samplers[random_index]._append(SMC, random_index)
        # print('replaced index {}'.format(random_index))
    return curr_samplers


def loss_choice(loss_name, true_data, smc_samplers, **kwargs):
    if loss_name == "smcwake2":
        return smc_wake_loss2(true_data, smc_samplers, **kwargs)
    elif loss_name == "wake":
        return wake_loss(true_data, **kwargs)
    else:
        raise ValueError("Specify an appropriate loss name string.")


@hydra.main(version_base=None, config_path="../conf", config_name="config_mnist")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config6")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (
        xs,
        true_data,
        true_digits,
        K,
        prior,
        smc_samplers,
        epochs,
        device,
        mb_size,
        encoder,
        optimizer_encoder,
        logger_string,
        model,
        optimizer_model,
        kwargs,
    ) = setup(cfg)

    loss_name = kwargs["loss_name"]
    kwargs.pop("loss_name")

    if not exists("./normalized_mnist/logs/{}".format(logger_string)):
        os.mkdir("./normalized_mnist/logs/{}".format(logger_string))

    for j in range(epochs):
        # Update model
        optimizer_model.zero_grad()
        optimizer_encoder.zero_grad()
        loss = iwbo_loss(true_data, **kwargs)
        print("Model loss iter {} is {}".format(j, loss.item()))

        if torch.isnan(loss).any():
            continue
        loss.backward()
        optimizer_model.step()
        # writer.add_scalar('Model Loss', loss.item(), j)
        del loss

        # Update encoder
        optimizer_encoder.zero_grad()
        optimizer_model.zero_grad()
        loss = loss_choice(loss_name, true_data, smc_samplers, **kwargs)
        print("Encoder loss iter {} is {}".format(j, loss.item()))
        loss.backward()
        optimizer_encoder.step()
        # writer.add_scalar('Encoder Loss', loss.item(), j)
        del loss

        # Refresh SMC samplers
        if ("smc" in loss_name) and (j % cfg.smc.refresh_every == 0):
            smc_samplers = refresh_samplers_prior(true_data, smc_samplers, **kwargs)

        if (j + 1) % 5000 == 0:
            torch.save(
                model.state_dict(),
                "./normalized_mnist/logs/{}/model_{}.pth".format(logger_string, j + 1),
            )
            torch.save(
                encoder.state_dict(),
                "./normalized_mnist/logs/{}/encoder_{}.pth".format(
                    logger_string, j + 1
                ),
            )

    torch.save(
        model.state_dict(),
        "./normalized_mnist/logs/{}/model_end.pth".format(logger_string),
    )
    torch.save(
        encoder.state_dict(),
        "./normalized_mnist/logs/{}/encoder_end.pth".format(logger_string),
    )


if __name__ == "__main__":
    main()
