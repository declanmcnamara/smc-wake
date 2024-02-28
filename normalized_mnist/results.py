import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import sys

sys.path.append("../")
import random
from operator import add

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from hydra import compose, initialize
from modules import SBN, Encoder
from omegaconf import DictConfig
from setup import log_prior, log_target, proposal, setup
from torch.utils.tensorboard import SummaryWriter

from smc.smc_sampler import LikelihoodTemperedSMC


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
    ) = setup(cfg)

    kwargs["K"] = 100
    kwargs["mb_size"] = len(true_data)

    # Assessment for latent dimension 16
    iteration = 50000

    strings = [
        "smcwake2,0.0005,100,16,True,32,1_digits_",
        "wake,0.0005,100,16,True,32,1_digits_",
    ]

    # Select a visual example to compare Wake and SMC-Wake(b)
    plt.rcParams.update({"font.size": 42})
    cap = 6
    for digit in range(0, 10):
        this_digit_collection_true_wake = []
        this_digit_collection_smcwake = []
        counter = 0
        for j in range(1000):
            # Check if correct digit
            if true_digits[j].item() != digit:
                continue
            counter += 1

            real_digit = true_x[j].reshape((28, 28))

            # Wake
            s = strings[-1]
            kwargs["encoder"].load_state_dict(
                torch.load(
                    "./normalized_mnist/logs/{}/encoder_{}.pth".format(s, iteration)
                )
            )
            kwargs["model"].load_state_dict(
                torch.load(
                    "./normalized_mnist/logs/{}/model_{}.pth".format(s, iteration)
                )
            )
            encoding_mean = kwargs["encoder"].get_q(true_data[j : j + 1]).loc
            decoding_mean = (
                kwargs["model"](encoding_mean, true_digits[j : j + 1])
                .reshape((28, 28))
                .detach()
                .numpy()
            )

            # SMC-Wake(b)
            s = strings[0]
            kwargs["encoder"].load_state_dict(
                torch.load(
                    "./normalized_mnist/logs/{}/encoder_{}.pth".format(s, iteration)
                )
            )
            kwargs["model"].load_state_dict(
                torch.load(
                    "./normalized_mnist/logs/{}/model_{}.pth".format(s, iteration)
                )
            )
            encoding_mean_smc = kwargs["encoder"].get_q(true_data[j : j + 1]).loc
            decoding_mean_smc = (
                kwargs["model"](encoding_mean_smc, true_digits[j : j + 1])
                .reshape((28, 28))
                .detach()
                .numpy()
            )

            this_digit_collection_true_wake.append((real_digit, decoding_mean))
            this_digit_collection_smcwake.append(decoding_mean_smc)

            if counter > cap:
                break

        fig, ax = plt.subplots(
            nrows=3,
            ncols=cap,
            sharey=True,
            figsize=(30, 15),
            gridspec_kw={"wspace": 0.1, "hspace": 0},
        )
        for j in range(cap):
            ax[0, j].imshow(this_digit_collection_true_wake[j][0])
            ax[1, j].imshow(this_digit_collection_true_wake[j][1])
            ax[0, j].set_xticklabels([])
            ax[0, j].set_yticklabels([])
            ax[1, j].set_xticklabels([])
            ax[1, j].set_yticklabels([])
            ax[2, j].imshow(this_digit_collection_smcwake[j])
            ax[2, j].set_xticklabels([])
            ax[2, j].set_yticklabels([])
            # ax[j, 2].remove()
            # ax[2].imshow(decoding_mean_smc)

        ax[0, 0].set_ylabel("Actual")
        ax[1, 0].set_ylabel("Wake")
        ax[2, 0].set_ylabel("SMC-Wake(b)")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(
            "./normalized_mnist/figs/train_digit_collection_{}.png".format(digit),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
