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
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 30})


@hydra.main(config_path="../conf", config_name="config_many")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config_many")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    one_sampler_results = np.load(
        "./many_samplers/logs/bigloss=smcwake_og,init=1,rs=10,k=50,n=100,lr=0.0001,sigma=1,tau=1,npts=10,K=10000,refresh=10000000,epochs=100000refreshtime=10000000fullrank_fklsrw.npy"
    )
    mini_samplers_results = np.load(
        "./many_samplers/logs/littleloss=smcwake_og,init=100,rs=10,k=50,n=100,lr=0.0001,sigma=1,tau=1,npts=10,K=100,refresh=10000000,epochs=100000refreshtime=10000000fullrank_fklsrw.npy"
    )

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(one_sampler_results, label="$M=1$, $K=10000$")
    ax.plot(mini_samplers_results, label="$M=100$, $K=100$")
    ax.set_xlabel("Gradient Step", fontsize=40)
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Forward KL Divergence", fontsize=40)
    leg = ax.legend(fontsize=50)
    for line in leg.get_lines():
        line.set_linewidth(5.0)
    plt.savefig("./many_samplers/figs/big_vs_small.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
