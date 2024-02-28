import os
import random

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torchvision import datasets, transforms


@hydra.main(version_base=None, config_path="../conf", config_name="config_mnist")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config_mnist")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    num_obs = 1000

    mnist = datasets.MNIST(root="./normalized_mnist/data", train=False, download=True)

    # Labeled data
    data = mnist.data[:num_obs] / 255.0
    labels = mnist.test_labels[:num_obs]
    torch.save(data, "./normalized_mnist/data/continuous_data_BIG.pt")
    torch.save(labels, "./normalized_mnist/data/labels_BIG.pt")

    test_data = mnist.data[num_obs : 2 * num_obs] / 255.0
    test_labels = mnist.test_labels[num_obs : 2 * num_obs]
    torch.save(test_data, "./normalized_mnist/data/continuous_data_test_BIG.pt")
    torch.save(test_labels, "./normalized_mnist/data/labels_test_BIG.pt")


if __name__ == "__main__":
    main()
