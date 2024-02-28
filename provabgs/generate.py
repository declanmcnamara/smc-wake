import numpy as np
import torch
import torch.distributions as D
from utils import prior_t_sample, resample, transform_thetas


def generate_data_emulator(n_samples=100, return_theta=True, **kwargs):
    emulator = kwargs["model"]
    multiplicative_noise = kwargs["multiplicative_noise"]
    device = kwargs["device"]

    thetas = prior_t_sample(n_samples, **kwargs).to(device)
    means = emulator(thetas).clamp(min=0.0)
    means = means.detach()

    samples = D.Normal(means, multiplicative_noise * torch.abs(means) + 1e-8).sample()

    if not return_theta:
        return samples
    else:
        return thetas, samples
