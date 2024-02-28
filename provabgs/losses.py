import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from generate import generate_data_emulator
from utils import log_t_prior


def get_smc_items(x, smc_samplers, **kwargs):
    mb_size = kwargs["mb_size"]

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))

    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    # Sample and select particles, weights
    # Sample and select particles, weights
    all_part_weights = [smc._get_weights_particles(10) for smc in these_meta_samplers]
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K r d -> K r b d")
    weights = rearrange(all_weights, "b r K -> b r K")

    return indices, particles, weights


def smc_wake_loss_a(x, smc_samplers, **kwargs):
    # Choose data points
    device = kwargs["device"]
    indices, particles, weights = get_smc_items(x, smc_samplers, **kwargs)
    pts = x[indices].to(device)
    encoder = kwargs["encoder"]
    K = kwargs["K"]

    # Compute Forward KL Loss
    n_samplers_all = particles.shape[1]
    rpts = repeat(pts, "b dim -> K b nsamp dim", K=K, nsamp=n_samplers_all).to(device)
    to_eval = rearrange(particles, "K nsamp b k -> K b nsamp k").to(device)

    reshaped_zs = rearrange(to_eval, "K b nsamp d -> (K b nsamp) d")
    reshaped_xs = rearrange(rpts, "K b nsamp d -> (K b nsamp) d")

    lps = encoder.log_prob(reshaped_zs, reshaped_xs)
    lps = rearrange(lps, "(K b nsamp) -> b K nsamp", K=K, nsamp=n_samplers_all)

    mms = torch.bmm(weights.to(device).float(), lps)
    diags = torch.diagonal(mms, dim1=-2, dim2=-1)
    return -1 * diags.mean()
