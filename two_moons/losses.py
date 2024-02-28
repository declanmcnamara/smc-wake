import torch
import torch.distributions as D
import torch.nn as nn
from einops import pack, rearrange, reduce, repeat, unpack
from generate import generate_data, generate_data_favi
from utils import log_post

# ------HELPER FUNCTIONS------#


def get_smc_items(x, smc_samplers, **kwargs):
    mb_size = kwargs["mb_size"]

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))

    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    # Sample and select particles, weights
    all_part_weights = [smc._get_weights_particles(10) for smc in these_meta_samplers]
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K r d -> K r b d")
    weights = rearrange(all_weights, "b r K -> b r K")

    return indices, particles, weights


def get_imp_weights(pts, log=False, prop_prior=0.0, **kwargs):
    mb_size = kwargs["mb_size"]
    encoder = kwargs["encoder"]
    K = kwargs["K"]
    prior = kwargs["prior"]
    device = kwargs["device"]

    num_prior = int(K * prop_prior // 1)
    num_q = int(K * (1 - prop_prior) // 1)

    particlesp = prior.sample((mb_size, num_prior))
    log_denomsp = prior.log_prob(particlesp).sum(-1).to(device)
    particlesp = particlesp.to(device)

    if num_q > 0:
        particlesq, log_denomsq = encoder.sample_and_log_prob(
            num_samples=num_q, context=pts.to(device)
        )
        particles = torch.cat([particlesq, particlesp], dim=1)
        log_denoms = torch.cat([log_denomsq, log_denomsp], dim=-1)
    else:
        particles = particlesp
        log_denoms = log_denomsp

    particles = particles.reshape(K * mb_size, -1).clamp(-kwargs["val"], kwargs["val"])
    log_denoms = log_denoms.view(K, -1)
    repeated_pts = pts.repeat(K, 1, 1).reshape(K * mb_size, -1).to(device)
    log_nums = log_post(repeated_pts, particles, **kwargs).reshape(K, mb_size)
    log_weights = log_nums - log_denoms
    weights = nn.Softmax(0)(log_weights)
    if log:
        return particles, weights, log_weights
    else:
        return particles, weights


# ------LOSS FUNCTIONS------#
def smc_wake_loss_naive(x, smc_samplers, **kwargs):
    # Choose data points
    device = kwargs["device"]
    mb_size = kwargs["mb_size"]
    encoder = kwargs["encoder"]
    K = kwargs["K"]

    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices].to(device)
    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    all_part_weights = [smc._get_recent() for smc in these_meta_samplers]
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K d -> K b d").to(device)
    weights = rearrange(all_weights, "b K -> b K").to(device)
    contexts = repeat(pts, "b d -> K b d", K=K)

    # Compute Forward KL Loss
    lps = encoder.log_prob(
        rearrange(particles, "K b d -> (K b) d"),
        rearrange(contexts, "K b d -> (K b) d"),
    )
    lps = rearrange(lps, "(K b) -> K b", K=K)
    return -1 * torch.diag(weights @ lps.double()).mean()


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


def smc_wake_loss_b(x, smc_samplers, **kwargs):
    mb_size = kwargs["mb_size"]
    device = kwargs["device"]

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))

    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]
    encoder = kwargs["encoder"]

    # Sample and concatenate particles
    info = [smc._get_resampled_weights_particles() for smc in these_meta_samplers]
    particles = [pair[0] for pair in info]
    weights = [pair[1] for pair in info]
    contexts = [pair[2] for pair in info]

    all_particles = torch.cat(particles, dim=0).to(device)
    all_weights = torch.cat(weights, dim=0).float().to(device)
    all_contexts = torch.cat(contexts, 0)
    nsamp = all_particles.shape[-2]

    lps = encoder.log_prob(
        rearrange(all_particles, "b nsamp d -> (b nsamp) d"),
        rearrange(all_contexts, "b nsamp d -> (b nsamp) d"),
    )
    lps = reduce(lps, "(b nsamp) -> b", nsamp=nsamp, reduction="mean")

    dotted = torch.dot(lps, all_weights)
    return -1 * dotted / mb_size  # can do a straight average due to resampling


def smc_wake_loss_c(x, smc_samplers, **kwargs):
    device = kwargs["device"]
    mb_size = kwargs["mb_size"]
    encoder = kwargs["encoder"]
    K = kwargs["K"]

    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices].to(device)
    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    all_part_weights = [smc._get_recent() for smc in these_meta_samplers]
    weight_last_sampler = torch.cat(
        [nn.Softmax(0)(smc.log_ZTs)[-1:] for smc in these_meta_samplers], 0
    ).to(device)
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K d -> (K b) d").to(device)
    weights = rearrange(all_weights, "b K -> b K").to(device)
    rpts = repeat(pts, "b d -> (K b) d", K=K)

    lps = encoder.log_prob(particles, rpts)
    lps = rearrange(lps, "(K b) -> K b", K=K)

    ind_dotted = -1 * torch.diag(weights.float() @ lps)
    dotted = torch.dot(ind_dotted, weight_last_sampler.float())
    return dotted / mb_size


def wake_loss(x, mdn=True, flow=False, prop_prior=0.0, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    mb_size = kwargs["mb_size"]
    encoder = kwargs["encoder"]
    K = kwargs["K"]
    device = kwargs["device"]

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices]

    particles, weights = get_imp_weights(pts, prop_prior=prop_prior, **kwargs)
    weights = weights.detach()
    reshaped = particles.reshape(K * mb_size, -1).to(device)
    context = pts.to(device).repeat(K, 1, 1).reshape(K * mb_size, -1)
    lps = encoder.log_prob(reshaped, context)
    lps = lps.reshape(K, -1)
    return -1 * torch.diag(weights.T @ lps.double()).mean()
