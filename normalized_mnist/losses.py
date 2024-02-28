import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat


def get_imp_weights(particles, pts, log=False, **kwargs):
    """
    Given a set of particles, and data poins pts, uses encoder and model to construct
    importance weights.
    """
    device = kwargs["device"]
    encoder = kwargs["encoder"]
    prior = kwargs["prior"]
    model = kwargs["model"]

    log_prior = prior.log_prob(particles).sum(-1)
    log_q = encoder.log_prob(particles, pts).sum(-1)
    image_part, label_part = pts[..., :-1], pts[..., -1:]
    log_p = model.log_prob(particles, label_part.long(), image_part)

    log_weights = log_prior + log_p - log_q
    weights = nn.Softmax(0)(log_weights)

    if log:
        return weights.to(device), log_weights.to(device)
    else:
        return weights.to(device)


# Define loss functions
def smc_wake_loss2(xs, smc_samplers, **kwargs):
    # Choose data points
    device = kwargs["device"]
    mb_size = kwargs["mb_size"]
    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    enc = kwargs["encoder"]

    # Get particles
    these_meta_samplers = [smc_samplers[index] for index in indices]
    info = [smc._get_resampled_weights_particles() for smc in these_meta_samplers]
    particles = [pair[0] for pair in info]
    weights = [pair[1] for pair in info]
    contexts = [pair[2] for pair in info]

    all_particles = torch.cat(particles, dim=0).to(device)
    all_weights = torch.cat(weights, dim=0).float().to(device)
    all_contexts = torch.cat(contexts, 0).to(device)

    # Compute Forward KL Loss
    lps = enc.get_q(all_contexts).log_prob(all_particles).sum(-1)
    averaged = reduce(lps, "n_samp_total n_resample -> n_samp_total", "mean")
    dotted = torch.dot(averaged, all_weights)
    return -1 * dotted / mb_size


def wake_loss(xs, samplers=None, **kwargs):
    device = kwargs["device"]
    mb_size = kwargs["mb_size"]
    # Choose data points
    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = xs[indices].to(device)

    # Access quantities from kwargs
    encoder = kwargs["encoder"]
    K = kwargs["K"]

    # Get samples, weights
    particles = encoder.get_q(pts).sample((K,))
    weights = get_imp_weights(particles, pts, **kwargs).detach()

    # Get log_probs
    lps = encoder.get_q(pts).log_prob(particles).sum(-1)

    # Return loss
    return -1 * torch.diag(weights.T @ lps).mean()


def iwbo_loss(xs, samplers=None, **kwargs):
    # Choose data points
    device = kwargs["device"]
    mb_size = kwargs["mb_size"]
    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = xs[indices].to(device)

    encoder = kwargs["encoder"]
    K = kwargs["K"]

    # Get samples, weights
    particles = encoder.get_q(pts).rsample((K,))
    weights, log_weights = get_imp_weights(particles, pts, log=True, **kwargs)
    weights = weights.detach()

    # Return loss
    return -1 * torch.diag(weights.T @ log_weights).mean()
