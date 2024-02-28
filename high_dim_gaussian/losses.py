import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from generate import generate_data
from utils import vector_gather

# ------HELPER FUNCTIONS------#


def get_imp_weights(particles, pts, log=False, **kwargs):
    """
    Given a set of particles, and data poins pts, uses encoder and model to construct
    importance weights.
    """
    device = kwargs["device"]
    enc = kwargs["enc"]
    A = kwargs["A"].to(device)
    tau = kwargs["tau"]
    n = kwargs["n"]
    K = kwargs["K"]
    prior = kwargs["prior"]

    batch_size = particles.shape[1]
    rA = repeat(A, "d k -> c d k", c=batch_size * K)
    viewedZ = rearrange(particles, "K b k -> (K b) k 1")
    means = rearrange(torch.bmm(rA, viewedZ), "(K b) d 1 -> K b d", K=K, b=batch_size)
    distr = D.MultivariateNormal(means.to(device), tau**2 * torch.eye(n).to(device))
    eval_at = repeat(pts, "b 1 d -> K b d", K=K).to(device)
    # eval_at = pts.repeat(z_to_use.shape[0], 1, 1).to(device)#.transpose(0, 1)
    likelihoods = distr.log_prob(eval_at)  # shape N x len(true_x)
    prior_dens = prior.log_prob(particles.to(prior.loc.device)).to(device)

    log_pxz = prior_dens + likelihoods
    log_qzx = enc.get_q(pts).log_prob(rearrange(particles, "K b k -> K b 1 k"))
    log_weights = log_pxz - rearrange(log_qzx, "K b 1 -> K b")
    weights = nn.Softmax(0)(log_weights)

    if log:
        return weights.to(device), log_weights.to(device)
    else:
        return weights.to(device)


def markovian_score_climbing_loss(j, xs, mb_size, samplers=None, **kwargs):
    device = kwargs["device"]
    old_particles = kwargs["old_particles"]
    # Choose data points
    mb_indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = rearrange(xs[mb_indices].to(device), "b d -> b 1 d")
    these_old_particles = rearrange(old_particles[mb_indices], "b k -> 1 b k")

    # Access quantities from kwargs
    enc = kwargs["enc"]
    K = kwargs["K"]

    # Get samples, weights
    particles = rearrange(enc.get_q(pts).sample((K - 1,)), "K b 1 dim -> K b dim")
    particles = torch.cat([particles, these_old_particles], dim=0)

    weights = get_imp_weights(particles, pts, **kwargs).detach()
    selected_indices = torch.multinomial(weights.T, 1)  # b x 1
    selected_indices = rearrange(selected_indices, "b 1 -> b")
    particles = rearrange(particles, "K b d -> b K d")

    sel_particles = vector_gather(particles, selected_indices)

    # Get log_probs
    lps = enc.get_q(pts).log_prob(rearrange(sel_particles, "b k -> b 1 k"))

    # Return loss
    return -1 * lps.mean(), sel_particles, mb_indices


# Define loss functions
def smc_wake_loss(j, xs, mb_size, smc_samplers, **kwargs):
    # Choose data points
    device = kwargs["device"]

    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = rearrange(xs[indices].to(device), "b d -> b 1 d")
    # Grab its SMC approximation
    these_meta_samplers = [smc_samplers[index] for index in indices]

    all_part_weights = [smc._get_recent() for smc in these_meta_samplers]
    all_part = [x[0] for x in all_part_weights]
    all_weights = [x[1] for x in all_part_weights]

    # Stack for ease
    particles = rearrange(all_part, "b K d -> K b d").to(device)
    weights = rearrange(all_weights, "b K -> b K").to(device)

    encoder = kwargs["enc"]

    # Compute Forward KL Loss
    lps = encoder.get_q(pts).log_prob(rearrange(particles, "K b k -> K b 1 k"))
    lps = rearrange(lps, "K b 1 -> K b")
    return -1 * torch.diag(weights @ lps).mean()
