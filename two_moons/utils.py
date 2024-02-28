import math

import torch


def log_post(x, theta, **kwargs):
    """vectorized version of the above in theta for fixed x"""
    assert (
        theta.shape[1] == 2
    ), "not yet implemented for evaluation on multiple points at once"
    r_dist = kwargs["r_dist"]
    new1 = -1 * torch.sum(theta, 1).abs() / math.sqrt(2)
    new2 = (-1 * theta[:, 0] + theta[:, 1]) / math.sqrt(2)
    new = torch.stack([new1, new2]).T
    p = x - new
    u = p[:, 0] - 0.25
    v = p[:, 1]
    r = torch.sqrt(u**2 + v**2)  # note the angle distribution is uniform
    to_adjust = r_dist.log_prob(r)
    adjusted = torch.where(u < 0.0, -torch.inf, to_adjust.double())
    return adjusted
