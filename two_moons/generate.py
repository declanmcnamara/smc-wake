import math

import torch
import torch.distributions as D


def get_p(r, a):
    one = r * torch.cos(a) + 0.25
    two = r * torch.sin(a)
    return torch.stack([one, two])


def get_x(r, a, theta):
    p_val = get_p(r, a)
    new1 = -1 * torch.sum(theta, -1).abs() / math.sqrt(2)
    new2 = (-1 * theta[..., 0] + theta[..., 1]) / math.sqrt(2)
    new = torch.stack([new1, new2])
    return (p_val + new).T


def generate_data(n_pts, return_theta=False):
    prior = D.Uniform(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
    a_dist = D.Uniform(-math.pi / 2, math.pi / 2)
    r_dist = D.Normal(0.1, 0.01)

    theta, a, r = (
        prior.sample((n_pts,)),
        a_dist.sample((n_pts,)),
        r_dist.sample((n_pts,)),
    )
    x = get_x(r, a, theta)
    if return_theta:
        return theta, x
    else:
        return x


def generate_data_favi(n_pts, return_theta=False, **kwargs):
    prior = kwargs["prior"]
    a_dist = D.Uniform(-math.pi / 2, math.pi / 2)
    r_dist = D.Normal(0.1, 0.01)

    theta, a, r = (
        prior.sample((n_pts,)),
        a_dist.sample((n_pts,)),
        r_dist.sample((n_pts,)),
    )
    x = get_x(r, a, theta)
    theta = theta.to(kwargs["device"])
    x = x.to(kwargs["device"])
    if return_theta:
        return theta, x
    else:
        return x
