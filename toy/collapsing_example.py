import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn as nn

"""
Consider

z ~ N(0, 10^2)
x | z ~ N(z, 1)

The exact posterior is
z | x ~ N((100/101)x, 100/101).


For this example problem, we set z = 10, x = 12.
"""
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)


true_z = torch.tensor(10.0)
true_x = torch.tensor(12.0)
exact_posterior = D.Normal((100 / 101) * true_x, math.sqrt(100 / 101))

"""Consider a range of "dumb' variational posteriors, with mean zero
and increasingly smaller variance."""
posteriors_to_try = [
    D.Normal(0.0, 1e-1),
    D.Normal(0.0, 1e-2),
    D.Normal(0.0, 1e-3),
    D.Normal(0.0, 1e-4),
    D.Normal(0.0, 1e-5),
    D.Normal(0.0, 1e-6),
    D.Normal(0.0, 1e-7),
    exact_posterior,
]

"""Calculate the wake-phase objective using self-normalized importance sampling."""
"""Propose from the exact posterior, and the dumb variational posteriors."""


def wake_loss(proposal_dist, num_IS_draws):
    proposed_particles = proposal_dist.sample((num_IS_draws,))
    log_numerator = D.Normal(0.0, 10.0).log_prob(proposed_particles) + D.Normal(
        proposed_particles, 1.0
    ).log_prob(true_x)
    log_denominator = proposal_dist.log_prob(proposed_particles)
    log_weights = log_numerator - log_denominator
    normalized_weights = nn.Softmax(0)(log_weights)
    return -1 * torch.dot(normalized_weights, log_denominator)


results = [
    [wake_loss(this_proposal, 10000).item() for this_proposal in posteriors_to_try]
    for _ in range(100)
]
results = np.stack(results)
means = results.mean(0)
stds = results.std(0)
vals_for_table = [
    "{0:.3f} ({1:.3f})".format(means[j], stds[j]) for j in range(len(means))
]

names = [
    "Mean: {0}, Stddev: {1:.6f}".format(
        this_proposal.loc, torch.round(this_proposal.scale, decimals=6)
    )
    for this_proposal in posteriors_to_try
]
names[-1] = "Exact Posterior"

to_show = pd.DataFrame(
    {"Proposal Distribution": names, "Wake Objective Value": vals_for_table}
)

to_show.to_latex("concentrated_proposals.tex")
