import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from scipy.optimize import brentq, fsolve
from scipy.special import logsumexp as lse


class EmpiricalDistribution(object):
    def __init__(self, items, weights, log_weights):
        """
        items: Tensor (n_items, ...)
        weights: Tensir (n_items, )
        self.log_raw_weights: Tensor (n_items, ...)

        """
        self.items = items
        self.weights = weights
        self.log_weights = log_weights

    def sample(self, n=10):
        indices = torch.multinomial(self.weights, num_samples=n, replacement=True)
        return self.items[indices]


class Sampler(object):
    def __init__(self, init_objs, init_weights, init_log_raw_weights):
        """
        current_ed: empirical distribution of current iteration (time t)
        eds: list of past Empirical Distributions
        """
        self.eds = []
        first_ed = EmpiricalDistribution(init_objs, init_weights, init_log_raw_weights)
        self.eds.append(first_ed)
        self.current_ed = first_ed

    def update(self, new_objs, new_weights, new_log_raw_weights):
        new_ed = EmpiricalDistribution(new_objs, new_weights, new_log_raw_weights)
        self.eds.append(new_ed)
        self.current_ed = new_ed


class LikelihoodTemperedSMC(object):
    def __init__(
        self,
        init_objs,
        init_weights,
        init_raw_log_weights,
        final_target_fcn,
        prior,
        log_prior,
        log_target_fcn,
        proposal_fcn,
        max_mc_steps=100,
        context=None,
        z_min=None,
        z_max=None,
        kwargs=None,
    ):
        """
        init_objs, init_weight as in Sampler class
        target_fcn: callable, given Tensor of particles z, return log p(x | z) for data x
        prior: callable, given Tensor of particles z, returns log p(z)
        """
        self.sampler = Sampler(init_objs, init_weights, init_raw_log_weights)
        self.num_particles = self.sampler.current_ed.items.shape[0]
        self.prior = prior
        if hasattr(prior, "support"):
            self.z_min = (
                prior.support.lower_bound
                if hasattr(prior.support, "lower_bound")
                else -np.inf
            )
            self.z_max = (
                prior.support.upper_bound
                if hasattr(prior.support, "upper_bound")
                else np.inf
            )
        else:
            self.z_min = z_min
            self.z_max = z_max
        self.log_prior = log_prior
        self.log_target_fcn = log_target_fcn
        self.proposal_fcn = proposal_fcn
        self.curr_tau = 0.0
        self.final_target_fcn = final_target_fcn
        self.ESS_min_prop = 0.5
        self.ESS_min = math.floor(self.num_particles * self.ESS_min_prop)
        self.curr_stage = 1
        self.max_mc_steps = max_mc_steps
        self.softmax = nn.Softmax(0)
        self.tau_list = [self.curr_tau]
        self.context = context
        self.kwargs = kwargs
        self.cached_log_targets = None

    def _aux_solver_func(self, delta, curr_particles):
        if self.cached_log_targets is not None:
            log_targets = self.cached_log_targets
        else:
            log_targets = self.final_target_fcn(curr_particles)
            log_targets = log_targets.detach().cpu().numpy()
            self.cached_log_targets = log_targets
        log_numerator = 2 * lse(np.nan_to_num(delta * log_targets, nan=-np.inf))
        log_denominator = lse(2 * np.nan_to_num(delta * log_targets, nan=-np.inf))
        result = log_numerator - log_denominator - np.log(self.ESS_min)
        return result

    def concatenate(self, old_objs, new_objs):
        """
        old_objs: Tensor n x (K-1) x ...
        new_objs Tensor n x ...

        Output: Tensor n x K x ...
        """
        return torch.cat([old_objs, new_objs.unsqueeze(1)], 1)

    def chi_square_dist(self, curr_particles):
        func = lambda delta: self._aux_solver_func(delta, curr_particles)
        x0 = 0.0
        sign0 = np.sign(func(x0))
        bs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        signs = np.array([np.sign(func(x)) for x in bs])
        diffs = sign0 - signs
        if np.sum(diffs != 0) > 0:
            my_b = bs[np.where(diffs != 0)[0][0]]
            solutions = brentq(func, a=0.0, b=my_b)
            delta = np.sort(
                np.clip(np.array([solutions]), a_min=0.0, a_max=1.0 - self.curr_tau)
            )[0]
        else:
            solutions = fsolve(func=func, x0=x0, maxfev=1000)
            delta = np.sort(
                np.clip(np.array([solutions]), a_min=0.0, a_max=1.0 - self.curr_tau)
            )[0][0]

        self.cached_log_targets = None
        if delta <= 1e-20:
            if len(self.tau_list) >= 2:
                prev_diff = self.tau_list[-1] - self.tau_list[-2]
                return min(max(0.0, prev_diff), 1 - self.curr_tau)
            else:
                return 1e-20
        else:
            return delta

    def log_weight_helper(self, curr_zs, prev_log_target, curr_log_target):
        """
        prev_zs: sampled (latent) particles from previous iteration
        curr_zs: sampled (latent) particles from current iteration
        prev_target: previous iteration's unnormalized target
        curr target: this iteration's unnormalized target
        proposal: proposal distribution from previous to current iteration

        All of these last three should be able to call .log_prob(zs) to return
        a log density for each row of zs.
        """
        z_to_use = curr_zs[:, -1, ...]
        num = curr_log_target(z_to_use)
        denom = prev_log_target(z_to_use)
        return num, denom

    def ess(self):
        """
        Check ESS of current set of particles and weights.
        """
        weights = self.sampler.current_ed.weights
        return weights.square().sum() ** (-1)

    def one_step(self):
        # Check ESS
        ess = self.ess()

        # Optionally resample
        if ess <= self.ESS_min:
            # Resample
            samples = self.sampler.current_ed.sample(self.num_particles)
            log_w_hat = torch.zeros(self.sampler.current_ed.weights.shape)
        else:
            samples = self.sampler.current_ed.items
            log_w_hat = self.sampler.current_ed.log_weights[:, -1]

        # Compute next tau in schedule
        z_to_use = samples[:, -1, ...]
        delta = self.chi_square_dist(z_to_use)
        if delta <= 1e-6:
            delta = 1e-6
        # if delta >= .2:
        #     delta = .2
        next_tau = self.curr_tau + delta
        print(next_tau)

        # Construct targets
        curr_target = (
            lambda z: self.log_prior(z, **self.kwargs).cpu()
            + self.log_target_fcn(z, self.context, **self.kwargs).cpu() * next_tau
        )
        prev_target = lambda z: self.log_prior(
            z, **self.kwargs
        ).cpu() + torch.nan_to_num(
            self.log_target_fcn(z, self.context, **self.kwargs).cpu() * self.curr_tau,
            -torch.inf,
        )

        # Propagate
        new_zs = self.proposal_fcn(
            z_to_use, self.context, curr_target, **self.kwargs
        )  # context can be anything
        new_zs = new_zs.clamp(min=self.z_min, max=self.z_max)

        # Concatenate
        new_histories = self.concatenate(samples, new_zs)

        # Compute weights
        log_curr_target, log_prev_target = self.log_weight_helper(
            new_histories, prev_target, curr_target
        )
        log_weights = log_w_hat + log_curr_target.view(-1) - log_prev_target.view(-1)
        weights = self.softmax(torch.nan_to_num(log_weights, -torch.inf))

        # Concatenate history of weights
        new_log_weights = self.concatenate(
            self.sampler.current_ed.log_weights, log_weights
        )

        # Update state of the SMC system
        self.sampler.update(new_histories, weights, new_log_weights)
        self.curr_stage += 1

        # Update rolling quanitities
        self.curr_tau = next_tau
        self.curr_log_targets = log_curr_target
        self.tau_list.append(self.curr_tau)

    def run(self):
        while (self.curr_tau < 1.0) and (self.curr_stage < self.max_mc_steps):
            self.one_step()

    def log_evidence_estimate(self, t=-1):
        """
        Returns the estimate of the normalization constant at time $t$.
        When $t=-1$ returns this quantity for the final step.

        Formulae: page. 130 Chopin & Papaspilious
        l_t = (1/K)\sum_{i=1}^K w_i, for unnormalized weights w_i
        L_t = \prod_{s=1}^t l_s
        """

        log_unnormalized_weights = torch.clone(self.sampler.current_ed.log_weights)
        log_unnormalized_weights = torch.nan_to_num(
            log_unnormalized_weights, nan=-torch.inf
        )
        K = log_unnormalized_weights.shape[0]
        log_unnormalized_weights += torch.log(torch.tensor(1 / K))
        log_l_ts = log_unnormalized_weights.logsumexp(
            dim=0
        )  # across all $K$ log_weights
        # To multiply the first few together, add the logs
        log_L_ts = torch.cumsum(log_l_ts, dim=0)
        # Return the element we want (-1) for last
        return log_L_ts[t]


class MetaImportanceSampler(object):
    def __init__(
        self,
        curr_smc_samplers,
        data,
        i,
        keep_all=False,
        n_resample=1,
        Mprime=1,
        kwargs=None,
    ):
        """
        curr_smc_samplers = List[LikelihoodTemperedSMC]
        This will be mutable, and we will be constantly appending new samplers to the list.
        We can access log normalization constants for each of these, and will use these
        as the log_weights within this Meta Importance Sampler.

        keep_all = True reverts to MetaIS
        keep_all = False results in random-weight IS
        """
        self.keep_all = keep_all
        self.n_resample = n_resample
        self.log_ZTs = torch.cat(
            [
                smc.log_evidence_estimate().reshape(
                    1,
                )
                for smc in curr_smc_samplers
            ],
            0,
        )
        self.data = data
        self.this_index = i
        self.kwargs = kwargs
        self.Mprime = Mprime

        # Decide how to keep track of the individual samplers.
        if self.keep_all:
            # Necessary for estimator \hat{\nabla}^{(a)} from manuscript.
            # High memory cost, only do if necessary
            self.curr_smc_samplers = curr_smc_samplers
        else:
            # Lower memory cost version. Keep a fixed tensor of draws from each distribution.
            # n_resample << K, so this can save an order of magnitude of memory
            self.curr_particles = torch.stack(
                [
                    smc.sampler.current_ed.sample(self.n_resample)[:, -1, ...]
                    for smc in curr_smc_samplers
                ]
            )  # n_samplers x n_resample x dim

    def _append(self, new_sampler, index_used):
        assert (
            index_used == self.this_index
        ), "Error: you are appending an SMC sampler for a different point."
        self.log_ZTs = torch.cat(
            [
                self.log_ZTs,
                new_sampler.log_evidence_estimate().reshape(
                    1,
                ),
            ],
            0,
        )
        if self.keep_all:
            self.curr_smc_samplers.append(new_sampler)
        else:
            new_samples = new_sampler.sampler.current_ed.sample(self.n_resample)[
                :, -1, ...
            ]
            self.curr_particles = torch.cat(
                [
                    self.curr_particles,
                    rearrange(new_samples, "n_resample d -> 1 n_resample d"),
                ],
                0,
            )
        return

    def _append_pimh(self, new_sampler, index_used):
        assert (
            index_used == self.this_index
        ), "Error: you are appending an SMC sampler for a different point."
        assert (
            self.keep_all
        ), "Error, need to have keep_all=True for PIMH implementation"
        curr_log_ZT = self.log_ZTs[0].item()
        new_log_ZT = new_sampler.log_evidence_estimate().item()
        log_alpha = new_log_ZT - curr_log_ZT
        prob = 1 if log_alpha > 0.0 else math.exp(log_alpha)
        coin_flip = D.Bernoulli(prob).sample().item()
        if coin_flip == 1:
            self.log_ZTs = torch.tensor(new_log_ZT).reshape(
                1,
            )
            self.curr_smc_samplers = [new_sampler]
        return

    def _get_recent(self):
        """
        Access particles and weights for most recent sampler. No weighting
        of samplers; could be used for naive, biased version of SMC-Wake
        """
        assert self.keep_all is True
        curr_sampler = self.curr_smc_samplers[-1]
        return (
            curr_sampler.sampler.current_ed.items[:, -1, ...],
            curr_sampler.sampler.current_ed.weights,
        )

    # METHODS FOR ESTIMATOR (a)
    def _get_meta_weights(self, n_samp=100):
        """
        Get meta-weights for this importance sampler.
        Computed as Z_T,i/\sum_j Z_T,j for normalizing constants Z_T,j
        """
        assert self.keep_all is True
        meta_weights = torch.softmax(torch.tensor(self.log_ZTs), dim=0)
        n_to_resample = n_samp
        indices = D.Categorical(meta_weights).sample((n_to_resample,))
        return meta_weights, indices

    def _get_weights_particles(self, n_samp=100):
        """
        Lower cost approximation for estimator (a). Instead of using
        all $M$ samplers, when $M$ is large resamples only n_samp
        samplers using their meta importance weights.
        """
        assert self.keep_all is True
        _, indices = self._get_meta_weights(n_samp)
        chosen_smcs = [self.curr_smc_samplers[i] for i in indices]
        particles = [smc.sampler.current_ed.items[:, -1, ...] for smc in chosen_smcs]
        particles = torch.stack(particles)
        particles = particles.transpose(0, 1)  # for shapes
        weights = [smc.sampler.current_ed.weights for smc in chosen_smcs]
        weights = torch.stack(weights)
        return particles, weights

    def _get_resampled_particles(self, n_samp=100):
        """
        Even lower cost approximation for estimator (a). Instead of using
        all $M$ samplers, only nsamp as above. Further, instead of using all
        $K$ particles for each sampler, only use nsamp of them sampled
        using appropriate weights from each.
        """
        assert self.keep_all is True
        _, indices = self._get_meta_weights(n_samp)
        chosen_smcs = [self.curr_smc_samplers[i] for i in indices]
        particles = [
            smc.sampler.current_ed.sample(n_samp)[:, -1, ...] for smc in chosen_smcs
        ]
        particles = torch.stack(particles)
        particles = particles.transpose(0, 1)  # for shapes
        return particles

    # METHODS FOR ESTIMATOR (b)
    def _get_resampled_weights_particles(self):
        """
        Lightweight method for estimator (b). Returns:
        - self.curr_particles, a tensor of particles (one for each sampler)
        - weights for each of these (same as meta weights for samplers)
        - a context variable that repeats this data point for log likelihood evaluation.
        """
        context = repeat(
            self.data[self.this_index],
            "d -> b n_resamp d",
            b=self.curr_particles.shape[0],
            n_resamp=self.n_resample,
        )
        return self.curr_particles, nn.Softmax(0)(self.log_ZTs), context

    # NAIVE ESTIMATOR WITHOUT WEIGHTING
    def _get_random_subset_weights_particles(self):
        assert self.keep_all is True
        subset = torch.randint(low=0, high=len(self.log_ZTs), size=(self.Mprime,))
        chosen_smcs = [self.curr_smc_samplers[i] for i in subset]
        particles = [smc.sampler.current_ed.items[:, -1, ...] for smc in chosen_smcs]
        particles = torch.stack(particles)
        particles = particles.transpose(0, 1)  # for shapes
        weights = [smc.sampler.current_ed.weights for smc in chosen_smcs]
        weights = torch.stack(weights)
        return particles, weights
