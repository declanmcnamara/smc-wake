import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import sys

sys.path.append("../")
import copy

import numpy as np
import torch

# -- plotting --
import torch.distributions as D
import torch.nn as nn
from generate import generate_data_emulator
from modules import PROVABGSEmulator, TransformedFlatDirichlet, TransformedUniform
from utils import log_t_prior, prior_t_sample

from cde.nsf import EmbeddingNet, build_nsf
from smc.smc_sampler import LikelihoodTemperedSMC, MetaImportanceSampler


def log_target(thetas, sed, **kwargs):
    """
    Use chi_square based llk calculation from our generative model.
    Maybe figure out how this is implicitly defined.

    tthetas: (n_batch, 12) batch of thetas
    sed: a single SED

    Returns: (n_batch,) array of log likelihoods
    """
    multiplicative_noise = kwargs["multiplicative_noise"]
    device = kwargs["device"]
    kwargs["model"] = kwargs["model"].to("cpu")
    model = kwargs["model"]

    means = model(thetas).clamp(min=0.0)
    diffs = means - sed.view(1, -1)
    real_noise = torch.abs(means) * multiplicative_noise + 1e-8
    multiplier = -0.5 * real_noise ** (-2)
    results = torch.multiply(multiplier, torch.square(diffs)).sum(1)

    results = torch.nan_to_num(results, nan=-torch.inf)
    kwargs["model"] = kwargs["model"].to(device)
    return results.detach().cpu()


def mh_step(params, context, target_fcn, **kwargs):
    z_min = kwargs["z_min"]
    z_max = kwargs["z_max"]
    z_to_use = params
    proposed_particles = (
        D.Normal(z_to_use, 0.1).sample().clamp(min=z_min, max=z_max).float()
    )
    lps_curr = torch.nan_to_num(target_fcn(z_to_use), -torch.inf)
    lps_new = torch.nan_to_num(target_fcn(proposed_particles), -torch.inf)
    lp_ratios = torch.nan_to_num(lps_new - lps_curr, -torch.inf)
    lp_ratios = torch.exp(lp_ratios).clamp(min=0.0, max=1.0)
    flips = D.Bernoulli(lp_ratios).sample()
    indices_old = torch.arange(len(flips))[flips == 0]
    indices_new = torch.arange(len(flips))[flips == 1]
    new = torch.empty(proposed_particles.shape).float()
    new[indices_new] = proposed_particles[indices_new].float()
    new[indices_old] = z_to_use[indices_old].float()
    return new


def proposal(params, context, target_fcn, **kwargs):
    """
    Given a 'params' object of size N x c x ...,
    where N is the number of particles, c is current number
    of SMC steps, ... is remaining dimension.

    Returns propoal object q(z_{c+1} | z_{1:c})

    We propose using the current encoder q_\phi(z \mid x)

    We propose using most recent step z_{c-1} by random walk, i.e.
    q(z_{c+1} | z_{1:c}) = N(z_c, \sigma)
    """
    new = params
    for _ in range(50):
        new = mh_step(new, context, target_fcn, **kwargs)
    return new


def setup(cfg):

    my_t_priors = [
        TransformedFlatDirichlet(dim=4),
        TransformedUniform(0.0, 1.0),
        TransformedUniform(1e-2, 13.27),
        TransformedUniform(4.5e-5, 1.5e-2),
        TransformedUniform(4.5e-5, 1.5e-2),
        TransformedUniform(0.0, 3.0),
        TransformedUniform(0.0, 3.0),
        TransformedUniform(-2.0, 1.0),
    ]

    sizes = [3, 1, 1, 1, 1, 1, 1, 1]
    sizes_transformed = [4, 1, 1, 1, 1, 1, 1, 1]
    jitter = cfg.data.jitter
    z_min = torch.tensor(
        [
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    z_max = torch.tensor(
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    )

    K = cfg.smc.K
    refresh_every = cfg.smc.refresh_rate
    n_pts = cfg.data.n_pts
    multiplicative_noise = cfg.data.multiplicative_noise
    scale = cfg.data.scale
    smooth = cfg.data.smooth
    smooth_parameter = cfg.data.smooth_parameter
    obs_grid = np.arange(3000.0, 10000.0, 5.0)
    device = cfg.training.device
    kwargs = {
        "K": K,
        "n_pts": n_pts,
        "sizes": sizes,
        "sizes_transformed": sizes_transformed,
        "jitter": jitter,
        "z_min": z_min,
        "z_max": z_max,
        "obs_grid": obs_grid,
        "my_t_priors": my_t_priors,
        "multiplicative_noise": multiplicative_noise,
        "smooth": smooth,
        "refresh_every": refresh_every,
        "scale": scale,
        "smooth_parameter": smooth_parameter,
        "log_target": log_target,
        "proposal": proposal,
        "device": device,
    }

    # Set up emulator
    z_dim = 10
    x_dim = 1400
    emulator = PROVABGSEmulator(dim_in=z_dim, dim_out=x_dim)
    emulator.load_state_dict(
        torch.load(
            "./provabgs/emulator_weights/weights_min=10,max=11,epochs=2000.pth",
            map_location=device,
        )
    )
    my_emulator_copy = copy.deepcopy(emulator)
    my_emulator_copy.to(device)
    my_emulator_copy.eval()
    kwargs["model"] = my_emulator_copy

    del emulator

    thetas, seds = generate_data_emulator(n_pts, True, **kwargs)

    epochs = cfg.training.epochs
    device = cfg.training.device
    mb_size = cfg.training.mb_size

    kwargs["mb_size"] = mb_size
    kwargs["device"] = device
    kwargs["epochs"] = epochs

    # Set up encoder
    z_dim = thetas.shape[-1]
    x_dim = seds.shape[-1]
    fake_zs = torch.randn((K * mb_size, z_dim))
    fake_xs = torch.randn((K * mb_size, x_dim))
    encoder = build_nsf(
        fake_zs,
        fake_xs,
        z_score_x="structured",
        z_score_y="structured",
        hidden_features=128,
        embedding_net=EmbeddingNet(len(obs_grid)).float(),
    )

    # Set up logging string
    kwargs["encoder"] = encoder
    mdn = False
    flow = True
    logger_string = "{},{},{},{},mult={},smooth={},refresh={}".format(
        cfg.training.loss,
        "flow",
        cfg.training.lr,
        K,
        multiplicative_noise,
        smooth,
        refresh_every,
    )
    encoder.to(device)
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)
    # optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Select loss function
    loss_name = cfg.training.loss
    kwargs["loss"] = loss_name

    smc_meta_samplers = []
    if ("smc" in loss_name) and (cfg.smc.run):
        for j in range(len(seds)):
            init_samplers = []
            for n_samp in range(cfg.smc.num_init):
                # Set up SMC
                particles = prior_t_sample(K, **kwargs)
                if loss_name == "vsmc":
                    particles.requires_grad_(True)
                particles = particles.unsqueeze(1)
                init_log_weights = torch.zeros((K, 1))
                init_weights = (nn.Softmax(0)(init_log_weights)).view(-1)
                final_target_fcn = lambda z: log_t_prior(z, **kwargs) + log_target(
                    z, seds[j].cpu(), **kwargs
                )

                SMC = LikelihoodTemperedSMC(
                    particles,
                    init_weights,
                    init_log_weights,
                    final_target_fcn,
                    None,
                    log_t_prior,
                    log_target,
                    proposal,
                    max_mc_steps=100,
                    context=seds[j].cpu(),
                    z_min=z_min,
                    z_max=z_max,
                    kwargs=kwargs,
                )

                # Run SMC
                SMC.run()
                init_samplers.append(SMC)
            keep_all = True
            this_point_meta_IS = MetaImportanceSampler(
                init_samplers, seds, j, keep_all=keep_all, n_resample=cfg.smc.n_resample
            )
            smc_meta_samplers.append(this_point_meta_IS)

    return (
        thetas,
        seds,
        epochs,
        device,
        mb_size,
        encoder,
        smc_meta_samplers,
        mdn,
        flow,
        logger_string,
        optimizer_encoder,
        refresh_every,
        kwargs,
    )
