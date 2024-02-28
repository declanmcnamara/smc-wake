import torch
import torch.distributions as D
import torch.nn as nn


class TransformedUniform(nn.Module):
    """
    A Uniform Random variable defined on the real line.
    Transformed by sigmoid to reside between low and high.
    """

    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high
        self.length = high - low
        self.instance = D.Uniform(low=self.low, high=self.high)
        self.jitter = 1e-8

    def transform(self, value: torch.Tensor):
        tt = torch.sigmoid(value) * self.length + self.low
        clamped = tt.clamp(min=self.low + self.jitter, max=self.high - self.jitter)
        return clamped

    def inv_transform(self, tval: torch.Tensor):
        assert (self.low <= tval).all(), "Input is outside of the support."
        assert (self.high >= tval).all(), "Input is outside of the support."
        to_invert = (tval - self.low) / self.length
        return torch.logit(to_invert)

    def log_prob(self, value: torch.Tensor):
        tval = self.transform(value)
        return self.instance.log_prob(tval)

    def sample(self, shape):
        tsamples = self.instance.sample(shape)
        return self.inv_transform(tsamples)


class TransformedFlatDirichlet(nn.Module):
    """
    A transformed dirichlet distribution, where sampling occurs on
    and n-1 dimensional space. We follow the warped manifold tranformation
    from Betancourt (2013) https://arxiv.org/pdf/1010.3436.pdf.

    The transform from an n-1 collection of Unif(0,1) r.v's to a
    n-dimensional Dirichlet is described in the above. We further
    use a logit transform on the (0,1) space to operate on an unconstrained space.
    """

    def __init__(self, dim=4):
        super().__init__()
        self.concentration = torch.ones(dim)
        self.ndim = len(self.concentration)
        self.ndim_sampling = self.ndim - 1

    def transform(self, tt):
        """
        tt is a (...,n-1) shaped tensor.
        Return a (..., n) shaped tensor.
        """
        tt = torch.sigmoid(tt)
        tt_d = torch.empty(tt.shape[:-1] + (self.ndim,))

        tt_d[..., 0] = 1.0 - tt[..., 0]
        for i in range(1, self.ndim_sampling):
            tt_d[..., i] = torch.prod(tt[..., :i], axis=-1) * (1.0 - tt[..., i])
        tt_d[..., -1] = torch.prod(tt, axis=-1)
        return tt_d

    def inv_transform(self, tt_d):
        """reverse the warped manifold transformation
        i.e. go from n dimensions to n-1.

        Afterward, go from n-1 observations on (0,1) to real numbers
        by logit transformation.
        """
        assert tt_d.shape[-1] == self.ndim
        tt = torch.empty(tt_d.shape[:-1] + (self.ndim_sampling,))

        tt[..., 0] = 1.0 - tt_d[..., 0]
        for i in range(1, self.ndim_sampling):
            tt[..., i] = 1.0 - (tt_d[..., i] / torch.prod(tt[..., :i], axis=-1))
        return torch.logit(tt)

    def log_prob(self, theta):
        """
        Assume that theta is on the sampling space. Unsure if valid for anything
        other than a flat dirichlet distribution.
        """
        ttheta = self.transform(theta)
        assert (
            ttheta.shape[-1] == self.ndim
        ), "Provided observations reside in wrong dimensional space"
        return D.Dirichlet(self.concentration).log_prob(ttheta)

    def sample_actual(self, shape):
        return D.Dirichlet(self.concentration).sample(shape)

    def sample(self, shape):
        transformed = self.sample_actual(shape)
        return self.inv_transform(transformed)


class PROVABGSEmulator(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.emulator = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim_out),
        )

    def forward(self, params):
        return self.emulator(params)
