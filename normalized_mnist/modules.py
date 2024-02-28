import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch.nn.functional import one_hot


# Credit: https://gist.github.com/EricCousineau-TRI/cc2dc27c7413ea8e5b4fd9675050b1c0
def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


class SBN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W0 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W0 = nn.init.xavier_uniform_(self.W0)
        self.b0 = torch.empty((dim_out,), requires_grad=True)
        self.b0 = nn.init.constant_(self.b0, 0.0)
        self.W1 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W1 = nn.init.xavier_uniform_(self.W1)
        self.b1 = torch.empty((dim_out,), requires_grad=True)
        self.b1 = nn.init.constant_(self.b1, 0.0)
        self.W2 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W2 = nn.init.xavier_uniform_(self.W2)
        self.b2 = torch.empty((dim_out,), requires_grad=True)
        self.b2 = nn.init.constant_(self.b2, 0.0)
        self.W3 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W3 = nn.init.xavier_uniform_(self.W3)
        self.b3 = torch.empty((dim_out,), requires_grad=True)
        self.b3 = nn.init.constant_(self.b3, 0.0)
        self.W4 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W4 = nn.init.xavier_uniform_(self.W4)
        self.b4 = torch.empty((dim_out,), requires_grad=True)
        self.b4 = nn.init.constant_(self.b4, 0.0)
        self.W5 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W5 = nn.init.xavier_uniform_(self.W5)
        self.b5 = torch.empty((dim_out,), requires_grad=True)
        self.b5 = nn.init.constant_(self.b5, 0.0)
        self.W6 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W6 = nn.init.xavier_uniform_(self.W6)
        self.b6 = torch.empty((dim_out,), requires_grad=True)
        self.b6 = nn.init.constant_(self.b6, 0.0)
        self.W7 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W7 = nn.init.xavier_uniform_(self.W7)
        self.b7 = torch.empty((dim_out,), requires_grad=True)
        self.b7 = nn.init.constant_(self.b7, 0.0)
        self.W8 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W8 = nn.init.xavier_uniform_(self.W8)
        self.b8 = torch.empty((dim_out,), requires_grad=True)
        self.b8 = nn.init.constant_(self.b8, 0.0)
        self.W9 = torch.empty((dim_out, dim_in), requires_grad=True)
        self.W9 = nn.init.xavier_uniform_(self.W9)
        self.b9 = torch.empty((dim_out,), requires_grad=True)
        self.b9 = nn.init.constant_(self.b9, 0.0)
        self.megaW = nn.Parameter(
            torch.stack(
                [
                    self.W0,
                    self.W1,
                    self.W2,
                    self.W3,
                    self.W4,
                    self.W5,
                    self.W6,
                    self.W7,
                    self.W8,
                    self.W9,
                ]
            )
        )
        self.megab = nn.Parameter(
            torch.stack(
                [
                    self.b0,
                    self.b1,
                    self.b2,
                    self.b3,
                    self.b4,
                    self.b5,
                    self.b6,
                    self.b7,
                    self.b8,
                    self.b9,
                ]
            )
        )
        self.jitter = 1e-2

    def forward(self, theta, label):
        assert theta.shape[-1] == self.dim_in
        if len(theta.shape) == 2:
            rtheta = repeat(theta, "b d -> 10 d b")
            multiplied = torch.bmm(self.megaW, rtheta)
            multiplied = rearrange(multiplied, "ndigit d b -> b ndigit d")
            added = multiplied + self.megab
            if label.shape[0] == theta.shape[0]:
                selected = vector_gather(added, label)
            else:
                rlabel = label.repeat(theta.shape[0])
                selected = vector_gather(added, rlabel)
        else:
            rtheta = repeat(theta, "K b d -> K 10 d b")
            K = rtheta.shape[0]
            rtheta = rearrange(rtheta, "K ndig d b -> (K ndig) d b")
            rmegaW = repeat(
                self.megaW,
                "ndig dig_dim latent_dim -> (K ndig) dig_dim latent_dim",
                K=K,
            )
            multiplied = torch.bmm(rmegaW, rtheta)
            multiplied = rearrange(multiplied, "(K ndig) d b -> K b ndig d", K=K)
            added = multiplied + self.megab
            label = repeat(label, "b 1 -> K b", K=K)
            rearr_added = rearrange(added, "K b ndig d -> (K b) ndig d")
            rearr_label = rearrange(label, "K b -> (K b)")
            selected = vector_gather(rearr_added, rearr_label)
            selected = rearrange(selected, "(K b) d -> K b d", K=K)

        probs = torch.sigmoid(selected)
        # probs = probs.clamp(min=self.jitter, max=1-self.jitter)
        return probs

    def sample(self, theta, label):
        probs = self.forward(theta, label)
        distr = D.Normal(probs, self.jitter)
        return distr.sample()

    def _log_prob(self, theta, label, x):
        assert theta.shape[0] == x.shape[0]
        probs = self.forward(theta, label)
        # distr = D.Bernoulli(probs)
        distr = D.Normal(probs, self.jitter)
        return distr.log_prob(x).sum(-1)

    def log_prob(self, theta, label, x):
        if theta.shape[0] == x.shape[0]:
            if len(theta.shape) == len(x.shape):
                return self._log_prob(theta, label, x)
            else:
                reshaped_x = x.unsqueeze(1).repeat(1, theta.shape[1], 1)
                return self._log_prob(theta, label, reshaped_x)
        elif len(x.shape) == 1:
            if len(theta.shape) == 2:
                batch_size = theta.shape[0]
                return self._log_prob(theta, label, x.repeat(batch_size, 1))
            else:
                batch_size = theta.shape[0]
                K = theta.shape[1]
                return self._log_prob(theta, label, x.repeat((batch_size, K, 1)))
        else:
            return self._log_prob(theta, label, x.repeat(theta.shape[0], 1, 1))


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, x):
        assert x.shape[-1] == self.obs_dim
        return self.net(x)

    def get_q(self, x):
        outputs = self.forward(x)
        means, sds = (
            outputs[..., : self.latent_dim],
            outputs[..., self.latent_dim :].clamp(-10.0, 10.0).exp(),
        )
        return D.Normal(means, sds)

    def log_prob(self, theta, x):
        distr = self.get_q(x)
        return distr.log_prob(theta)
