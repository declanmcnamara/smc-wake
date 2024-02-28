import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat


class Encoder(nn.Module):
    def __init__(self, n, k):
        super(Encoder, self).__init__()
        self.n = n
        self.k = k
        self.network = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, k * 2),
        )

    def forward(self, x):
        out = self.network(x)
        return out

    def get_q(self, x):
        out = self.forward(x)
        mean, log_sd = out[..., : self.k], out[..., self.k :]
        log_sd = log_sd.clamp(min=-10.0, max=10.0)
        sd = torch.exp(log_sd)
        return D.MultivariateNormal(mean, torch.diag_embed(sd) ** 2)


# class FullRankEncoder(nn.Module):
#     def __init__(self, n, k, device):
#         super(FullRankEncoder, self).__init__()
#         self.n = n
#         self.k = k
#         self.device = device
#         self.network = nn.Sequential(
#             nn.Linear(n, 64),
#             nn.ReLU(),
#             nn.Linear(64,64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, k + int(k*(k+1)/2)),
#         )

#     def forward(self, x):
#         out = self.network(x)
#         return out

#     def get_q(self, x):
#         batch_size = x.shape[0]
#         out = self.forward(x)
#         mean, cov = out[...,:self.k], out[...,self.k:]
#         diagonal = cov[...,:self.k]
#         #diagonal = torch.exp(cov[...,:self.k])
#         rest = cov[...,self.k:]
#         m = torch.zeros((batch_size, self.k, self.k)).to(self.device)
#         tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=-1)
#         m[..., tril_indices[0], tril_indices[1]] = rest
#         Ls = torch.diag_embed(diagonal) + m
#         cov_matrices = torch.bmm(Ls, Ls.transpose(1,2))


#         # #x = torch.arange(6) + 1
#         # xc = torch.cat([cov[-self.k:], cov.flip(dims=[0])])
#         # y = xc.view(self.k, self.k)
#         # torch.tril(y)

#         # tensor([[4, 0, 0],
#         #         [6, 5, 0],
#         #         [3, 2, 1]])

#         # m = torch.zeros((self.k, self.k))
#         # tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=0)
#         # m[tril_indices[0], tril_indices[1]] = cov

#         # Exp diagonal for log Cholesky parameterization


#         # indices = torch.tril_indices(row=self.k, col=self.k)
#         # diagonal = cov[...,]
#         # log_sd = log_sd.clamp(min=-10., max=10.)
#         # sd = torch.exp(log_sd)
#         return D.MultivariateNormal(mean, cov_matrices)

# class FullRankEncoder(nn.Module):
#     def __init__(self, n, k, device):
#         super(FullRankEncoder, self).__init__()
#         self.n = n
#         self.k = k
#         self.device = device
#         self.network = nn.Sequential(
#             nn.Linear(n, 64),
#             nn.ReLU(),
#             nn.Linear(64,64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, k + int(k*(k+1)/2)),
#         )

#     def forward(self, x):
#         out = self.network(x)
#         return out

#     def get_q(self, x):
#         batch_size = x.shape[0]
#         out = self.forward(x)
#         mean, cov_entries = out[...,:self.k], out[...,self.k:]#.reshape(batch_size, self.k,self.k)
#         m = torch.zeros((batch_size, self.k, self.k)).to(self.device)
#         tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=0)
#         m[..., tril_indices[0], tril_indices[1]] = cov_entries

#         jitter = 1e-4
#         jitter = torch.tensor(jitter)
#         diagonal = jitter.repeat(self.k)
#         cov_matrices = torch.bmm(m, m.transpose(1,2))+torch.diag(diagonal).to(self.device)


#         # #x = torch.arange(6) + 1
#         # xc = torch.cat([cov[-self.k:], cov.flip(dims=[0])])
#         # y = xc.view(self.k, self.k)
#         # torch.tril(y)

#         # tensor([[4, 0, 0],
#         #         [6, 5, 0],
#         #         [3, 2, 1]])

#         # m = torch.zeros((self.k, self.k))
#         # tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=0)
#         # m[tril_indices[0], tril_indices[1]] = cov

#         # Exp diagonal for log Cholesky parameterization


#         # indices = torch.tril_indices(row=self.k, col=self.k)
#         # diagonal = cov[...,]
#         # log_sd = log_sd.clamp(min=-10., max=10.)
#         # sd = torch.exp(log_sd)
#         return D.MultivariateNormal(mean, cov_matrices)


class FullRankEncoder(nn.Module):
    def __init__(self, n, k, device):
        super(FullRankEncoder, self).__init__()
        self.n = n
        # self.d = d
        self.k = k
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(n, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, k + int(k * (k + 1) / 2)),
        )

    def forward(self, x):
        out = self.network(x)
        return out

    def get_q(self, x):
        batch_size = x.shape[0]
        d = x.shape[1]
        out = self.forward(x)
        mean, cov_entries = (
            out[..., : self.k],
            out[..., self.k :],
        )  # .reshape(batch_size, self.k,self.k)
        m = torch.zeros((batch_size, d, self.k, self.k)).to(self.device)
        tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=0)
        m[..., tril_indices[0], tril_indices[1]] = cov_entries

        jitter = 1e-4
        jitter = torch.tensor(jitter)
        diagonal = torch.diag(jitter.repeat(self.k))
        r_diagonal = repeat(
            diagonal, "k1 k2 -> batch d k1 k2", batch=batch_size, d=d
        ).to(self.device)
        temp = rearrange(m, "b d k1 k2 -> (b d) k1 k2")
        mmT = torch.bmm(temp, rearrange(temp, "c k1 k2 -> c k2 k1"))
        mmT_reshp = rearrange(mmT, "(b d) k1 k2 -> b d k1 k2", b=batch_size, d=d)
        cov_matrices = mmT_reshp + r_diagonal

        # #x = torch.arange(6) + 1
        # xc = torch.cat([cov[-self.k:], cov.flip(dims=[0])])
        # y = xc.view(self.k, self.k)
        # torch.tril(y)

        # tensor([[4, 0, 0],
        #         [6, 5, 0],
        #         [3, 2, 1]])

        # m = torch.zeros((self.k, self.k))
        # tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=0)
        # m[tril_indices[0], tril_indices[1]] = cov

        # Exp diagonal for log Cholesky parameterization

        # indices = torch.tril_indices(row=self.k, col=self.k)
        # diagonal = cov[...,]
        # log_sd = log_sd.clamp(min=-10., max=10.)
        # sd = torch.exp(log_sd)
        return D.MultivariateNormal(mean, cov_matrices)
