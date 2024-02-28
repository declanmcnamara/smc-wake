import torch
from einops import rearrange, reduce, repeat


def exact_posteriors(zs, xs, A, sigma, tau):
    """
    For this particular problem, returns the exact posterior for the
    generative model
    Z \sim N(0, \sigma^2 I_k)
    X | Z \sim N(AZ, \tau^2 I_n)

    The exact posterior can be derived by completing the square, e.g.
    https://davidrosenberg.github.io/mlcourse/Notes/completing-the-square.pdf
    """
    n_pts = xs.shape[0]
    k = zs.shape[1]
    M = torch.eye(k) / (sigma**2) + (A.T @ A) / (tau**2)
    bs = (A.T @ xs.T).T / (tau**2)
    rep_M_inv = repeat(torch.inverse(M), "k1 k2 -> n_pts k1 k2", n_pts=n_pts)
    bs = rearrange(bs, "n_pts k -> n_pts k 1")
    posterior_means = rearrange(torch.bmm(rep_M_inv, bs), "n_pts k 1 -> n_pts k")
    posterior_cov = torch.inverse(M)
    return posterior_means, posterior_cov


# Credit: Eric Cousineau
# https://gist.github.com/EricCousineau-TRI/cc2dc27c7413ea8e5b4fd9675050b1c0
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
