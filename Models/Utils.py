import torch as th
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoder, self).__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t.view(-1), self.dim*2)


class StupidPositionalEncoder(nn.Module):
    def __init__(self, T_MAX):
        super(StupidPositionalEncoder, self).__init__()
        self.T_MAX = T_MAX

    def forward(self, t):
        return t.float() / self.T_MAX


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
