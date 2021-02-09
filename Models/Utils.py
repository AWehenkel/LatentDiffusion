import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoder, self).__init__()
        self.dim = dim

    def forward(self, t):
        emb = t / torch.exp(torch.arange(self.dim).float() / self.dim * torch.log(torch.ones(1, self.dim) * 100)).to(
            t.device)
        return torch.cat((torch.sin(emb), torch.cos(emb)), 1)


class StupidPositionalEncoder(nn.Module):
    def __init__(self, T_MAX):
        super(StupidPositionalEncoder, self).__init__()
        self.T_MAX = T_MAX

    def forward(self, t):
        return t.float() / self.T_MAX
