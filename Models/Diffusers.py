import torch
import torch.nn as nn


class DataDiffuser(nn.Module):
    def __init__(self, beta_min=1e-4, beta_max=.02, t_min=0, t_max=1000):
        super(DataDiffuser, self).__init__()
        self.register_buffer('betas', torch.arange(beta_min, beta_max + 1e-10, (beta_max - beta_min) / (t_max - t_min)))
        self.register_buffer('alphas_t', (1 - self.betas))
        self.register_buffer('alphas', self.alphas_t.log().cumsum(0).exp())

    def diffuse(self, x_t0, t, t0=0, noise=None):
        if noise is None:
            noise = torch.randn(x_t0.shape).to(x_t0.device)
        alpha_t0 = 1 * (t0 == 0).float() + (1 - (t0 == 0).float()) * self.alphas[t0 - 1]

        mu = x_t0 * (self.alphas[t] / alpha_t0).sqrt().unsqueeze(1).expand(-1, x_t0.shape[1]).float()
        # mu = x_t0 * self.alphas[t].sqrt().unsqueeze(1).expand(-1, x_t0.shape[1]).float()
        sigma_t = ((self.alphas[t] / alpha_t0) * (1 - alpha_t0) + (1 - self.alphas[t])).sqrt()
        sigma = sigma_t.unsqueeze(1).expand(-1, x_t0.shape[1]).float()
        # sigma = (1 - self.alphas[t].unsqueeze(1).expand(-1, x_t0.shape[1]).float()).sqrt()
        return mu + noise * sigma, sigma_t

    def prev_mean(self, x_t, x_0, t):
        alphas = self.alphas.unsqueeze(1).expand(-1, x_t.shape[1]).float()
        betas = self.betas.unsqueeze(1).expand(-1, x_t.shape[1]).float()
        alphas_t = self.alphas_t.unsqueeze(1).expand(-1, x_t.shape[1]).float()
        mu = alphas[t - 1].sqrt() * betas[t] * x_0 / (1 - alphas[t]) + alphas_t[t].sqrt() * (
                    1 - alphas[t - 1]) * x_t / (1 - alphas[t])
        sigma = ((1 - self.alphas[t - 1]) / (1 - self.alphas[t]) * self.betas[t]).sqrt()
        return mu, sigma


class TransitionNet(nn.Module):
    def __init__(self, z_dim, layers, t_dim=1, act=nn.ReLU):
        super(TransitionNet, self).__init__()
        layers = [z_dim + t_dim] + layers + [z_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)

    def forward(self, z, t):
        return self.net(torch.cat((z, t), 1)) #+ z
