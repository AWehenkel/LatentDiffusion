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


class AsynchronousDiffuser(nn.Module):
    def __init__(self, betas_min, betas_max, ts_min, ts_max, var_sizes):
        super(AsynchronousDiffuser, self).__init__()
        if not (len(betas_max) == len(betas_min) == len(ts_max) == len(ts_min) == len(var_sizes)):
            raise AssertionError

        t0 = 0
        T = max(ts_max)
        self.T = T
        betas = []
        alphas_t = []
        alphas = []
        for b_min, b_max, t_min, t_max, var_size in zip(betas_min, betas_max, ts_min, ts_max, var_sizes):
            beta = torch.zeros(var_size, T + 1)
            beta[:, t_min:t_max+1] = torch.linspace(b_min, b_max, 1 + t_max - t_min)
            alpha_t = 1 - beta
            alpha = alpha_t.cumprod(1)

            betas.append(beta)
            alphas_t.append(alpha_t)
            alphas.append(alpha)

        self.register_buffer('betas', torch.cat(betas, 0).permute(1, 0).float())
        self.register_buffer('alphas_t', torch.cat(alphas, 0).permute(1, 0).float())
        self.register_buffer('alphas', torch.cat(alphas, 0).permute(1, 0).float())

    def diffuse(self, z_t0, t):
        t = t.view(-1)
        mu = z_t0 * self.alphas[t, :].sqrt()
        sigma = (1 - self.alphas[t, :]).sqrt()
        z_t = mu + torch.randn_like(z_t0) * sigma
        return z_t

    def reverse(self, z_t0, z_t, t_1):
        t_1 = t_1.view(-1)

        is_dirac = (self.betas[t_1, :] == 0.).float()
        alphas = self.alphas
        betas = self.betas
        alphas_t = self.alphas_t

        mult_z0 = alphas[t_1, :].sqrt() * betas[t_1 + 1, :] / (1 - alphas[t_1 + 1, :])
        mult_zt = alphas_t[t_1 + 1, :].sqrt() * (1 - alphas[t_1, :]) / (1 - alphas[t_1 + 1, :])
        mult_z0[mult_z0 / mult_z0 != mult_z0 / mult_z0] = 0.
        mult_zt[mult_zt / mult_zt != mult_zt / mult_zt] = 1.

        mu_cond = mult_z0 * z_t0 + mult_zt * z_t

        sigma_cond = ((1 - self.alphas[t_1, :]) / (1 - self.alphas[t_1 + 1, :]) * self.betas[t_1 + 1, :]).sqrt()

        mu_cond[mu_cond/mu_cond != mu_cond/mu_cond] = 0.
        sigma_cond[sigma_cond / sigma_cond != sigma_cond / sigma_cond] = 0.

        mu = z_t * is_dirac + (1 - is_dirac) * mu_cond

        return mu + sigma_cond * torch.randn_like(z_t)

    def past_sample(self, mu_z_pred, t_1):
        return mu_z_pred + torch.randn_like(mu_z_pred) * self.betas[t_1.view(-1), :]


class TransitionNet(nn.Module):
    def __init__(self, z_dim, layers, t_dim=1, diffuser=None, pos_enc=None, act=nn.SELU, simplified_trans=True):
        super(TransitionNet, self).__init__()
        layers = [z_dim + t_dim] + layers + [z_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)
        self.diffuser = diffuser
        self.z_dim = z_dim
        self.device = 'cpu'
        self.pos_enc = pos_enc
        self.simplified_trans = simplified_trans

    def forward(self, z, t):

        t = self.pos_enc(t) if self.pos_enc is not None else t

        mu_z_pred = self.net(torch.cat((z, t), 1))
        return mu_z_pred#self.net(torch.cat((z, t), 1)) #+ z


    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples, t0=0, temperature=1.):
        if self.diffuser is None:
            raise NotImplementedError

        zT = torch.randn(nb_samples, self.z_dim).to(self.device) * temperature
        T = self.diffuser.T
        z_t = zT
        for t in range(T - 1, t0-1, -1):
            t_t = torch.ones(nb_samples, 1).to(self.device).long() * t

            mu_z_pred = self.forward(z_t, t_t)
            if self.simplified_trans:
                alpha_bar_t = self.diffuser.alphas[t_t.view(-1), :]
                alpha_t = self.diffuser.alphas_t[t_t.view(-1), :]
                beta_t = self.diffuser.betas[t_t.view(-1), :]

                is_dirac = torch.logical_or((1 - alpha_bar_t) == 0., alpha_t == 1.)
                beta_t[is_dirac] = 1.
                alpha_bar_t[is_dirac] = 0.
                alpha_t[is_dirac] = 1.
                #print(alpha_t.sqrt().min())
                #print(((1 - alpha_t)/(1 - alpha_bar_t).sqrt()).max())
                mu_z_pred = (z_t - (1 - alpha_t)/(1 - alpha_bar_t).sqrt() * mu_z_pred)/alpha_t.sqrt()
                z_t = z_t * is_dirac.float() + (1 - is_dirac.float()) * self.diffuser.past_sample(mu_z_pred, t_t)
                #print(z_t.norm(), z_t.std(), z_t.mean())
            else:
                z_t = self.diffuser.past_sample(mu_z_pred, t_t)
        print(z_t.norm(), z_t.std(), z_t.mean())
        return z_t


