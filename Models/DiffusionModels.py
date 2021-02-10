import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DataDiffuser, TransitionNet, SimpleImageDecoder, SimpleImageEncoder, PositionalEncoder, StupidPositionalEncoder


class ProperDiffusionModel(nn.Module):
    def __init__(self, **kwargs):
        super(ProperDiffusionModel, self).__init__()
        self.T_MAX = kwargs['T_MAX']
        self.latent_s = kwargs['latent_s']
        self.t_emb_s = kwargs['t_emb_s']
        #self.CNN = kwargs['CNN']
        self.register_buffer("beta_min", torch.tensor(kwargs['beta_min']))
        self.register_buffer("beta_max", torch.tensor(kwargs['beta_max']))
        self.device = 'cpu'
        self.img_size = kwargs['img_size']
        self.L_simple = kwargs['L_simple']
        if self.t_emb_s > 1:
            self.pos_enc = PositionalEncoder(self.t_emb_s // 2)
        else:
            self.pos_enc = StupidPositionalEncoder(self.T_MAX)

        self.simplified_trans = kwargs['simplified_trans']

        trans_net = [kwargs['trans_w']] * kwargs['trans_l']
        self.trans = TransitionNet(self.latent_s, trans_net, self.t_emb_s)
        self.dif = DataDiffuser(beta_min=self.beta_min, beta_max=self.beta_max, t_max=self.T_MAX)

    def loss(self, x_0):

        t0 = torch.zeros(x_0.shape[0]).to(self.device).long()

        t = torch.torch.distributions.Uniform(t0.float(), torch.ones_like(t0) * self.T_MAX).sample().long().to(self.device)

        epsilon = torch.randn(x_0.shape, device=self.device)
        x_t, sigma_z = self.dif.diffuse(x_0, t, t0, epsilon)

        if self.simplified_trans:
            alpha_bar_t = self.dif.alphas[t].unsqueeze(1)
            alpha_t = self.dif.alphas_t[t].unsqueeze(1)
            beta_t = self.dif.betas[t].unsqueeze(1)
            epsilon_pred = self.trans(x_t, self.pos_enc(t.float().unsqueeze(1)))
            mu_x_pred = (x_t - beta_t/(1-alpha_bar_t).sqrt() * epsilon_pred)/alpha_t.sqrt()
        else:
            mu_x_pred = self.trans(x_t, self.pos_enc(t.float().unsqueeze(1)))
        mu, sigma = self.dif.prev_mean(x_0, x_t, t)

        if self.L_simple:
            KL = ((mu - mu_x_pred) ** 2).sum(1)
        else:
            KL = ((mu - mu_x_pred) ** 2).sum(1) / sigma ** 2

        loss = KL.mean(0)

        return loss

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1):
        zT = torch.randn(nb_samples, self.latent_s).to(self.device)
        z_t = zT
        for t in range(self.T_MAX - 1, 0, -1):
            t_t = torch.ones(64, 1).to(self.device) * t
            if t > 0:
                sigma = ((1 - self.dif.alphas[t - 1]) / (1 - self.dif.alphas[t]) * self.dif.betas[t]).sqrt()
            else:
                sigma = 0
            if self.simplified_trans:
                alpha_bar_t = self.dif.alphas[t]
                alpha_t = self.dif.alphas_t[t]
                beta_t = self.dif.betas[t]
                mu_z_pred = (z_t - beta_t / (1 - alpha_bar_t).sqrt() * self.trans(z_t, self.pos_enc(t_t))) / alpha_t.sqrt()
            else:
                mu_z_pred = self.trans(z_t, self.pos_enc(t_t))
            z_t = mu_z_pred + torch.randn(z_t.shape, device=self.device) * sigma

        x_0 = z_t

        return x_0


class CNNDiffusionModel(nn.Module):
    def __init__(self, **kwargs):
        super(CNNDiffusionModel, self).__init__()
        self.T_MAX = kwargs['T_MAX']
        self.latent_s = kwargs['latent_s']
        self.t_emb_s = kwargs['t_emb_s']
        self.CNN = kwargs['CNN']
        self.register_buffer("beta_min", torch.tensor(kwargs['beta_min']))
        self.register_buffer("beta_max", torch.tensor(kwargs['beta_max']))
        self.device = 'cpu'
        self.img_size = kwargs['img_size']
        if self.t_emb_s > 1:
            self.pos_enc = PositionalEncoder(self.t_emb_s // 2)
        else:
            self.pos_enc = StupidPositionalEncoder(self.T_MAX)
        self.simplified_trans = kwargs['simplified_trans']

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']
        trans_net = [kwargs['trans_w']] * kwargs['trans_l']

        if self.CNN:
            self.enc = SimpleImageEncoder(self.img_size, self.latent_s, enc_net, t_dim=self.t_emb_s)
            self.dec = SimpleImageDecoder(self.enc.features_dim, self.latent_s, dec_net, t_dim=self.t_emb_s,
                                          out_c=self.img_size[0])
        else:
            self.dec = TemporalDecoder(32**2, self.latent_s, dec_net, self.t_emb_s)
            self.enc = TemporalEncoder(32**2, self.latent_s, enc_net, self.t_emb_s)

        self.trans = TransitionNet(self.latent_s, trans_net, self.t_emb_s)
        self.dif = DataDiffuser(beta_min=self.beta_min, beta_max=self.beta_max, t_max=self.T_MAX)
        self.sampling_t0 = False

    def loss(self, x0):
        bs = x0.shape[0]

        if self.sampling_t0:
            t0 = torch.randint(0, self.T_MAX - 1, [x0.shape[0]]).to(self.device)
            x_t0, sigma_x_t0 = self.dif.diffuse(x0, t0, torch.zeros(x0.shape[0]).long().to(self.device))
        else:
            t0 = torch.zeros(x0.shape[0]).to(self.device).long()
            x_t0 = x0

        z_t0 = self.enc(x_t0.view(-1, *self.img_size), self.pos_enc(t0.float().unsqueeze(1)))
        # z_t0 = z_t0 + torch.randn(z_t0.shape).to(dev) * (1 - dif.alphas[t0]).sqrt().unsqueeze(1).expand(-1, z_t0.shape[1])
        t = torch.torch.distributions.Uniform(t0.float() + 1, torch.ones_like(t0) * self.T_MAX).sample().long().to(self.device)

        z_t, sigma_z = self.dif.diffuse(z_t0, t, t0)
        x_t, sigma_x = self.dif.diffuse(x_t0, t, t0)

        mu_x_pred = self.dec(z_t, self.pos_enc(t.float().unsqueeze(1)))
        KL_x = ((mu_x_pred - x_t.view(bs, *self.img_size)) ** 2).view(bs, -1).sum(1) / sigma_x ** 2

        if self.simplified_trans:
            alpha_bar_t = self.dif.alphas[t].unsqueeze(1)#.expand(-1, self.latent_s)
            alpha_t = self.dif.alphas_t[t].unsqueeze(1)#.expand(-1, self.latent_s)
            beta_t = self.dif.betas[t].unsqueeze(1)#.expand(-1, self.latent_s)

            mu_z_pred = (z_t - beta_t/(1-alpha_bar_t).sqrt() * self.trans(z_t, self.pos_enc(t.float().unsqueeze(1))))/alpha_t.sqrt()
        else:
            mu_z_pred = self.trans(z_t, self.pos_enc(t.float().unsqueeze(1)))
        mu, sigma = self.dif.prev_mean(z_t0, z_t, t)

        KL_z = ((mu - mu_z_pred) ** 2).sum(1) / sigma ** 2

        loss = KL_x.mean(0) + KL_z.mean(0)

        return loss

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1):
        zT = torch.randn(64, self.latent_s).to(self.device)
        z_t = zT
        for t in range(self.T_MAX - 1, 0, -1):
            t_t = torch.ones(64, 1).to(self.device) * t
            if t > 0:
                sigma = ((1 - self.dif.alphas[t - 1]) / (1 - self.dif.alphas[t]) * self.dif.betas[t]).sqrt()
            else:
                sigma = 0
            if self.simplified_trans:
                alpha_bar_t = self.dif.alphas[t]
                alpha_t = self.dif.alphas_t[t]
                beta_t = self.dif.betas[t]
                mu_z_pred = (z_t - beta_t / (1 - alpha_bar_t).sqrt() * self.trans(z_t, self.pos_enc(t_t))) / alpha_t.sqrt()
            else:
                mu_z_pred = self.trans(z_t, self.pos_enc(t_t))
            z_t = mu_z_pred + torch.randn(z_t.shape, device=self.device) * sigma

        x_0 = self.dec(z_t, self.pos_enc(torch.zeros((nb_samples, 1), device=self.device))).view(nb_samples, -1)

        return x_0

