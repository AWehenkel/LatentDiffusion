import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DCDecoder, DCEncoder, AsynchronousDiffuser, ImprovedTransitionNet, SimpleImageDecoder, SimpleImageEncoder, PositionalEncoder, StupidPositionalEncoder
import math

class VAEModel(nn.Module):
    def __init__(self, **kwargs):
        super(VAEModel, self).__init__()
        self.latent_s = kwargs['latent_s']
        self.CNN = kwargs['CNN']
        self.device = 'cpu'
        self.img_size = kwargs['img_size']

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']

        if self.CNN:
            self.enc = DCEncoder(self.img_size, self.latent_s*2, enc_net, t_dim=0)
            self.dec = DCDecoder(self.enc.features_dim, [self.latent_s], dec_net, t_dim=0,
                                          out_c=self.img_size[0], img_width=self.img_size[1])
        else:
            self.enc = TemporalEncoder(32**2, self.latent_s*2, enc_net, 0)
            self.dec = TemporalDecoder(32**2, self.latent_s, dec_net, 0)

    def loss(self, x0):
        bs = x0.shape[0]

        # Encoding
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), self.latent_s, 1)
        KL_z = (-log_sigma_z/2 + (mu_z ** 2)/2 + torch.exp(log_sigma_z)/2).sum(1)

        # Decoding
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device))
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1)


        loss = KL_x.mean(0) + KL_z.mean(0)

        return loss

    def forward(self, x0):
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), self.latent_s, 1)
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device))
        return mu_x_pred

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1):
        z = torch.randn(64, self.latent_s).to(self.device)
        x = self.dec(z).view(nb_samples, -1)

        return x


class DDPMPriorVAEModel(nn.Module):
    def __init__(self, **kwargs):
        super(DDPMPriorVAEModel, self).__init__()
        self.latent_s = kwargs['latent_s']
        self.CNN = kwargs['CNN']
        self.device = 'cpu'
        self.img_size = kwargs['img_size']

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']
        trans_net = [[kwargs['trans_w']] * kwargs['trans_l']] * kwargs['n_res_blocks']

        self.t_emb_s = kwargs['t_emb_s']
        if self.t_emb_s > 1:
            self.t_emb_s = (self.t_emb_s // 2) * 2
            pos_enc = PositionalEncoder(self.t_emb_s // 2)
        else:
            pos_enc = StupidPositionalEncoder(self.T_MAX)

        if self.CNN:
            self.enc = DCEncoder(self.img_size, sum(self.latent_s), enc_net, t_dim=0)
            self.dec = DCDecoder(self.enc.features_dim, self.latent_s, dec_net, t_dim=0,
                                          out_c=self.img_size[0], img_width=self.img_size[1])
        else:
            self.enc = TemporalEncoder(32**2, sum(self.latent_s), enc_net, 0)
            self.dec = TemporalDecoder(32**2, self.latent_s, dec_net, 0)

        self.diffuser = AsynchronousDiffuser(betas_min=kwargs['beta_min'], betas_max=kwargs['beta_max'],
                                             ts_min=kwargs['t_min'], ts_max=kwargs['t_max'], var_sizes=self.latent_s)
        self.trans_net = ImprovedTransitionNet(sum(self.latent_s), trans_net, self.t_emb_s, self.diffuser,
                                                       pos_enc=pos_enc,
                                                       simplified_trans=kwargs['simplified_trans'])
        self.T = self.diffuser.T


    def loss(self, x0):
        bs = x0.shape[0]
        dev = x0.device
        # Encoding
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), sum(self.latent_s), 1)
        z0 = mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=dev)

        entropy_posterior_z = log_sigma_z.sum(1) + math.log(2*(math.pi*math.e)**.5)
        t = torch.randint(1, self.T, (bs, 1), device=dev)

        # HERE we trick to compute p(z_T|x) directly in order to take into account the known randomness of z_0
        _, (mu_T, sigma_T) = self.diffuser.diffuse(mu_z, t * 0 + self.T, t*0)
        sigma_T = (sigma_T ** 2 - (sigma_T ** 2 - 1) * torch.exp(2 * log_sigma_z)).sqrt()

        zt, _ = self.diffuser.diffuse(z0, t)

        if self.T >= 1:
            mu_zt_1, sigma_zt_1 = self.diffuser.prev_mean_var(zt, z0, t)
            zt_1_rec = self.trans_net(zt, t - 1)

            KL_rev_diffusion = (((mu_zt_1 - zt_1_rec) ** 2)).sum(1)
        else:
            KL_rev_diffusion = 0.


        KL_prior_diffusion = (-torch.log(sigma_T) + (mu_T ** 2)/2 + .5*sigma_T**2).sum(1)

        KL_pst_z_prior_z = entropy_posterior_z - (KL_prior_diffusion + KL_rev_diffusion)

        # Decoding
        mu_x_pred = self.dec(z0)
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1)

        loss = KL_x.mean(0) - KL_pst_z_prior_z.mean(0)

        return loss

    def forward(self, x0):
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), sum(self.latent_s), 1)
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device))
        return mu_x_pred

    def to(self, device):
        super().to(device)
        self.trans_net.to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1):
        if self.T >= 1:
            z = self.trans_net.sample(nb_samples)
        else:
            z = torch.randn(nb_samples, sum(self.latent_s)).to(self.device)
        x = self.dec(z).view(nb_samples, -1)

        return x

