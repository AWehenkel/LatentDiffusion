import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DCDecoder, DCEncoder, AsynchronousDiffuser, ImprovedTransitionNet, \
    SimpleImageDecoder, SimpleImageEncoder, PositionalEncoder, StupidPositionalEncoder, TwoStagesDCDecoder, \
    TwoStagesDCEncoder, UNetTransitionNet, DDPMUNetTransitionNet
import math
import NF

class VAEModel(nn.Module):
    def __init__(self, **kwargs):
        super(VAEModel, self).__init__()
        self.latent_s = sum(kwargs['latent_s'])
        self.CNN = kwargs['CNN']
        self.device = 'cpu'
        self.img_size = kwargs['img_size']

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']

        if self.CNN:
            self.enc = DCEncoder(self.img_size, self.latent_s, enc_net, t_dim=0)
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
        z = torch.randn(nb_samples, self.latent_s).to(self.device)
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
        self.exact_ddpm_loss = kwargs['exact_ddpm_loss']
        self.KL_prior_diffusion = kwargs['KL_prior_diffusion']


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
            if self.exact_ddpm_loss:
                KL_rev_diffusion = self.T*(((mu_zt_1 - zt_1_rec) ** 2)/(2*sigma_zt_1**2)).sum(1)
            else:
                KL_rev_diffusion = self.T*(((mu_zt_1 - zt_1_rec) ** 2)).sum(1)
        else:
            KL_rev_diffusion = 0.

        if self.KL_prior_diffusion:
            KL_prior_diffusion = (-torch.log(sigma_T) + (mu_T ** 2)/2 + .5*sigma_T**2).sum(1)
        else:
            KL_prior_diffusion = 0.

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


class NFPriorVAEModel(nn.Module):
    def __init__(self, **kwargs):
        super(NFPriorVAEModel, self).__init__()
        self.latent_s = kwargs['latent_s']
        self.CNN = kwargs['CNN']
        self.device = 'cpu'
        self.img_size = kwargs['img_size']

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']

        conditioner_type = NF.AutoregressiveConditioner
        conditioner_args = {"in_size": sum(kwargs['latent_s']), "hidden": [kwargs['cond_w']] * kwargs['cond_l'],
                            "out_size": 2}

        normalizer_type = NF.AffineNormalizer
        normalizer_args = {}

        nb_flow = kwargs['n_nf_steps']
        self.flow = NF.buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args)

        if self.CNN:
            self.enc = DCEncoder(self.img_size, sum(self.latent_s), enc_net, t_dim=0)
            self.dec = DCDecoder(self.enc.features_dim, self.latent_s, dec_net, t_dim=0,
                                          out_c=self.img_size[0], img_width=self.img_size[1])
        else:
            self.enc = TemporalEncoder(32**2, sum(self.latent_s), enc_net, 0)
            self.dec = TemporalDecoder(32**2, self.latent_s, dec_net, 0)


    def loss(self, x0):
        bs = x0.shape[0]
        dev = x0.device
        # Encoding
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), sum(self.latent_s), 1)
        z0 = mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=dev)

        entropy_posterior_z = log_sigma_z.sum(1) + math.log(2*(math.pi*math.e)**.5)

        KL_prior_nf, _ = self.flow.compute_ll(z0)
        KL_prior_nf = -KL_prior_nf

        KL_pst_z_prior_z = entropy_posterior_z - KL_prior_nf

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
        self.device = device
        return self

    def sample(self, nb_samples=1):

        z = torch.randn(nb_samples, sum(self.latent_s)).to(self.device)
        z0 = self.flow.invert(z)
        x = self.dec(z0).view(nb_samples, -1)

        return x



class TwoStagesDDPMPriorVAEModel(nn.Module):
    def __init__(self, **kwargs):
        super(TwoStagesDDPMPriorVAEModel, self).__init__()
        self.device = 'cpu'
        self.img_size = kwargs['img_size']

        self.t_emb_s = kwargs['t_emb_s']
        if self.t_emb_s > 1:
            self.t_emb_s = (self.t_emb_s // 2) * 2
            pos_enc = PositionalEncoder(self.t_emb_s // 2)
        else:
            pos_enc = StupidPositionalEncoder(self.T_MAX)

        z_2_dim = [4, 16, 16] if self.img_size[1] == 32 else [24, 16, 16]

        self.enc = TwoStagesDCEncoder(self.img_size)
        self.dec = TwoStagesDCDecoder(out_c=self.img_size[0], img_width=self.img_size[1], z_2_dim=z_2_dim)

        self.z1_dim = 16*6
        self.z_2_dim = z_2_dim

        self.diffuser_1 = AsynchronousDiffuser(betas_min=kwargs['beta_min'], betas_max=kwargs['beta_max'],
                                             ts_min=kwargs['t_min'], ts_max=kwargs['t_max'],
                                               var_sizes=[self.z1_dim])

        self.diffuser_2 = AsynchronousDiffuser(betas_min=kwargs['beta_min'], betas_max=kwargs['beta_max'],
                                               ts_min=kwargs['t_min'], ts_max=kwargs['t_max'],
                                               var_sizes=[z_2_dim[0]*z_2_dim[1]*z_2_dim[2]])

        trans_net = [[kwargs['trans_w']] * kwargs['trans_l']] * kwargs['n_res_blocks']
        self.trans_net_1 = ImprovedTransitionNet(self.z1_dim, trans_net, self.t_emb_s, self.diffuser_1, pos_enc=pos_enc,
                                                 simplified_trans=kwargs['simplified_trans'])

        if True:
            self.trans_net_2 = UNetTransitionNet(z_dim=z_2_dim, t_dim=self.t_emb_s, diffuser=self.diffuser_2,
                                                 cond_in=self.z1_dim,
                                                 pos_enc=pos_enc, simplified_trans=kwargs['simplified_trans'])
        else:
            self.trans_net_2 = DDPMUNetTransitionNet(z_dim=z_2_dim, diffuser=self.diffuser_2,
                                                     simplified_trans=kwargs['simplified_trans'])
        self.T = max(kwargs['t_max'])

        nz = self.z1_dim
        ngf = 64
        act = nn.SELU
        self.t_conv1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            #nn.BatchNorm2d(ngf * 8),
            act(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ngf * 4),
            act(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True))

        self.conv1 = nn.Sequential(nn.Conv2d(ngf * 2 + 12, 16, 3, 1, 1), #nn.BatchNorm2d(16), act(),
                                   nn.Conv2d(16, z_2_dim[0] * 2, 3, 1, 1))


    def loss(self, x0):
        bs = x0.shape[0]
        dev = x0.device
        # Encoding
        h1, h2 = self.enc(x0.view(-1, *self.img_size))
        mu_z_1, log_sigma_z_1 = torch.split(h1, self.z1_dim, 1)
        z0_1 = mu_z_1 + torch.exp(log_sigma_z_1) * torch.randn(mu_z_1.shape, device=dev)

        #print(h2.shape, self.t_conv1(z0_1).shape, self.conv1(torch.cat((self.t_conv1(z0_1), h2), 1)).shape)
        mu_z_2, log_sigma_z_2 = torch.split(self.conv1(torch.cat((self.t_conv1(z0_1), h2), 1)), self.z_2_dim[0], 1)
        z0_2 = mu_z_2 + torch.exp(log_sigma_z_2) * torch.randn(mu_z_2.shape, device=dev)

        z1_shape, z2_shape = mu_z_1.shape, mu_z_2.shape

        entropy_posterior_z = log_sigma_z_1.view(bs, -1).sum(1) + log_sigma_z_2.view(bs, -1).sum(1) + math.log(2*(math.pi*math.e)**.5)
        t = torch.randint(1, self.T, (bs, 1), device=dev)

        # HERE we trick to compute p(z_T|x) directly in order to take into account the known randomness of z_0
        _, (mu_T_1, sigma_T_1) = self.diffuser_1.diffuse(mu_z_1.view(bs, -1), t * 0 + self.T, t*0)
        sigma_T_1 = (sigma_T_1 ** 2 - (sigma_T_1 ** 2 - 1) * torch.exp(2 * log_sigma_z_1.view(bs, -1))).sqrt()
        _, (mu_T_2, sigma_T_2) = self.diffuser_2.diffuse(mu_z_2.view(bs, -1), t * 0 + self.T, t*0)
        sigma_T_2 = (sigma_T_2 ** 2 - (sigma_T_2 ** 2 - 1) * torch.exp(2 * log_sigma_z_2.view(bs, -1))).sqrt()

        zt_1, _ = self.diffuser_1.diffuse(z0_1.view(bs, -1), t)
        zt_2, _ = self.diffuser_2.diffuse(z0_2.view(bs, -1), t)

        if self.T >= 1:
            mu_zt_1_1, sigma_zt_1_1 = self.diffuser_1.prev_mean_var(zt_1, z0_1.view(bs, -1), t)
            mu_zt_1_2, sigma_zt_1_2 = self.diffuser_2.prev_mean_var(zt_2, z0_2.view(bs, -1), t)
            zt_2_rec = self.trans_net_2(zt_2.view(*z2_shape), t - 1, z0_1.view(bs, self.z1_dim, 1, 1)).view(bs, -1)
            zt_1_rec = self.trans_net_1(zt_1, t - 1)

            KL_rev_diffusion = (((mu_zt_1_1 - zt_1_rec) ** 2)).sum(1) + (((mu_zt_1_2 - zt_2_rec) ** 2)).sum(1)
        else:
            KL_rev_diffusion = 0.

        KL_prior_diffusion = (-torch.log(sigma_T_1) + (mu_T_1 ** 2)/2 + .5*sigma_T_1**2).sum(1) + \
                             (-torch.log(sigma_T_2) + (mu_T_2 ** 2) / 2 + .5 * sigma_T_2 ** 2).sum(1)

        KL_pst_z_prior_z = entropy_posterior_z - (KL_prior_diffusion + KL_rev_diffusion)

        # Decoding
        mu_x_pred = self.dec(z0_1.view(*z1_shape), z0_2.view(*z2_shape))
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1)

        loss = KL_x.mean(0) - KL_pst_z_prior_z.mean(0)

        return loss

    def forward(self, x0):
        h1, h2 = self.enc(x0.view(-1, *self.img_size))
        mu_z_1, log_sigma_z_1 = torch.split(h1, self.z1_dim, 1)
        z0_1 = mu_z_1 + torch.exp(log_sigma_z_1) * torch.randn(mu_z_1.shape, device=self.device)

        mu_z_2, log_sigma_z_2 = torch.split(self.conv1(torch.cat((self.t_conv1(z0_1), h2), 1)), self.z_2_dim[0], 1)
        z0_2 = mu_z_2 + torch.exp(log_sigma_z_2) * torch.randn(mu_z_2.shape, device=self.device)

        mu_x_pred = self.dec(z0_1, 0 * z0_2)
        return mu_x_pred

    def to(self, device):
        super().to(device)
        self.trans_net_2.to(device)
        self.trans_net_1.to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1):
        z0_1 = self.trans_net_1.sample(nb_samples).view(nb_samples, self.z1_dim, 1, 1)
        z0_2 = 0 * self.trans_net_2.sample(z0_1, nb_samples)

        mu_x_pred = self.dec(z0_1, z0_2).view(nb_samples, -1)

        return mu_x_pred


