import torch
import torch.nn as nn
from Models import TemporalDecoder, DCDecoder, DCEncoder, TemporalEncoder, AsynchronousDiffuser, TransitionNet, \
    SimpleImageDecoder, SimpleImageEncoder, PositionalEncoder, StupidPositionalEncoder, ProgressiveDecoder, \
    ProgressiveDecoder2, ImprovedTransitionNet

decoder_types = {'Progressive': ProgressiveDecoder, 'DC': DCDecoder, 'Progressive2': ProgressiveDecoder2}
# VAE that takes temporal information and play with it.
class AugmentedVAEModel(nn.Module):
    def __init__(self, **kwargs):
        super(AugmentedVAEModel, self).__init__()
        self.latent_s = kwargs['latent_s']
        self.CNN = kwargs['CNN']
        self.device = 'cpu'
        self.img_size = kwargs['img_size']
        self.t_emb_s = kwargs['t_emb_s']

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']

        if self.t_emb_s > 1:
            self.t_emb_s = (self.t_emb_s // 2) * 2
            self.pos_enc = PositionalEncoder(self.t_emb_s // 2)
        else:
            self.pos_enc = StupidPositionalEncoder(self.T_MAX)

        if self.CNN:
            self.enc = SimpleImageEncoder(self.img_size, self.latent_s*2, enc_net, t_dim=self.t_emb_s)
            self.dec = SimpleImageDecoder(self.enc.features_dim, self.latent_s, dec_net, t_dim=self.t_emb_s,
                                          out_c=self.img_size[0])
        else:
            self.enc = TemporalEncoder(32**2, self.latent_s*2, enc_net, self.t_emb_s)
            self.dec = TemporalDecoder(32**2, self.latent_s, dec_net, self.t_emb_s)

    def forward(self, x0, t):
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size), self.pos_enc(t)), self.latent_s, 1)
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device),
                             self.pos_enc(t))
        return mu_x_pred

    def loss(self, x0, t):
        bs = x0.shape[0]

        # Encoding
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size), self.pos_enc(t)), self.latent_s, 1)
        KL_z = (-log_sigma_z/2 + (mu_z ** 2)/2 + torch.exp(log_sigma_z)/2).sum(1)

        # Decoding
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device), self.pos_enc(t))
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1)



        loss = KL_x.mean(0) + KL_z.mean(0)

        return loss

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1, temperature=1.):
        z = torch.randn(nb_samples, self.latent_s).to(self.device) * temperature
        t = torch.zeros((nb_samples, 1), device=self.device)
        x = self.dec(z, self.pos_enc(t)).view(nb_samples, -1)

        return x



# VAEs which takes temporal information and use it to

class HeatedLatentDiffusionModel(nn.Module):
    def __init__(self, encoder=None, decoder=None, diffuser=None, latent_transition=None):
        super(HeatedLatentDiffusionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffuser = diffuser
        self.latent_transition = latent_transition

    def loss(self, x0, xt, xt_1, t):
        bs = x0.shape[0]

        z0 = self.encoder(x0.view(-1, *self.img_size), t * 0)
        #x0_rec = self.decoder(z0, t).view(bs, -1)
        #KL_x0 = ((x0 - x0_rec)**2).mean(1)

        # TODO, should I detach z0?
        zt = self.diffuser.diffuse(z0, t)

        #zt_1_rec_enc = self.encoder(xt_1.view(-1, *self.img_size), t-1)

        zt_rec = self.encoder(xt.view(-1, *self.img_size), t)

        xt_rec = self.decoder(zt, t).view(bs, -1)
        KL_xt = ((xt - xt_rec)**2).view(bs, -1).mean(1)

        zt_1 = self.diffuser.reverse(z0.detach(), zt, t - 1)
        zt_1_rec = self.latent_transition(zt, t - 1)

        sigma_cond = ((1 - self.diffuser.alphas[t.view(-1) - 1, :]) / (
                    1 - self.diffuser.alphas[t.view(-1), :]) * self.diffuser.betas[t.view(-1),
                                                               :]).sqrt()

        dirac_z0 = self.diffuser.dirac_z0[t.view(-1), :]
        dirac_zt_1 = self.diffuser.dirac_zt_1[t.view(-1), :]
        no_dirac = self.diffuser.no_dirac[t.view(-1), :]

        sigma_cond[sigma_cond / sigma_cond != sigma_cond / sigma_cond] = 0.
        sigma_cond = 1. * dirac_zt_1 + dirac_z0 * 0. + no_dirac * sigma_cond
        zt_1_rec = zt_1_rec * (1 - dirac_zt_1)

        xt_1_rec = self.decoder(zt_1_rec + sigma_cond * torch.randn_like(zt_1), t - 1).view(bs, -1)
        sigma_cond[sigma_cond / sigma_cond != sigma_cond / sigma_cond] = 1.

        KL_xt_1 = ((xt_1 - xt_1_rec)**2).view(bs, -1).mean(1)

        if self.simple_transition:
            alpha_bar_t = self.diffuser.alphas[t.view(-1), :]
            alpha_t = self.diffuser.alphas_t[t.view(-1), :]
            beta_t = self.diffuser.betas[t.view(-1), :]

            is_dirac = torch.logical_or((1 - alpha_bar_t) == 0., beta_t == 0.)
            beta_t[is_dirac] = 1.


            epsilon = is_dirac.float() * torch.zeros_like(zt) + (1 - is_dirac.float()) * ((zt - alpha_t.sqrt() * zt_1) * (1-alpha_bar_t.sqrt())/beta_t)
            KL_diffusion = (((epsilon - zt_1_rec)/sigma_cond)** 2).mean(1)
        else:
            KL_diffusion = (((zt_1 - zt_1_rec) / sigma_cond) ** 2).mean(1)

        # TODO check effect of sigma_cond
        KL_zt = (((zt - zt_rec)/sigma_cond)**2).mean(1)

        #KL_zt_1 = (((zt_1_rec_enc - zt_1_rec.detach())/sigma_cond)**2).mean(1)

        #loss = (KL_x0 + KL_zt + KL_xt + KL_diffusion).mean()
        loss = (KL_zt + KL_xt + KL_diffusion + KL_xt_1).mean()
        return loss

    def to(self, device):
        super().to(device)
        self.device = device
        self.latent_transition.to(device)
        return self

    def sample(self, nb_samples=1, t0=0, temperature=1.):
        z_0 = self.latent_transition.sample(nb_samples, t0, temperature=temperature)
        x_0 = self.decoder(z_0, torch.ones(nb_samples, 1).to(self.device) * t0).view(nb_samples, -1)
        return x_0

    def forward(self, x, t):
        z0 = self.encoder(x.view(-1, *self.img_size), t * 0)
        x0_rec = self.decoder(z0, t).view(x.shape[0], -1)
        return x0_rec


class CNNHeatedLatentDiffusion(HeatedLatentDiffusionModel):
    def __init__(self, **kwargs):
        super(CNNHeatedLatentDiffusion, self).__init__()

        self.latent_s = sum(kwargs['var_sizes'])#kwargs['latent_s']
        self.CNN = kwargs['CNN']
        self.device = 'cpu'
        self.img_size = kwargs['img_size']
        self.t_emb_s = kwargs['t_emb_s']
        self.register_buffer("beta_min", torch.tensor(kwargs['beta_min']))
        self.register_buffer("beta_max", torch.tensor(kwargs['beta_max']))
        self.simple_transition = kwargs['simplified_trans']

        if self.t_emb_s > 1:
            self.t_emb_s = (self.t_emb_s // 2) * 2
            pos_enc = PositionalEncoder(self.t_emb_s // 2)
        else:
            pos_enc = StupidPositionalEncoder(self.T_MAX)

        enc_net = [kwargs['enc_w']] * kwargs['enc_l']
        dec_net = [kwargs['dec_w']] * kwargs['dec_l']
        trans_net = [[kwargs['trans_w']] * kwargs['trans_l']] * kwargs['n_res_blocks']

        self.encoder = DCEncoder(self.img_size, self.latent_s, enc_net, t_dim=self.t_emb_s, pos_enc=pos_enc)
        self.decoder = decoder_types[kwargs['decoder_type']](self.encoder.features_dim, kwargs['var_sizes'], dec_net, t_dim=self.t_emb_s,
                                          pos_enc=pos_enc, out_c=self.img_size[0], img_width=self.img_size[1])

        self.diffuser = AsynchronousDiffuser(betas_min=[self.beta_min]*len(kwargs['ts_min']),
                                             betas_max=[self.beta_max]*len(kwargs['ts_min']),
                                             ts_min=kwargs['ts_min'], ts_max=kwargs['ts_max'],
                                             var_sizes=kwargs['var_sizes'])
        self.latent_transition = ImprovedTransitionNet(self.latent_s, trans_net, self.t_emb_s, self.diffuser, pos_enc=pos_enc,
                                               simplified_trans=kwargs['simplified_trans'])

