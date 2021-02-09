import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DataDiffuser, TransitionNet, SimpleImageDecoder, SimpleImageEncoder, PositionalEncoder, StupidPositionalEncoder


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
            self.enc = SimpleImageEncoder(self.img_size, self.latent_s*2, enc_net, t_dim=0)
            self.dec = SimpleImageDecoder(self.enc.features_dim, self.latent_s, dec_net, t_dim=0,
                                          out_c=self.img_size[0])
        else:
            self.enc = TemporalEncoder(32**2, self.latent_s*2, enc_net, 0)
            self.dec = TemporalDecoder(32**2, self.latent_s, dec_net, 0)

    def loss(self, x0):
        bs = x0.shape[0]

        # Encoding
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), self.latent_s, 1)
        KL_z = (-log_sigma_z + (mu_z ** 2)/2 + torch.exp(log_sigma_z)/2).sum(1)

        # Decoding
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape))
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1)



        loss = KL_x.mean(0) + KL_z.mean(0)

        return loss

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1):
        z = torch.randn(64, self.latent_s).to(self.device)
        x = self.dec(z).view(nb_samples, -1)

        return x

