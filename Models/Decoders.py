import torch
import torch.nn as nn


class TemporalDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, act=nn.ReLU, out_dim=[1, 32, 32]):
        super(TemporalDecoder, self).__init__()
        # decoder part
        layers = [z_dim + t_dim] + layers + [x_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)
        self.out_dim = out_dim

    def forward(self, z, t=None):
        if t is not None:
            z = torch.cat((z, t), 1)
        return self.net(z).view(z.shape[0], *self.out_dim)


class SimpleImageDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=1, pos_enc=None, act=nn.ReLU, out_c=1):
        super(SimpleImageDecoder, self).__init__()
        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalDecoder(features_dim, z_dim, layers, t_dim, act, self.features_dim)
        init_channels = 8
        kernel_size = 4
        self.conv = nn.Sequential(nn.ConvTranspose2d(self.features_dim[0], init_channels * 8, kernel_size, stride=1, padding=0),
                                  act(),
                                  nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 2, out_c, kernel_size, stride=2,
                                                     padding=1))
        self.pos_enc = pos_enc

    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        features = self.fc(z, t)
        return self.conv(features)


class ImprovedImageDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=1, pos_enc=None, act=nn.ReLU, out_c=1):
        super(ImprovedImageDecoder, self).__init__()
        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalDecoder(features_dim, z_dim, layers, t_dim, act, self.features_dim)
        init_channels = 8
        kernel_size = 4
        self.conv = nn.Sequential(nn.ConvTranspose2d(self.features_dim[0], init_channels * 8, kernel_size, stride=1, padding=0),
                                  act(),
                                  nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 2, 10, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(10, 10, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.Conv2d(10, 5, 2, 1), act(),
                                  nn.Conv2d(5, out_c, 2, 2, 1)
                                  )
        self.pos_enc = pos_enc


    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        features = self.fc(z, t)
        return self.conv(features)
