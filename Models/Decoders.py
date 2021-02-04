import torch
import torch.nn as nn


class TemporalDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, act=nn.ReLU):
        super(TemporalDecoder, self).__init__()
        # decoder part
        layers = [z_dim + t_dim] + layers + [x_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)

    def forward(self, z, t):
        return self.net(torch.cat((z, t), 1))


class SimpleImageDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=1, act=nn.ReLU):
        super(SimpleImageDecoder, self).__init__()
        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalDecoder(features_dim, z_dim, layers, t_dim, act)
        self.conv = nn.Sequential(nn.ConvTranspose2d(4, 16, 2, stride=2), act(), nn.ConvTranspose2d(16, 1, 2, stride=2))

    def forward(self, z, t):
        features = self.fc(z, t).view(z.shape[0], *self.features_dim)
        return self.conv(features)
