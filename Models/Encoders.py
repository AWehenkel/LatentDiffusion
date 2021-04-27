import torch.nn as nn
import torch

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


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        super(Encoder, self).__init__()
        # decoder part
        self.net = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, z_dim))

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


class TemporalEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, act=nn.SELU):
        super(TemporalEncoder, self).__init__()
        layers = [x_dim + t_dim] + layers + [z_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)

    def forward(self, x, t=None):
        if t is not None:
            x = torch.cat((x.view(x.shape[0], -1), t), 1)
        return self.net(x.view(x.shape[0], -1))


class SimpleImageEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, pos_enc=None, act=nn.SELU):
        super(SimpleImageEncoder, self).__init__()
        image_channels = x_dim[0]
        init_channels = 25
        kernel_size = 4
        self.conv = nn.Sequential(nn.Conv2d(image_channels, init_channels, kernel_size, padding=1, stride=2), act(), #nn.MaxPool2d(2, 2),
                                  nn.Conv2d(init_channels, init_channels*2, kernel_size, padding=1, stride=2), act(),
                                  nn.Conv2d(init_channels*2, init_channels*4, kernel_size, padding=1, stride=2), act(),
                                  nn.Conv2d(init_channels*4, init_channels*4, kernel_size, padding=0, stride=2), act())
                                  #nn.MaxPool2d(2, 2))
        self.features_dim = self.conv(torch.zeros(x_dim).unsqueeze(0)).shape[1:]
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalEncoder(features_dim, z_dim, layers, t_dim, act)
        self.pos_enc = pos_enc

    def forward(self, x, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        features = self.conv(x).view(x.shape[0], -1)
        return self.fc(features, t)


class DCEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, pos_enc=None, act=nn.SELU):
        super(DCEncoder, self).__init__()
        image_channels = x_dim[0]
        init_channels = int(z_dim/4)
        print(z_dim)
        kernel_size = 4
        if x_dim[1] == 32:
            self.conv = nn.Sequential(nn.Conv2d(image_channels, init_channels, kernel_size, padding=1, stride=2), act(),
                                      # nn.MaxPool2d(2, 2),
                                      nn.Conv2d(init_channels, init_channels * 2, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 2),
                                      nn.Conv2d(init_channels * 2, init_channels * 4, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 4),
                                      nn.Conv2d(init_channels * 4, init_channels * 8, kernel_size, padding=0, stride=2))
            # nn.MaxPool2d(2, 2))
        elif x_dim[1] == 64:
            self.conv = nn.Sequential(nn.Conv2d(image_channels, init_channels, kernel_size, padding=1, stride=2), act(),
                                      # nn.MaxPool2d(2, 2),
                                      nn.Conv2d(init_channels, init_channels * 2, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 2),
                                      nn.Conv2d(init_channels * 2, init_channels * 4, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 4),
                                      nn.Conv2d(init_channels * 4, init_channels * 4, kernel_size, padding=0, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 4),
                                      nn.Conv2d(init_channels * 4, init_channels * 8, kernel_size, padding=1, stride=2))
        elif x_dim[1] == 256:
            self.conv = nn.Sequential(nn.Conv2d(image_channels, init_channels, kernel_size, padding=1, stride=2), act(),
                                      # nn.MaxPool2d(2, 2),
                                      nn.BatchNorm2d(init_channels),
                                      nn.Conv2d(init_channels, init_channels * 2, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 2),
                                      nn.Conv2d(init_channels * 2, init_channels * 4, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 4),
                                      nn.Conv2d(init_channels * 4, init_channels * 8, kernel_size, padding=0, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 8),
                                      nn.Conv2d(init_channels * 8, init_channels * 16, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 16),
                                      nn.Conv2d(init_channels * 16, init_channels * 8, kernel_size, padding=1, stride=2),
                                      act(),
                                      nn.BatchNorm2d(init_channels * 8),
                                      nn.Conv2d(init_channels * 8, init_channels * 8, kernel_size, padding=1, stride=2))
        self.features_dim = self.conv(torch.zeros(x_dim).unsqueeze(0)).shape[1:]
        #features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        #self.fc = TemporalEncoder(features_dim, z_dim, layers, t_dim, act)
        #self.pos_enc = pos_enc

    def forward(self, x, t=None):
        #t = self.pos_enc(t) if self.pos_enc is not None else t
        features = self.conv(x).view(x.shape[0], -1)
        return features

