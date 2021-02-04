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
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, act=nn.ReLU):
        super(TemporalEncoder, self).__init__()
        layers = [x_dim + t_dim] + layers + [z_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)

    def forward(self, x, t):
        return self.net(torch.cat((x, t), 1))


class SimpleImageEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, act=nn.ReLU):
        super(SimpleImageEncoder, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(x_dim[0], 16, 3, padding=1), act(), nn.MaxPool2d(2, 2),
                                  nn.Conv2d(16, 4, 3, padding=1), act(), nn.MaxPool2d(2, 2))
        self.features_dim = self.conv(torch.zeros(x_dim).unsqueeze(0)).shape[1:]
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalEncoder(features_dim, z_dim, layers, t_dim, act)

    def forward(self, x, t):
        features = self.conv(x).view(x.shape[0], -1)
        return self.fc(features, t)
