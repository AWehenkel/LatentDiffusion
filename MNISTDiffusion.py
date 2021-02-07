import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DataDiffuser, TransitionNet, SimpleImageDecoder, SimpleImageEncoder
import torch.optim as optim
from torchvision.utils import save_image
import torchvision
from torchvision import datasets, transforms
import numpy as np


def getMNISTDataLoader(bs):
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        AddUniformNoise(),
        ToTensor()
        # transforms.ToTensor()
    ]))
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      AddUniformNoise(),
                                      ToTensor()
                                      # transforms.ToTensor()
                                  ]))

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    return train_loader, test_loader


def logit(x, alpha=1E-6):
    y = alpha + (1.-2*alpha)*x
    return np.log(y) - np.log(1. - y)


def logit_back(x, alpha=1E-6):
    y = torch.sigmoid(x)
    return (y - alpha)/(1.-2*alpha)


class AddUniformNoise(object):
    def __init__(self, alpha=1E-6):
        self.alpha = alpha
    def __call__(self,samples):
        samples = np.array(samples,dtype = np.float32)
        samples += np.random.uniform(size = samples.shape)
        samples = logit(samples/256., self.alpha)
        return samples


class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self,samples):
        samples = torch.from_numpy(np.array(samples,dtype = np.float32)).float()
        return samples


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


class CNNDiffusionModel(nn.Module):
    def __init__(self, cnn=False):
        super(CNNDiffusionModel, self).__init__()
        self.T_MAX = 20
        self.latent_s = 20
        self.device = 'cpu'
        t_emb_s = 20
        self.pos_enc = PositionalEncoder(t_emb_s // 2)  # StupidPositionalEncoder(T_MAX)  #
        # dec = TemporalDecoder(784, latent_s, [256]*2, t_emb_s).to(dev)
        # enc = TemporalEncoder(784, latent_s, [256]*4, t_emb_s).to(dev)
        if cnn:
            self.enc = SimpleImageEncoder([1, 32, 32], self.latent_s, [200] * 4, t_dim=t_emb_s).to(dev)
            self.dec = SimpleImageDecoder(self.enc.features_dim, self.latent_s, [200] * 3, t_dim=t_emb_s).to(dev)
        else:
            self.dec = TemporalDecoder(784, self.latent_s, [256] * 2, t_emb_s).to(dev)
            self.enc = TemporalEncoder(784, self.latent_s, [256] * 4, t_emb_s).to(dev)

        self.trans = TransitionNet(self.latent_s, [100] * 3, t_emb_s).to(dev)
        self.dif = DataDiffuser(beta_min=1e-2, beta_max=1., t_max=self.T_MAX).to(dev)
        self.sampling_t0 = False

    def loss(self, x0):
        if self.sampling_t0:
            t0 = torch.randint(0, self.T_MAX - 1, [x0.shape[0]]).to(dev)
            x_t0, sigma_x_t0 = self.dif.diffuse(x0, t0, torch.zeros(x0.shape[0]).long().to(dev))
        else:
            t0 = torch.zeros(x0.shape[0]).to(dev).long()
            x_t0 = x0

        z_t0 = self.enc(x_t0.view(-1, 1, 32, 32), self.pos_enc(t0.float().unsqueeze(1)))
        # z_t0 = z_t0 + torch.randn(z_t0.shape).to(dev) * (1 - dif.alphas[t0]).sqrt().unsqueeze(1).expand(-1, z_t0.shape[1])
        t = torch.torch.distributions.Uniform(t0.float() + 1, torch.ones_like(t0) * self.T_MAX).sample().long().to(dev)

        z_t, sigma_z = self.dif.diffuse(z_t0, t, t0)
        x_t, sigma_x = self.dif.diffuse(x_t0, t, t0)

        mu_x_pred = self.dec(z_t, self.pos_enc(t.float().unsqueeze(1)))
        KL_x = ((mu_x_pred - x_t.view(bs, 1, 32, 32)) ** 2).view(bs, -1).sum(1) / sigma_x ** 2

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
            z_t = self.trans(z_t, self.pos_enc(t_t)) + torch.randn(z_t.shape).to(dev) * sigma

        x_0 = self.dec(z_t, self.pos_enc(torch.zeros(nb_samples, 1))).view(-1, 784)

        return x_0

if __name__ == "__main__":
    bs = 100

    train_loader, test_loader = getMNISTDataLoader(bs)

    # Compute Mean abd std per pixel
    x_mean = 0
    x_mean2 = 0
    for batch_idx, (cur_x, target) in enumerate(train_loader):
        cur_x = cur_x.view(bs, -1).float()
        x_mean += cur_x.mean(0)
        x_mean2 += (cur_x ** 2).mean(0)
    x_mean /= batch_idx + 1
    x_std = (x_mean2 / (batch_idx + 1) - x_mean ** 2) ** .5
    x_std[x_std == 0.] = 1.

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CNNDiffusionModel(True).to(dev)#CNNDiffusionModel().to(dev)

    optimizer = optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)


    def get_X_back(x):
        nb_x = x.shape[0]
        x = x * x_std.to(dev).unsqueeze(0).expand(nb_x, -1) + x_mean.to(dev).unsqueeze(0).expand(nb_x, -1)
        return logit_back(x)


    def train(epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            x0 = data.view(data.shape[0], -1).to(dev)

            x0 = (x0 - x_mean.to(dev).unsqueeze(0).expand(bs, -1)) / x_std.to(dev).unsqueeze(0).expand(bs, -1)
            optimizer.zero_grad()

            loss = model.loss(x0)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(data)))

        save_image(get_X_back(model.sample(64)).view(64, 1, 32, 32), './Samples/Generated/sample_gen_' + str(epoch)
                   + '.png')
        scheduler.step(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


    for i in range(500):
        train(i)
