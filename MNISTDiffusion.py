import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DataDiffuser, TransitionNet, SimpleImageDecoder, SimpleImageEncoder, PositionalEncoder, StupidPositionalEncoder
import torch.optim as optim
from torchvision.utils import save_image
import torchvision
from torchvision import datasets, transforms
import numpy as np


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x

def getMNISTDataLoader(bs):
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      #transforms.ToTensor(),
                                      #add_noise,
                                      ToTensor(),
        AddUniformNoise()
                                  ]))
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      #transforms.ToTensor(),
                                      #add_noise,
                                      ToTensor(),
                                      AddUniformNoise()
                                  ]))

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    return train_loader, test_loader

def getCIFAR10DataLoader(bs):
    # MNIST Dataset
    train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, download=True, transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      add_noise,
                                      # transforms.ToTensor()
                                  ]))
    test_dataset = datasets.CIFAR10(root='./cifar10_data/', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      add_noise,
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
        self.img_size = [1, 32, 32]
        self.pos_enc = PositionalEncoder(self.t_emb_s // 2)  # StupidPositionalEncoder(T_MAX)  #

        if self.CNN:
            self.enc = SimpleImageEncoder(self.img_size, self.latent_s, kwargs['enc_net'], t_dim=self.t_emb_s).to(dev)
            self.dec = SimpleImageDecoder(self.enc.features_dim, self.latent_s, kwargs['dec_net'], t_dim=self.t_emb_s,
                                          out_c=self.img_size[0]).to(dev)
        else:
            self.dec = TemporalDecoder(32**2, self.latent_s, kwargs['dec_net'], self.t_emb_s).to(dev)
            self.enc = TemporalEncoder(32**2, self.latent_s, kwargs['enc_net'], self.t_emb_s).to(dev)

        self.trans = TransitionNet(self.latent_s, kwargs['trans_net'], self.t_emb_s).to(dev)
        self.dif = DataDiffuser(beta_min=self.beta_min, beta_max=self.beta_max, t_max=self.T_MAX).to(dev)
        self.sampling_t0 = False

    def loss(self, x0):
        if self.sampling_t0:
            t0 = torch.randint(0, self.T_MAX - 1, [x0.shape[0]]).to(dev)
            x_t0, sigma_x_t0 = self.dif.diffuse(x0, t0, torch.zeros(x0.shape[0]).long().to(dev))
        else:
            t0 = torch.zeros(x0.shape[0]).to(dev).long()
            x_t0 = x0

        z_t0 = self.enc(x_t0.view(-1, *self.img_size), self.pos_enc(t0.float().unsqueeze(1)))
        # z_t0 = z_t0 + torch.randn(z_t0.shape).to(dev) * (1 - dif.alphas[t0]).sqrt().unsqueeze(1).expand(-1, z_t0.shape[1])
        t = torch.torch.distributions.Uniform(t0.float() + 1, torch.ones_like(t0) * self.T_MAX).sample().long().to(dev)

        z_t, sigma_z = self.dif.diffuse(z_t0, t, t0)
        x_t, sigma_x = self.dif.diffuse(x_t0, t, t0)

        mu_x_pred = self.dec(z_t, self.pos_enc(t.float().unsqueeze(1)))
        KL_x = ((mu_x_pred - x_t.view(bs, *self.img_size)) ** 2).view(bs, -1).sum(1) / sigma_x ** 2

        alpha_bar_t = self.dif.alphas[t].unsqueeze(1)#.expand(-1, self.latent_s)
        alpha_t = self.dif.alphas_t[t].unsqueeze(1)#.expand(-1, self.latent_s)
        beta_t = self.dif.betas[t].unsqueeze(1)#.expand(-1, self.latent_s)

        mu_z_pred = (z_t - beta_t/(1-alpha_bar_t).sqrt() * self.trans(z_t, self.pos_enc(t.float().unsqueeze(1))))/alpha_t.sqrt()
        #mu_z_pred = self.trans(z_t, self.pos_enc(t.float().unsqueeze(1)))
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
            alpha_bar_t = self.dif.alphas[t]
            alpha_t = self.dif.alphas_t[t]
            beta_t = self.dif.betas[t]
            mu_z_pred = (z_t - beta_t / (1 - alpha_bar_t).sqrt() * self.trans(z_t, self.pos_enc(t_t))) / alpha_t.sqrt()
            #mu_z_pred = self.trans(z_t, self.pos_enc(t_t))
            z_t = mu_z_pred + torch.randn(z_t.shape, device=self.device) * sigma

        x_0 = self.dec(z_t, self.pos_enc(torch.zeros((nb_samples, 1), device=self.device))).view(nb_samples, -1)

        return x_0


import wandb
wandb.init(project="latent_diffusion", entity="awehenkel")


if __name__ == "__main__":
    bs = 100
    config = {
        'data': 'MNIST',
        'T_MAX': 30,
        'latent_s': 30,
        't_emb_s': 20,
        'CNN': False,
        'enc_net': [200] * 3,
        'dec_net': [200] * 3,
        'trans_net': [150] * 3,
        "beta_min": 1e-2,
        "beta_max": .5
    }
    wandb.config.update(config)
    train_loader, test_loader = getMNISTDataLoader(bs)
    img_size = [1, 32, 32]

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
    model = CNNDiffusionModel(**config).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    wandb.watch(model)
    def get_X_back(x):
        nb_x = x.shape[0]
        x = x * x_std.to(dev).unsqueeze(0).expand(nb_x, -1) + x_mean.to(dev).unsqueeze(0).expand(nb_x, -1)
        return logit_back(x)


    #sample = list(train_loader)[0][0][[0]].expand(bs, -1, -1, -1)
    #save_image(get_X_back(sample.to(dev)[[0]].reshape(1, -1)).reshape(1, 3, 32, 32), './Samples/Generated/sample_rel_' + '.png')
    def train(epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            #data = sample
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
        samples = get_X_back(model.sample(64)).view(64, *img_size)
        save_image(samples, './Samples/Generated/sample_gen_' + str(epoch) + '.png')
        scheduler.step(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        wandb.log({"Train Loss": train_loss / len(train_loader.dataset), "Samples": [wandb.Image(samples)]})

    for i in range(500):
        train(i)
