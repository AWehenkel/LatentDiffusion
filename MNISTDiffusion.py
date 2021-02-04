import torch
import torch.nn as nn
from Models import TemporalDecoder, TemporalEncoder, DataDiffuser, TransitionNet, SimpleImageDecoder, SimpleImageEncoder
import torch.optim as optim
from torchvision.utils import save_image
import torchvision
from torchvision import datasets, transforms
import numpy as np

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


if __name__ == "__main__":
    bs = 100
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.Compose([
        AddUniformNoise(),
        ToTensor()
        # transforms.ToTensor()
    ]))
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True,
                                  transform=transforms.Compose([
                                      AddUniformNoise(),
                                      ToTensor()
                                      # transforms.ToTensor()
                                  ]))

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

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

    T_MAX = 25
    latent_s = 25
    t_emb_s = 1
    pos_enc = StupidPositionalEncoder(T_MAX)  # PositionalEncoder(t_emb_s//2)#
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #dec = TemporalDecoder(784, latent_s, [256]*2, t_emb_s).to(dev)
    #enc = TemporalEncoder(784, latent_s, [256]*4, t_emb_s).to(dev)
    enc = SimpleImageEncoder([1, 28, 28], latent_s, [100, 100, 100]).to(dev)
    dec = SimpleImageDecoder(enc.features_dim, latent_s, [30, 30]).to(dev)
    trans = TransitionNet(latent_s, [100]*2, t_emb_s).to(dev)
    dif = DataDiffuser(beta_min=1e-2, beta_max=1., t_max=T_MAX).to(dev)
    sampling_t0 = False

    optimizer = optim.Adam(list(dec.parameters()) + list(enc.parameters()) + list(trans.parameters()), lr=.001)
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

            if sampling_t0:
                t0 = torch.randint(0, T_MAX - 1, [x0.shape[0]]).to(dev)
                x_t0, sigma_x_t0 = dif.diffuse(x0, t0, torch.zeros(x0.shape[0]).long().to(dev))
            else:
                t0 = torch.zeros(x0.shape[0]).to(dev).long()
                x_t0 = x0

            z_t0 = enc(x_t0.view(-1, 1, 28, 28), pos_enc(t0.float().unsqueeze(1)))
            # z_t0 = z_t0 + torch.randn(z_t0.shape).to(dev) * (1 - dif.alphas[t0]).sqrt().unsqueeze(1).expand(-1, z_t0.shape[1])
            t = torch.torch.distributions.Uniform(t0.float() + 1, torch.ones_like(t0) * T_MAX).sample().long().to(dev)

            z_t, sigma_z = dif.diffuse(z_t0, t, t0)
            x_t, sigma_x = dif.diffuse(x_t0, t, t0)

            mu_x_pred = dec(z_t, pos_enc(t.float().unsqueeze(1)))
            KL_x = ((mu_x_pred - x_t.view(bs, 1, 28, 28)) ** 2).view(bs, -1).sum(1) / sigma_x ** 2

            mu_z_pred = trans(z_t, pos_enc(t.float().unsqueeze(1)))
            mu, sigma = dif.prev_mean(z_t0, z_t, t)
            KL_z = ((mu - mu_z_pred) ** 2).sum(1) / sigma ** 2

            loss = KL_x.mean(0) + KL_z.mean(0)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(data)))
        scheduler.step(train_loss)
        zT = torch.randn(64, latent_s).to(dev)
        z_t = zT
        for t in range(T_MAX - 1, 0, -1):
            t_t = torch.ones(64, 1).to(dev) * t
            if t > 0:
                sigma = ((1 - dif.alphas[t - 1]) / (1 - dif.alphas[t]) * dif.betas[t]).sqrt()
            else:
                sigma = 0
            z_t = trans(z_t, pos_enc(t_t)) + torch.randn(z_t.shape).to(dev) * sigma
            if (t - 1) % 100 == 0:
                x_t = dec(z_t, pos_enc(t_t - 1)).view(-1, 784)
                save_image(get_X_back(x_t).view(64, 1, 28, 28),
                           './Samples/Generated/sample_gen_' + str(epoch) + '_' + str(t - 1) + '.png')
                x_t, _ = dif.diffuse(x0, (torch.ones(x0.shape[0]).to(dev) * t - 1).long(),
                                     torch.zeros(x0.shape[0]).long().to(dev))
                save_image(get_X_back(x_t).view(x0.shape[0], 1, 28, 28),
                           './Samples/Real/sample_real_' + str(epoch) + '_' + str(t - 1) + '.png')

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


    for i in range(500):
        train(i)
