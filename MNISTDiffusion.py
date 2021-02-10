import torch
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import datasets, transforms
import numpy as np
from Models import LatentDiffusionModel


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


import wandb
wandb.init(project="latent_diffusion", entity="awehenkel")


if __name__ == "__main__":
    bs = 100
    config = {
        'data': 'MNIST',
        'T_MAX': 50,
        'latent_s': 60,
        't_emb_s': 30,
        'CNN': False,
        'enc_w': 200,
        'enc_l': 3,
        'dec_w': 200,
        'dec_l': 3,
        'trans_w': 200,
        'trans_l': 3,
        "beta_min": 1e-2,
        "beta_max": .9,
        'simplified_trans': True
    }
    wandb.config.update(config)
    train_loader, test_loader = getMNISTDataLoader(bs)
    img_size = [1, 32, 32]
    config["img_size"] = img_size
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
    model = LatentDiffusionModel(**config).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    wandb.watch(model)
    def get_X_back(x):
        nb_x = x.shape[0]
        x = x * x_std.to(dev).unsqueeze(0).expand(nb_x, -1) + x_mean.to(dev).unsqueeze(0).expand(nb_x, -1)
        return logit_back(x)


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
        scheduler.step(train_loss)
        return train_loss / len(train_loader.dataset)

    def test(epoch):
        test_loss = 0
        for batch_idx, (data, _) in enumerate(test_loader):
            #data = sample
            x0 = data.view(data.shape[0], -1).to(dev)

            x0 = (x0 - x_mean.to(dev).unsqueeze(0).expand(bs, -1)) / x_std.to(dev).unsqueeze(0).expand(bs, -1)
            optimizer.zero_grad()

            loss = model.loss(x0)

            test_loss += loss.item()
        return test_loss / len(test_loader.dataset)

    for i in range(500):
        train_loss = train(i)
        test_loss = test(i)
        model.alpha = i/500.
        samples = get_X_back(model.sample(64)).view(64, *img_size)
        #save_image(samples, './Samples/Generated/sample_gen_' + str(i) + '.png')
        print('====> Epoch: {} - Average Train loss: {:.4f} - Average Test Loss: {:.4f}'.format(i, train_loss, test_loss))
        wandb.log({"Train Loss": train_loss,
                   "Test Loss": test_loss,
                   "Samples": [wandb.Image(samples)],
                   "epoch": i})
