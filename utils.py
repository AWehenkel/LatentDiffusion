from torchvision import datasets, transforms
import numpy as np
import torch
from PIL import Image, ImageFilter
import torch.nn as nn

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    #x = x / 256
    return x


class RandomBlur(object):
    def __init__(self, T, level_max):
        super().__init__()
        self.T = T
        self.level_max = level_max

    def __call__(self, x):
        t = np.random.randint(0, self.T + 1)
        level = self.level_max / self.T * t
        gaussImage = x.filter(ImageFilter.GaussianBlur(level))
        return torch.tensor(np.array(x)), torch.tensor(np.array(gaussImage)).permute(2, 0, 1), torch.tensor([t])

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


def getDataLoader(dataset, bs, T=100, level_max=5.):
    if dataset == "MNIST":
        # MNIST Dataset
        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.Compose([
                                          transforms.Resize((32, 32)), ToTensor(),  AddUniformNoise()]))
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          ToTensor(),
                                          AddUniformNoise()
                                      ]))

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
        img_size = [1, 32, 32]
    elif dataset == "CIFAR10":
        # CIFAR10 Dataset
        train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             add_noise
                                         ]))
        test_dataset = datasets.CIFAR10(root='./cifar10_data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(32),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            add_noise
                                        ]))

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
        img_size = [3, 32, 32]
    elif dataset == "Heated_CIFAR10":
        # CIFAR10 Dataset
        train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, download=True,
                                         transform=transforms.Compose([
                                            transforms.Resize(32),
                                            transforms.RandomHorizontalFlip(),
                                            RandomBlur(T=T, level_max=level_max)
                                         ]))
        test_dataset = datasets.CIFAR10(root='./cifar10_data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(32),
                                            transforms.RandomHorizontalFlip(),
                                            RandomBlur(T=T, level_max=level_max)
                                        ]))

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
        img_size = [3, 32, 32]
    return train_loader, test_loader, img_size


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