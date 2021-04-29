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
    x = x / 256
    return x

def seed_set(i):
    np.random.seed(i)
    
class RandomBlur(object):
    def __init__(self, T, level_max):
        super().__init__()
        self.T = T
        self.level_max = level_max
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.seeded = False

    def __call__(self, x):

        if not self.seeded:
            self.seeded = True
            self.random = np.random.RandomState()
        x0 = x
        t = self.random.randint(1, self.T)
        level = self.level_max / self.T * t * (t + 1)/2
        xt = x.filter(ImageFilter.GaussianBlur(level))
        level = self.level_max / self.T * t * (t - 1) / 2
        xt_1 = x.filter(ImageFilter.GaussianBlur(level))
        return self.norm(add_noise(self.to_tensor(x0))), self.norm(add_noise(self.to_tensor(xt))), self.norm(add_noise(self.to_tensor(xt_1))), torch.tensor([t])

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


def getDataLoader(dataset, bs, T=100, level_max=5., n_workers=4, pin_memory=True):
    if dataset == "MNIST":
        # MNIST Dataset
        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.Compose([
                                          transforms.Resize((32, 32)), transforms.ToTensor(),
                                            add_noise,
                                          transforms.Normalize((0.5), (0.5))]))
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                            add_noise,
                                          transforms.Normalize((0.5), (0.5))
                                      ]))

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,
                                                   num_workers=n_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,
                                                  num_workers=n_workers, pin_memory=pin_memory)
        img_size = [1, 32, 32]
    elif dataset == "CIFAR10":
        # CIFAR10 Dataset
        train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            add_noise,
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))
        test_dataset = datasets.CIFAR10(root='./cifar10_data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(32),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            add_noise,
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,
                                                   num_workers=n_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,
                                                  num_workers=n_workers, pin_memory=pin_memory)
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
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,
                                                   num_workers=n_workers, worker_init_fn=seed_set, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,
                                                  num_workers=n_workers, pin_memory=pin_memory, worker_init_fn=seed_set)
        img_size = [3, 32, 32]

    elif dataset == "celeba":
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 64
        train_dataset = datasets.CelebA(dataroot, split='train', download=False,
                                       transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            add_noise,
                                       #transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, pin_memory=pin_memory,
                                                 shuffle=True, num_workers=n_workers, drop_last=True)
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 64
        test_dataset = datasets.CelebA(dataroot, split='test',download=False,
                                             transform=transforms.Compose([
                                                 transforms.Resize(image_size),
                                                 transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            add_noise
                                                 #transforms.ToTensor(),
                                                 ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ]))
        # Create the dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, pin_memory=pin_memory,
                                                   shuffle=False, num_workers=n_workers, drop_last=True)
        img_size = [3, 64, 64]

    elif dataset == "Heated_celeba":
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 64
        train_dataset = datasets.CelebA(dataroot, split='train', download=False,
                                                transform=transforms.Compose([
                                                 transforms.Resize(image_size),
                                                 transforms.CenterCrop(image_size),
                                                 RandomBlur(T=T, level_max=level_max)
                                             ]))
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, pin_memory=pin_memory,
                                                   shuffle=True, num_workers=n_workers, drop_last=True,
                                        worker_init_fn=seed_set)
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 64
        test_dataset = datasets.CelebA(dataroot, split='test', download=False,
                                            transform=transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.CenterCrop(image_size),
                                                RandomBlur(T=T, level_max=level_max)
                                            ]))
        # Create the dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, pin_memory=pin_memory,
                                                  shuffle=False, num_workers=n_workers, drop_last=True,
                                        worker_init_fn=seed_set)

        img_size = [3, 64, 64]

    elif dataset == "celeba_HQ":
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 256
        train_dataset = datasets.CelebA(dataroot, split='train', download=False,
                                       transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            add_noise,
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, pin_memory=pin_memory,
                                                 shuffle=True, num_workers=n_workers, drop_last=True,
                                        worker_init_fn=seed_set)
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 256
        test_dataset = datasets.CelebA(dataroot, split='test',download=False,
                                             transform=transforms.Compose([
                                                 transforms.Resize(image_size),
                                                 transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            add_noise,
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ]))
        # Create the dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, pin_memory=pin_memory,
                                                   shuffle=False, num_workers=n_workers, drop_last=True,
                                        worker_init_fn=seed_set)
        img_size = [3, 256, 256]

    elif dataset == "Heated_celeba_HQ":
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 256
        train_dataset = datasets.CelebA(dataroot, split='train',download=False,
                                                transform=transforms.Compose([
                                                 transforms.Resize(image_size),
                                                 transforms.CenterCrop(image_size),
                                                 RandomBlur(T=T, level_max=level_max)
                                             ]))
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, pin_memory=pin_memory,
                                                   shuffle=True, num_workers=n_workers, drop_last=True,
                                        worker_init_fn=seed_set)
        dataroot = '/scratch/users/awehenkel/celeba/'
        if not torch.cuda.is_available():
            dataroot = '.'
        image_size = 256
        test_dataset = datasets.CelebA(dataroot, split='test', download=False,
                                            transform=transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.CenterCrop(image_size),
                                                RandomBlur(T=T, level_max=level_max)
                                            ]))
        # Create the dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, pin_memory=pin_memory,
                                                  shuffle=False, num_workers=n_workers, drop_last=True,
                                        worker_init_fn=seed_set)

        img_size = [3, 256, 256]

    elif dataset == "Heated_CIFAR10_DEBUG":
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
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, drop_last=True, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True, pin_memory=pin_memory)
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