import torch
import torch.optim as optim
from Models import CNNHeatedLatentDiffusion
import wandb
from utils import getDataLoader, logit_back
from multiprocessing import Process, freeze_support
import os
import torch.nn as nn

config_celeba = {
        'data': 'celeba',

        'CNN': True,
        'enc_w': 300,
        'enc_l': 1,
        'dec_w': 300,
        'dec_l': 1,
        'trans_w': 1000,
        'trans_l': 5,
        "beta_min": 0.01,
        "beta_max": .75,
        'simplified_trans': False,
        't_emb_s': 50,
        'T': 50,
        'level_max': 1.5,
        'debug': False,
        'ts_min': [0, 5, 10, 15, 20, 30, 35, 40],
        'ts_max': [10, 15, 20, 30, 35, 40, 45, 50],
        'var_sizes': [50, 50, 50, 50, 50, 50, 50, 50],
        'decoder_type': 'Progressive2',
        'batch_size': 256
    }
config_celeba_hq = {
        'data': 'celeba_HQ',
        'CNN': True,
        'enc_w': 300,
        'enc_l': 1,
        'dec_w': 300,
        'dec_l': 1,
        'trans_w': 1000,
        'trans_l': 5,
        "beta_min": 0.01,
        "beta_max": .35,
        'simplified_trans': False,
        't_emb_s': 50,
        'T': 100,
        'level_max': 2.5,
        'debug': False,
        'ts_min': [60, 45, 30, 15, 0],
        'ts_max': [100, 85, 75, 55, 40],
        'var_sizes': [150, 100, 50, 50, 50],
        'decoder_type': 'Progressive2',
        'batch_size': 64
    }

config_cifar = {
        'data': 'CIFAR10',
        'CNN': True,
        'enc_w': 300,
        'enc_l': 1,
        'dec_w': 300,
        'dec_l': 1,
        'trans_w': 500,
        'trans_l': 5,
        "beta_min": 0.01,
        "beta_max": .3,
        'simplified_trans': False,
        't_emb_s': 50,
        'T': 50,
        'level_max': 1.25,
        'debug': False,
        'ts_min': [0, 5, 10, 15, 20, 30, 35, 40],
        'ts_max': [10, 15, 20, 30, 35, 40, 45, 50],
        'var_sizes': [50, 50, 50, 50, 50, 50, 50, 50],
        'decoder_type': 'Progressive2',
        'batch_size': 256
    }

if __name__ == "__main__":
    freeze_support()
    wandb.init(project="heat_diffusion", entity="awehenkel")

    config = config_cifar


    bs = config['batch_size']
    n_epoch = 500

    debug = config['debug']

    wandb.config.update(config)
    config = wandb.config
    train_loader, test_loader, img_size = getDataLoader("Heated_" + config["data"], bs, T=config['T'], level_max=config['level_max'])
    _, test_loader, _ = getDataLoader(config["data"], 100, T=config['T'], level_max=config['level_max'])

    config["img_size"] = img_size
    # Compute Mean abd std per pixel
    '''
    path = config["data"] + 'HD_standardizer.pkl'
    if os.path.exists(path):
        [x0_mean, x0_std, xt_mean, xt_std] = torch.load(path)
    else:
        x0_mean = 0
        x0_mean2 = 0
        xt_mean = 0
        xt_mean2 = 0
        for batch_idx, ([x0, xt, t], target) in enumerate(train_loader):
            x0 = x0.view(x0.shape[0], -1).float()
            x0_mean += x0.mean(0)
            x0_mean2 += (x0 ** 2).mean(0)

            xt = xt.view(x0.shape[0], -1).float()
            xt_mean += xt.mean(0)
            xt_mean2 += (xt ** 2).mean(0)
        x0_mean /= batch_idx + 1
        x0_std = (x0_mean2 / (batch_idx + 1) - x0_mean ** 2) ** .5
        x0_std[x0_std == 0.] = 1.

        xt_mean /= batch_idx + 1
        xt_std = (xt_mean2 / (batch_idx + 1) - xt_mean ** 2) ** .5
        xt_std[xt_std == 0.] = 1.

        torch.save([x0_mean, x0_std, xt_mean, xt_std], path)
'''
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CNNHeatedLatentDiffusion(**config).to(dev)

    optimizer = optim.AdamW(model.parameters(), lr=.0001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
    #                                                       patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    wandb.watch(model)

    def get_X_back(x0):
        nb_x = x0.shape[0]
        x = (x0 * .5) + .5
        #x = x0 * x0_std.to(dev).unsqueeze(0).expand(nb_x, -1) + x0_mean.to(dev).unsqueeze(0).expand(nb_x, -1)
        return x


    if debug:
        [x0, xt, t], _ = next(iter(train_loader))
        x0_debug = x0[[1]].expand(bs, -1, -1, -1)
        xt = xt[[1]].expand(bs, -1, -1, -1)
        t = t[[1]].expand(bs, -1, -1, -1)
    #x0_mean = x0_mean.to(dev)
    #x0_std = x0_std.to(dev)
    #xt_mean = xt_mean.to(dev)
    #xt_std = xt_std.to(dev)
    def train(epoch):
        train_loss = 0
        for batch_idx, ([x0, xt, xt_1, t], _) in enumerate(train_loader):
            if debug:
                x0 = x.to(dev).view(x.shape[0], -1)
                t = torch.zeros(x.shape[0], 1).to(dev)
            else:
                x0 = x0.view(x0.shape[0], -1).to(dev)
                xt = xt.view(x0.shape[0], -1).to(dev)
                xt_1 = xt_1.view(x0.shape[0], -1).to(dev)
                t = t.to(dev)

            #x0 = (x0 - x0_mean.unsqueeze(0).expand(x0.shape[0], -1)) / x0_std.unsqueeze(0).expand(x0.shape[0], -1)
            #xt = (xt - xt_mean.unsqueeze(0).expand(xt.shape[0], -1)) / xt_std.unsqueeze(0).expand(xt.shape[0], -1)



            optimizer.zero_grad()

            loss = model.loss(x0, xt, xt_1, t)

            loss.backward()

            #nn.utils.clip_grad_norm_(model.parameters(), .0000001)

            train_loss += loss.detach()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * bs, len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / bs))
        scheduler.step(train_loss)
        return train_loss.item() / len(train_loader.dataset)

    def test(epoch):
        test_loss = 0
        for batch_idx, (x0, _) in enumerate(test_loader):
            #data = sample
            if debug:
                x0 = x.to(dev).view(x.shape[0], -1)
                t = torch.zeros(x.shape[0], 1).to(dev).long()
            else:
                x0 = x0.view(x0.shape[0], -1).to(dev)
                xt = x0
                t = torch.zeros(x0.shape[0], 1).to(dev).long()

            #x0 = (x0 - x0_mean.unsqueeze(0).expand(x0.shape[0], -1)) / x0_std.unsqueeze(0).expand(
            #    x0.shape[0], -1)
            #xt = (xt - xt_mean.unsqueeze(0).expand(xt.shape[0], -1)) / xt_std.unsqueeze(0).expand(
            #    xt.shape[0], -1)
            optimizer.zero_grad()

            loss = model.loss(x0, xt, xt,  t)

            test_loss += loss.detach()
        reconstructed_test = model(x0[:64], t[:64])
        return test_loss.item() / len(test_loader.dataset), reconstructed_test, x0[:64]

    for i in range(n_epoch):
        model.train()
        train_loss = train(i)
        model.eval()
        with torch.no_grad():
            test_loss, x_rec, x = test(i)
            x = get_X_back(x.view(64, -1)).view(64, *img_size)
            x_rec = get_X_back(x_rec.view(64, -1)).view(64, *img_size)
            samples_1 = get_X_back(model.sample(64)).view(64, *img_size)
            samples_8 = get_X_back(model.sample(64, temperature=.8)).view(64, *img_size)
            samples_3 = get_X_back(model.sample(64, temperature=.3)).view(64, *img_size)
        print('====> Epoch: {} - Average Train loss: {:.4f} - Average Test Loss: {:.4f}'.format(i, train_loss, test_loss))
        torch.save(model.state_dict(), 'saved_models/' + wandb.run.name + '.pt')

        wandb.log({"Train Loss": train_loss,
                   "Test Loss": test_loss,
                   "Samples T°100": [wandb.Image(samples_1)],
                   "Samples T°80": [wandb.Image(samples_8)],
                   "Samples T°30": [wandb.Image(samples_3)],
                   "Reconstructed": [wandb.Image(x_rec)],
                   "Data": [wandb.Image(x)],
                   "epoch": i})

