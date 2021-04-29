import torch
import torch.optim as optim
from Models import CNNHeatedLatentDiffusion
import wandb
from utils import getDataLoader, logit_back
from multiprocessing import Process, freeze_support
import os
import torch.nn as nn
import argparse




def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



config_celeba = {
        'data': 'celeba',

        'CNN': True,
        'enc_w': 300,
        'enc_l': 1,
        'dec_w': 300,
        'dec_l': 1,
        'trans_w': 400,
        'trans_l': 3,
        'n_res_blocks': 5,
        "beta_min": 0.01,
        "beta_max": .75,
        'simplified_trans': False,
        't_emb_s': 50,
        'T': 1000,
        'level_max': .075,
        'debug': False,
        'ts_min': [0],
        'ts_max': [1000],
        'var_sizes': [200],
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
        'n_res_blocks': 3,
        "beta_min": 0.01,
        "beta_max": .35,
        'simplified_trans': False,
        't_emb_s': 50,
        'T': 1000,
        'level_max': .15,
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
        'trans_l': 4,
        'n_res_blocks': 1,
        "beta_min": 0.0001,
        "beta_max": .02,
        'simplified_trans': True,
        't_emb_s': 100,
        'level_max': .025,
        'debug': False,
        'ts_min': [0, 250, 500, 750, 1000],
        'ts_max': [1000, 1250, 1500, 1750, 2000],
        'var_sizes': [40, 40, 40, 40, 40],
        'decoder_type': 'Progressive2',
        'batch_size': 256
    }

if __name__ == "__main__":
    freeze_support()
    wandb.init(project="heat_diffusion", entity="awehenkel")

    config = config_cifar

    parser = argparse.ArgumentParser(description='Heat diffusion running parameters')
    for k, v in config.items():
        if isinstance(v, type(True)):
            parser.add_argument("-" + k, default=v, type=str2bool)
        elif isinstance(v, type([])):
            parser.add_argument("-" + k, default=v, nargs="+", type=type(v[0]))
        else:
            parser.add_argument("-" + k, default=v, type=type(v))

    config = vars(parser.parse_args())

    bs = int(config['batch_size'])
    n_epoch = 500

    debug = config['debug']

    wandb.config.update(config)
    config = wandb.config
    train_loader, test_loader, img_size = getDataLoader("Heated_" + config["data"], bs, T=max(config['ts_max']), level_max=config['level_max'])
    _, test_loader, _ = getDataLoader(config["data"], 100, T=max(config['ts_max']), level_max=config['level_max'])

    config["img_size"] = img_size
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CNNHeatedLatentDiffusion(**config).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=.0005)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
    #                                                       patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    wandb.watch(model)

    def get_X_back(x0):
        x = (x0 * .5) + .5
        return x


    if debug:
        [x0, xt, t], _ = next(iter(train_loader))
        x0_debug = x0[[1]].expand(bs, -1, -1, -1)
        xt = xt[[1]].expand(bs, -1, -1, -1)
        t = t[[1]].expand(bs, -1, -1, -1)

    def train(epoch):
        train_loss = 0
        for batch_idx, ([x0, xt, xt_1, t], _) in enumerate(train_loader):
            if debug:
                x0 = x.to(dev, non_blocking=True).view(x.shape[0], -1)
                t = torch.zeros(x.shape[0], 1).to(dev, non_blocking=True)
            else:
                x0 = x0.view(x0.shape[0], -1).to(dev, non_blocking=True)
                xt = xt.view(x0.shape[0], -1).to(dev, non_blocking=True)
                xt_1 = xt_1.view(x0.shape[0], -1).to(dev, non_blocking=True)
                t = t.to(dev, non_blocking=True)

            optimizer.zero_grad()

            loss = model.loss(x0, xt, xt_1, t)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), .25)

            train_loss += loss.detach()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * bs, len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / bs))
        #scheduler.step(train_loss)
        return train_loss.item() / len(train_loader.dataset)

    def test(epoch):
        test_loss = 0
        for batch_idx, (x0, _) in enumerate(test_loader):
            if debug:
                x0 = x.to(dev).view(x.shape[0], -1)
                t = torch.zeros(x.shape[0], 1).to(dev).long()
            else:
                x0 = x0.view(x0.shape[0], -1).to(dev, non_blocking=True)
                xt = x0
                t = torch.zeros(x0.shape[0], 1).to(dev, non_blocking=True).long() + 1

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

