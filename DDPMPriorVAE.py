import torch
import torch.optim as optim
from Models import VAEModel, DDPMPriorVAEModel, NFPriorVAEModel, TwoStagesDDPMPriorVAEModel
import wandb
from utils import getDataLoader
import torch.nn as nn
import argparse
import os

os.environ['WANDB_SILENT'] = 'false'


wandb.init(project='vae', entity='awehenkel')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


model_dict = {"DDPM": DDPMPriorVAEModel, "Classical": VAEModel, "NF": NFPriorVAEModel,
              "TwoStages": TwoStagesDDPMPriorVAEModel}


if __name__ == '__main__':
    config = {
        'data': 'MNIST',
        'latent_s': [200],
        'CNN': True,
        'enc_w': 400,
        'enc_l': 1,
        'dec_w': 400,
        'dec_l': 1,
        'trans_w': 300,
        'trans_l': 3,
        'n_res_blocks': 1,
        "beta_min": [.0001],
        "beta_max": [.02],
        'simplified_trans': True,
        't_emb_s': 100,
        't_min': [0],
        't_max': [1000],
        'batch_size': 100,
        'lr': .0005,
        'nb_epoch': 250,
        'model': 'DDPM',
        'cond_w': 300,
        'cond_l': 4,
        'n_nf_steps': 3,
        'n_workers': 4,
        'exact_ddpm_loss': False,
        'KL_prior_diffusion': True
    }

    parser = argparse.ArgumentParser(description='VAE running parameters')
    for k, v in config.items():
        if isinstance(v, type(True)):
            parser.add_argument('-' + k, default=v, type=str2bool)
        elif isinstance(v, type([])):
            parser.add_argument('-' + k, default=v, nargs='+', type=type(v[0]))
        else:
            parser.add_argument('-' + k, default=v, type=type(v))

    config = parser.parse_args()

    wandb.config.update(config)
    config = wandb.config

    bs = int(config['batch_size'])
    nb_epoch = int(config['nb_epoch'])

    train_loader, test_loader, img_size = getDataLoader(config['data'], bs, n_workers=config['n_workers'])
    config['img_size'] = img_size

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model_dict[config["model"]](**config).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=0.1, threshold_mode='rel', cooldown=0,
                                                           min_lr=0, eps=1e-08, verbose=True)

    wandb.watch(model)
    def get_X_back(x0):
        x = (x0 * .5) + .5
        return x


    def train(epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            x0 = data.view(data.shape[0], -1).to(dev)

            optimizer.zero_grad()

            loss = model.loss(x0)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .25)

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
            x0 = data.view(data.shape[0], -1).to(dev)

            optimizer.zero_grad()

            loss = model.loss(x0)

            test_loss += loss.item()
        reconstructed_test = model(x0[:64])
        return test_loss / len(test_loader.dataset), reconstructed_test, x0[:64]


    best_test_loss = float("inf")

    for i in range(nb_epoch):
        model.train()
        train_loss = train(i)
        model.eval()
        with torch.no_grad():
            test_loss, x_rec, x = test(i)
            x = get_X_back(x.view(64, -1)).view(64, *img_size)
            x_rec = get_X_back(x_rec.view(64, -1)).view(64, *img_size)
            samples = get_X_back(model.sample(64)).view(64, *img_size)
            print('====> Epoch: {} - Average Train loss: {:.4f} - Average Test Loss: {:.4f}'.format(i, train_loss, test_loss))
            wandb.log({"Train Loss": train_loss,
                       "Test Loss": test_loss,
                       "Samples": [wandb.Image(samples)],
                       "Reconstructed": [wandb.Image(x_rec)],
                       "Data": [wandb.Image(x)],
                       "epoch": i})
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "last_model.h5"))
            torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, "last_optimizer.h5"))
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.h5"))
            torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, "best_optimizer.h5"))
            best_test_loss = test_loss
