import torch
import torch.optim as optim
from Models import AugmentedVAEModel
import wandb
from utils import getDataLoader, logit_back
wandb.init(project="heated_vae", entity="awehenkel")


if __name__ == "__main__":
    bs = 100
    config = {
        'data': 'Heated_CIFAR10',
        'latent_s': 64,
        'CNN': True,
        'enc_w': 500,
        'enc_l': 4,
        'dec_w': 500,
        'dec_l': 4,
        't_emb_s': 100,
        'nb_steps': 1000,
        'level_max': 4.,
        'debug': False
    }

    debug = config['debug']

    wandb.config.update(config)
    config = wandb.config
    train_loader, test_loader, img_size = getDataLoader(config["data"], bs, nb_steps=config['nb_steps'], level_max=config['level_max'])
    _, test_loader, _ = getDataLoader("CIFAR10", bs, nb_steps=config['nb_steps'], level_max=config['level_max'])

    config["img_size"] = img_size
    # Compute Mean abd std per pixel
    x_mean = 0
    x_mean2 = 0
    for batch_idx, ([_, cur_x, _], target) in enumerate(train_loader):
        cur_x = cur_x.view(bs, -1).float()
        x_mean += cur_x.mean(0)
        x_mean2 += (cur_x ** 2).mean(0)
    x_mean /= batch_idx + 1
    x_std = (x_mean2 / (batch_idx + 1) - x_mean ** 2) ** .5
    x_std[x_std == 0.] = 1.

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = AugmentedVAEModel(**config).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    wandb.watch(model)
    def get_X_back(x):
        nb_x = x.shape[0]
        x = x * x_std.to(dev).unsqueeze(0).expand(nb_x, -1) + x_mean.to(dev).unsqueeze(0).expand(nb_x, -1)
        return x


    if debug:
        x = next(iter(test_loader))[0][[1]].expand(bs, -1, -1, -1)

    def train(epoch):
        train_loss = 0
        for batch_idx, ([_, data, t], _) in enumerate(train_loader):
            if debug:
                x0 = x.to(dev).view(x.shape[0], -1)
                t = torch.zeros(x.shape[0], 1).to(dev)
            else:
                x0 = data.view(data.shape[0], -1).to(dev)
                t = t.to(dev)

            x0 = (x0 - x_mean.to(dev).unsqueeze(0).expand(x0.shape[0], -1)) / x_std.to(dev).unsqueeze(0).expand(x0.shape[0], -1)
            optimizer.zero_grad()

            loss = model.loss(x0, t)

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
            if debug:
                x0 = x.to(dev).view(x.shape[0], -1)
                t = torch.zeros(x.shape[0], 1).to(dev)
            else:
                t = torch.zeros((data.shape[0], 1), device=dev)
                x0 = data.view(data.shape[0], -1).to(dev)


            x0 = (x0 - x_mean.to(dev).unsqueeze(0).expand(x0.shape[0], -1)) / x_std.to(dev).unsqueeze(0).expand(x0.shape[0], -1)
            optimizer.zero_grad()

            loss = model.loss(x0, t)

            test_loss += loss.item()
        reconstructed_test = model(x0[:64], t[:64])
        return test_loss / len(test_loader.dataset), reconstructed_test, x0[:64]

    for i in range(150):
        train_loss = train(i)
        test_loss, x_rec, x = test(i)
        x = get_X_back(x.view(64, -1)).view(64, *img_size)
        x_rec = get_X_back(x_rec.view(64, -1)).view(64, *img_size)
        samples_1 = get_X_back(model.sample(64)).view(64, *img_size)
        samples_8 = get_X_back(model.sample(64, temperature=.8)).view(64, *img_size)
        print('====> Epoch: {} - Average Train loss: {:.4f} - Average Test Loss: {:.4f}'.format(i, train_loss, test_loss))
        wandb.log({"Train Loss": train_loss,
                   "Test Loss": test_loss,
                   "Samples T°100": [wandb.Image(samples_1)],
                   "Samples T°80": [wandb.Image(samples_8)],
                   "Reconstructed": [wandb.Image(x_rec)],
                   "Data": [wandb.Image(x)],
                   "epoch": i})
