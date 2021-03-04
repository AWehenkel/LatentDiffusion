import torch
import torch.nn as nn


class TemporalDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, layers, t_dim=1, act=nn.SELU, out_dim=[1, 32, 32]):
        super(TemporalDecoder, self).__init__()
        # decoder part
        layers = [z_dim + t_dim] + layers + [x_dim]
        net = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)
        self.out_dim = out_dim

    def forward(self, z, t=None):
        if t is not None:
            z = torch.cat((z, t), 1)
        return self.net(z).view(z.shape[0], *self.out_dim)


class SimpleImageDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=1, pos_enc=None, act=nn.SELU, out_c=1):
        super(SimpleImageDecoder, self).__init__()
        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalDecoder(features_dim, z_dim, layers, t_dim, act, self.features_dim)
        init_channels = 8
        kernel_size = 4
        self.conv = nn.Sequential(nn.ConvTranspose2d(self.features_dim[0], init_channels * 8, kernel_size, stride=1, padding=0),
                                  act(),
                                  nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 2, out_c, kernel_size, stride=2,
                                                     padding=1))
        self.pos_enc = pos_enc

    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        features = self.fc(z, t)
        return self.conv(features)


class ImprovedImageDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=1, pos_enc=None, act=nn.SELU, out_c=1):
        super(ImprovedImageDecoder, self).__init__()
        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        self.fc = TemporalDecoder(features_dim, z_dim, layers, t_dim, act, self.features_dim)
        init_channels = 8
        kernel_size = 4
        self.conv = nn.Sequential(nn.ConvTranspose2d(self.features_dim[0], init_channels * 8, kernel_size, stride=1, padding=0),
                                  act(),
                                  nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(init_channels * 2, 10, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.ConvTranspose2d(10, 10, kernel_size, stride=2,
                                                     padding=1), act(),
                                  nn.Conv2d(10, 5, 2, 1), act(),
                                  nn.Conv2d(5, out_c, 2, 2, 1)
                                  )
        self.pos_enc = pos_enc


    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        features = self.fc(z, t)
        return self.conv(features)



class DCDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=0, pos_enc=None, act=nn.SELU, out_c=1, img_width=32):
        super(DCDecoder, self).__init__()
        self.pos_enc = pos_enc

        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        #self.fc = nn.Sequential(nn.Linear(z_dim + t_dim, z))#TemporalDecoder(features_dim, z_dim, layers, t_dim, act, self.features_dim)
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        image_size = 64

        # Number of channels in the training images. For color images this is 3
        nc = out_c

        # Size of z latent vector (i.e. size of generator input)
        nz = sum(z_dim)#features_dim

        # Size of feature maps in generator
        ngf = 64
        if t_dim > 0:
            self.weight_z = nn.Linear(t_dim, sum(z_dim))

        if img_width == 32:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                act(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                act(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                act(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ngf),
                nn.Tanh()
                # state size. (ngf) x 32 x 32
                #nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                #act()
                # state size. (nc) x 64 x 64
            )
        elif img_width == 64:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                #nn.BatchNorm2d(ngf * 8),
                act(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ngf * 4),
                act(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ngf * 2),
                act(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ngf),
                act(),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        elif img_width == 256:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 32),
                act(),
                nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 16),
                act(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                act(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                act(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf*2),
                act(),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                act(),
                # state size. (nc) x 64 x 64
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )


    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        if t is None:
            return self.main(z.unsqueeze(2).unsqueeze(2))
        z = torch.sigmoid(self.weight_z(t)) * z
        return self.main(z.unsqueeze(2).unsqueeze(2))


class ProgressiveDecoder(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=0, pos_enc=None, act=nn.SELU, out_c=1, img_width=32):
        super(ProgressiveDecoder, self).__init__()
        self.pos_enc = pos_enc

        self.features_dim = features_dim
        features_dim = self.features_dim[0] * self.features_dim[1] * self.features_dim[2]
        #self.fc = nn.Sequential(nn.Linear(z_dim + t_dim, z))#TemporalDecoder(features_dim, z_dim, layers, t_dim, act, self.features_dim)
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        image_size = 64

        # Number of channels in the training images. For color images this is 3
        nc = out_c

        # Size of z latent vector (i.e. size of generator input)
        nz = z_dim#features_dim

        # Size of feature maps in generator
        ngf = 64
        if t_dim > 0:
            self.weight_z = nn.Linear(t_dim, sum(z_dim))
        self.img_width = img_width
        self.nz = nz

        if img_width == 32:
            self.t_conv1 = nn.Sequential(nn.ConvTranspose2d(nz[0], ngf * 8, 4, 1, 0, bias=False),
                                         nn.BatchNorm2d(ngf * 8),
                                         act(), # state size. (ngf*8) x 4 x 4
                                         nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
                                         )
            self.t_conv1_bis = nn.ConvTranspose2d(nz[1], ngf * 4, 8, 1, 0, bias=False)

            self.t_conv2 = nn.Sequential(nn.BatchNorm2d(ngf * 4 * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False))
            self.t_conv2_bis = nn.ConvTranspose2d(nz[2], ngf * 2, 16, 1, 0, bias=False)

            self.t_conv3 = nn.Sequential(nn.BatchNorm2d(ngf * 2 * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 2 * 2, nc, 4, 2, 1, bias=False))
            self.t_conv3_bis = nn.Sequential(nn.ConvTranspose2d(nz[3], nc, 32, 1, 0, bias=False), act())

            self.conv3 = nn.Sequential(nn.Conv2d(nc*2, nc, 3, padding=1), nn.Tanh())


        elif img_width == 64:
            self.t_conv1 = nn.Sequential(nn.ConvTranspose2d(nz[0], ngf * 8, 4, 1, 0, bias=False),
                                         nn.BatchNorm2d(ngf * 8),
                                         act(),  # state size. (ngf*8) x 4 x 4
                                         nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
                                         )
            self.t_conv1_bis = nn.ConvTranspose2d(nz[1], ngf * 4, 8, 1, 0, bias=False)

            self.t_conv2 = nn.Sequential(nn.BatchNorm2d(ngf * 4 * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False))
            self.t_conv2_bis = nn.ConvTranspose2d(nz[2], ngf * 2, 16, 1, 0, bias=False)

            self.t_conv3 = nn.Sequential(nn.BatchNorm2d(ngf * 2 * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False))
            self.t_conv3_bis = nn.Sequential(nn.ConvTranspose2d(nz[3], ngf, 32, 1, 0, bias=False), act())

            self.t_conv4 = nn.Sequential(nn.BatchNorm2d(ngf * 2),
                                         act(),
                                         nn.ConvTranspose2d(2*ngf, nc, 4, 2, 1, bias=False))

            self.t_conv4_bis = nn.Sequential(nn.ConvTranspose2d(nz[4], nc, 64, 1, 0, bias=False), act())

            self.conv4 = nn.Sequential(nn.Conv2d(nc * 2, nc, 3, padding=1), nn.Tanh())

        elif img_width == 256:
            return None


    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        if t is None:
            return self.main(z.unsqueeze(2).unsqueeze(2))

        z = torch.sigmoid(self.weight_z(t)) * z

        z = torch.split(z.unsqueeze(2).unsqueeze(2), self.nz, 1)

        f1 = self.t_conv1(z[0])
        f1_bis = self.t_conv1_bis(z[1])
        f2 = self.t_conv2(torch.cat((f1, f1_bis), 1))
        f2_bis = self.t_conv2_bis(z[2])
        f3 = self.t_conv3(torch.cat((f2, f2_bis), 1))
        f3_bis = self.t_conv3_bis(z[3])


        if self.img_width == 32:
            out = self.conv3(torch.cat((f3, f3_bis), 1))
        elif self.img_width == 64:
            f4 = self.t_conv4(torch.cat((f3, f3_bis), 1))
            f4_bis = self.t_conv4_bis(z[4])
            out = self.conv4(torch.cat((f4, f4_bis), 1))

        return out#self.main(z.unsqueeze(2).unsqueeze(2))



class ProgressiveDecoder2(nn.Module):
    def __init__(self, features_dim, z_dim, layers, t_dim=0, pos_enc=None, act=nn.SELU, out_c=1, img_width=32):
        super(ProgressiveDecoder2, self).__init__()
        self.pos_enc = pos_enc
        # Number of channels in the training images. For color images this is 3
        nc = out_c

        # Size of z latent vector (i.e. size of generator input)
        nz = sum(z_dim)#features_dim

        # Size of feature maps in generator
        ngf = 64
        self.img_width = img_width
        self.nz = nz

        if img_width == 32:
            self.weight_z = nn.Sequential(nn.Linear(t_dim, sum(z_dim)), nn.Sigmoid())
            self.weight_z1 = nn.Sequential(nn.Linear(t_dim, ngf * 4), nn.Sigmoid())
            self.weight_z2 = nn.Sequential(nn.Linear(t_dim, ngf * 2), nn.Sigmoid())
            self.weight_z3 = nn.Sequential(nn.Linear(t_dim, ngf), nn.Sigmoid())
            self.t_conv1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                         nn.BatchNorm2d(ngf * 8),
                                         act(), # state size. (ngf*8) x 4 x 4
                                         nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
                                         )

            self.t_conv2 = nn.Sequential(nn.BatchNorm2d(ngf * 4),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))

            self.t_conv3 = nn.Sequential(nn.BatchNorm2d(ngf * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False))

            self.conv = nn.Sequential(nn.Conv2d(ngf, nc, 3, padding=1), nn.Tanh())


        elif img_width == 64:
            self.weight_z = nn.Sequential(nn.Linear(t_dim, sum(z_dim)), nn.Sigmoid())
            self.weight_z1 = nn.Sequential(nn.Linear(t_dim, ngf * 4), nn.Sigmoid())
            self.weight_z2 = nn.Sequential(nn.Linear(t_dim, ngf * 2), nn.Sigmoid())
            self.weight_z3 = nn.Sequential(nn.Linear(t_dim, ngf), nn.Sigmoid())
            self.weight_z4 = nn.Sequential(nn.Linear(t_dim, ngf), nn.Sigmoid())

            self.t_conv1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                         nn.BatchNorm2d(ngf * 8),
                                         act(),  # state size. (ngf*8) x 4 x 4
                                         nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
                                         )

            self.t_conv2 = nn.Sequential(nn.BatchNorm2d(ngf * 4),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))

            self.t_conv3 = nn.Sequential(nn.BatchNorm2d(ngf * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False))

            self.t_conv4 = nn.Sequential(nn.BatchNorm2d(ngf),
                                         act(),
                                         nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False))


            self.conv = nn.Sequential(nn.Conv2d(ngf, nc, 3, padding=1), nn.Tanh())

        elif img_width == 256:
            self.weight_z = nn.Sequential(nn.Linear(t_dim, sum(z_dim)), nn.Sigmoid())
            self.weight_z0 = nn.Sequential(nn.Linear(t_dim, ngf * 16), nn.Sigmoid())
            self.weight_z1 = nn.Sequential(nn.Linear(t_dim, ngf * 16), nn.Sigmoid())
            self.weight_z2 = nn.Sequential(nn.Linear(t_dim, ngf * 8), nn.Sigmoid())
            self.weight_z3 = nn.Sequential(nn.Linear(t_dim, ngf * 4), nn.Sigmoid())
            self.weight_z4 = nn.Sequential(nn.Linear(t_dim, ngf * 2), nn.Sigmoid())
            self.weight_z5 = nn.Sequential(nn.Linear(t_dim, ngf), nn.Sigmoid())
            self.weight_z6 = nn.Sequential(nn.Linear(t_dim, ngf), nn.Sigmoid())

            self.t_conv0 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                                         nn.BatchNorm2d(ngf * 16),
                                         act()
                                         )
            self.t_conv1 = nn.Sequential(nn.ConvTranspose2d(ngf * 16, ngf * 16, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(ngf * 16),
                                         act()  # state size. (ngf*8) x 4 x 4
                                         )

            self.t_conv2 = nn.Sequential(nn.BatchNorm2d(ngf * 16),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False))

            self.t_conv3 = nn.Sequential(nn.BatchNorm2d(ngf * 8),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))

            self.t_conv4 = nn.Sequential(nn.BatchNorm2d(ngf * 4),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))

            self.t_conv5 = nn.Sequential(nn.BatchNorm2d(ngf * 2),
                                         act(),
                                         nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False))

            self.t_conv6 = nn.Sequential(nn.BatchNorm2d(ngf),
                                         act(),
                                         nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False))

            self.weight_z5 = nn.Sequential(nn.Linear(t_dim, ngf), nn.Sigmoid())

            self.conv = nn.Sequential(nn.Conv2d(ngf, nc, 3, padding=1), nn.Tanh())

    def forward(self, z, t=None):
        t = self.pos_enc(t) if self.pos_enc is not None else t
        if t is None:
            return self.main(z.unsqueeze(2).unsqueeze(2))

        out = (self.weight_z(t) * z).unsqueeze(2).unsqueeze(2)

        if self.img_width >= 256:
            out = self.t_conv0(out) * self.weight_z0(t).unsqueeze(2).unsqueeze(2)

        out = self.t_conv1(out) * self.weight_z1(t).unsqueeze(2).unsqueeze(2)
        out = self.t_conv2(out) * self.weight_z2(t).unsqueeze(2).unsqueeze(2)
        out = self.t_conv3(out) * self.weight_z3(t).unsqueeze(2).unsqueeze(2)

        if self.img_width >= 64:
            out = self.t_conv4(out) * self.weight_z4(t).unsqueeze(2).unsqueeze(2)
        if self.img_width >= 256:
            out = self.t_conv5(out) * self.weight_z5(t).unsqueeze(2).unsqueeze(2)
            out = self.t_conv6(out) * self.weight_z6(t).unsqueeze(2).unsqueeze(2)

        out = self.conv(out)

        return out
