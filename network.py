import torch
import torch.nn as nn
import numpy as np

import HyperParams as Params

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                Params.LatentVectorSize, Params.Ngf * 8,
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d( Params.Ngf * 8 ),
            nn.ReLU( True ),

            nn.ConvTranspose2d(
                Params.Ngf * 8, Params.Ngf * 4,
                kernel_size=4, stride = 2, padding=1, bias=False
            ),
            nn.BatchNorm2d( Params.Ngf * 4 ),
            nn.ReLU( True ),

            nn.ConvTranspose2d(
                Params.Ngf * 4, Params.Ngf * 2,
                kernel_size=4, stride = 2, padding=1, bias=False
            ),
            nn.BatchNorm2d( Params.Ngf * 2 ),
            nn.ReLU( True ),

            nn.ConvTranspose2d(
                Params.Ngf * 2, Params.Ngf,
                kernel_size=4, stride = 2, padding=1, bias=False
            ),
            nn.BatchNorm2d( Params.Ngf ),
            nn.ReLU( True ),

            nn.ConvTranspose2d(
                Params.Ngf, 3,
                kernel_size=4, stride = 2, padding=1, bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.reshape(x, (batch_size, Params.LatentVectorSize,1,1))

        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, Params.Ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.Ndf, Params.Ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.Ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.Ndf * 2, Params.Ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.Ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.Ndf * 4, Params.Ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.Ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.Ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.reshape(x, (batch_size, 3, Params.Ndf, Params.Ndf))
        return self.layers(x)

    def clip_weights(self):
        self.weights = torch.clamp(self.weights, -1., 1.)
