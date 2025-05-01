import torch
import torch.nn as nn
from ops_mod1 import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, img_feat=4, n_feats=64, kernel_size=3, num_block=16, act=nn.PReLU(), scale=4):
        super(Generator, self).__init__()
        self.scale = scale  # Store the scale factor for upscaling

        # Initial convolutional layers
        self.conv01 = conv(img_feat, n_feats, kernel_size=9, BN=False, act=act)

        # Residual blocks
        resblocks = [ResBlock(channels=n_feats, kernel_size=3, act=act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)

        self.conv02 = conv(n_feats, n_feats, kernel_size=3, BN=True, act=None)

        # Upsampling layers
        if scale == 4:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)

        # Final convolutional layer
        self.last_conv = conv(n_feats, img_feat, kernel_size=3, BN=False, act=nn.Tanh())

        # Edge and mask processing
        self.edge_mask_block = EdgeMaskBlock(img_feat, n_feats, kernel_size=3, act=act)

        # Upsampling layer for mask and edge features
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Base SR stream
        x_base = self.conv01(x)
        _skip_connection = x_base

        x_base = self.body(x_base)
        x_base = self.conv02(x_base)
        feat = x_base + _skip_connection

        x_base = self.tail(feat)
        x_base = self.last_conv(x_base)

        # Edge and mask processing
        x_edge, mask = self.edge_mask_block(x)

        # Upsample mask and edge features to match x_base's spatial dimensions
        x_edge = self.upsample(x_edge)
        mask = self.upsample(mask)

        # Combine base SR and edge features using the mask
        x_final = x_base + mask * x_edge

        return x_final, feat

class Discriminator(nn.Module):

    def __init__(self, img_feat = 4, n_feats = 64, kernel_size = 3, act = nn.LeakyReLU(inplace = True), num_of_block = 3, patch_size = 96):
        super(Discriminator, self).__init__()
        self.act = act

        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act)
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act, stride = 2)

        body = [discrim_block(in_feats = n_feats * (2 ** i), out_feats = n_feats * (2 ** (i + 1)), kernel_size = 3, act = self.act) for i in range(num_of_block)]
        self.body = nn.Sequential(*body)

        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))

        tail = []

        tail.append(nn.Linear(self.linear_size, 1024))
        tail.append(self.act)
        tail.append(nn.Linear(1024, 1))
        tail.append(nn.Sigmoid())

        self.tail = nn.Sequential(*tail)


    def forward(self, x):

        x = self.conv01(x)
        x = self.conv02(x)
        x = self.body(x)
        x = x.view(-1, self.linear_size)
        x = self.tail(x)

        return x

