import os
import copy

import numpy as np

import torch
import torch.nn as nn

# the unet layers are sufficient to implement the FCN
from cnn4cmr.architectures.custom_layers import unet_layers as clay


############################################################
## Fully Convolutional Networks for Semantic Segmentation ##
############################################################
######################################################################
## Shelhamer E, Long J, Darrell T. Fully Convolutional Networks for ##
## Semantic Segmentation. IEEE Transactions on Pattern Analysis and ##
## Machine Intelligence. 2017 Apr;39(4):640–51.                     ##
######################################################################
class FCN(nn.Module):
    def __init__(self, net_depth, in_ch, st_ch, out_ch, d_rate=0.1, max_ch=512, act=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super().__init__()
        self.net_depth, self.in_ch, self.st_ch, self.out_ch, self.d_rate, self.max_ch = net_depth, in_ch, st_ch, out_ch, d_rate, max_ch

        self.down_layers, self.up_layers = nn.ModuleList(), nn.ModuleList()
        self.inp_conv = clay.UnetDoubleConv(in_ch, self.ch_lim(st_ch), d_rate=d_rate, act=act)

        for d in range(net_depth):
            d_inch = self.ch_lim(st_ch * 2**d); d_outch = self.ch_lim(2*d_inch) 
            self.down_layers.add_module('Down_'+str(d+1), clay.UnetDown(d_inch, d_outch, d_rate=d_rate, act=act))

        for i in range(net_depth):
            d = (net_depth-i)
            lower_ch = self.ch_lim(st_ch * 2**d)
            self.up_layers.add_module('Up_'+str(i+1), nn.ConvTranspose2d(lower_ch, st_ch, kernel_size=2**d, stride=2**d, padding=0))

        self.concat_conv = clay.UnetDoubleConv(st_ch*net_depth, self.ch_lim(st_ch), d_rate=d_rate, act=act)
        
        self.logits_conv = nn.Conv2d(in_channels=st_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.out = nn.Sigmoid()

    def ch_lim(self, ch):
        return ch if self.max_ch is None else min(ch, self.max_ch)

    def forward(self, x):
        x = self.inp_conv(x)
        layers = [x]
        for d, l in enumerate(self.down_layers): x = l(x); layers.append(x)
        for d, l in enumerate(self.up_layers  ): x = l(layers[-2*d-1]); layers.append(x)
        x = torch.cat(layers[-self.net_depth:], dim=1)
        x = self.concat_conv(x)
        x = self.logits_conv(x)
        return self.out(x)

    def init_weights(self): # might need adjustments, but kaiming is pretty robust
        def init_weights_inner(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: m.bias.data.fill_(0.01)
        self.apply(init_weights_inner)

