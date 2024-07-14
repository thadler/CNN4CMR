import os
import copy

import numpy as np

import torch
import torch.nn as nn

from cnn4cmr.architectures.custom_layers import unet_layers as clay


##################
## Vanilla Unet ## # # SEMANTIC SEGMENTATION
##################
#####################################################################
## https://arxiv.org/pdf/1505.04597.pdf                            ##
## U-Net: Convolutional Networks for Biomedical Image Segmentation ##
#####################################################################

# good default params
#    return cls({'net_depth':        7,     'in_ch':  1, 
#                'st_ch':            16,    'out_ch': 3, 
#                'd_rate':           0.1,   'max_ch': 256,
#                'deep_supervision': True})
#


class Unet(nn.Module):
    def __init__(self, net_depth, in_ch, st_ch, out_ch, d_rate=0.1, max_ch=512, deep_supervision=False, act=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super().__init__()
        self.net_depth, self.in_ch, self.st_ch, self.out_ch, self.d_rate, self.max_ch, self.deep_supervision = net_depth, in_ch, st_ch, out_ch, d_rate, max_ch, deep_supervision
        
        self.down_layers, self.up_layers = nn.ModuleList(), nn.ModuleList()
        
        self.inp_conv = clay.UnetDoubleConv(in_ch, self.ch_lim(st_ch), d_rate=d_rate, act=act)
        for d in range(net_depth):
            d_inch = self.ch_lim(st_ch * 2**d); d_outch = self.ch_lim(2*d_inch) # print('do ch: ', d, d_inch, out_ch)
            self.down_layers.add_module('Down_'+str(d+1), clay.UnetDown(d_inch, d_outch, d_rate=d_rate, act=act))
        for i in range(net_depth):
            d = (net_depth-i)
            left_ch = self.ch_lim(st_ch * 2**(d-1)); lower_ch = self.ch_lim(st_ch * 2**d) # print('up ch: ', i+1, left_ch, lower_ch)
            self.up_layers.add_module('Up_'+str(i+1), clay.UnetUp(left_ch, lower_ch, left_ch, drop_r=d_rate, act=act))
        # this conv is necessary as it follows the batch_norm
        self.logits_conv = nn.Conv2d(in_channels=st_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.out = nn.Sigmoid()

        # deep supervision outputs for levels 2 & 3
        self.logits_conv2 = nn.Conv2d(in_channels=st_ch*2, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.logits_conv3 = nn.Conv2d(in_channels=st_ch*4, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def ch_lim(self, ch):
        return ch if self.max_ch is None else min(ch, self.max_ch)

    def forward(self, x):
        x = self.inp_conv(x)
        layers = [x]
        for d, l in enumerate(self.down_layers): x = l(x); layers.append(x)
        for d, l in enumerate(self.up_layers  ): x = l(layers[-2*(d+1)], layers[-1]); layers.append(x)
        x = self.logits_conv(x)
        if self.deep_supervision: 
            return self.out(x), self.out(self.logits_conv2(layers[-2])), self.out(self.logits_conv3(layers[-3]))
        return self.out(x)

    def init_weights(self): # might need adjustments, but kaiming is pretty robust
        def init_weights_inner(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: m.bias.data.fill_(0.01)
        self.apply(init_weights_inner)

    def save_checkpoint(self, path, optimizer, parent_path, epoch, evaluation, augmentations):
        model_params = {'net_depth': self.net_depth, 'in_ch': self.in_ch, 'st_ch': self.st_ch, 
                        'out_ch':    self.out_ch,     'd_rate': self.d_rate, 'max_ch': self.max_ch, 
                        'd_sv': self.deep_supervision}
        model_name = 'Unet_Epoch_'+str(epoch)+'_'
        for k in model_params: model_name += k+'_'+str(model_params[k])+'_'
        model_name += '.pth'
        checkpoint_path = os.path.join(path, model_name)
        torch.save({'parent_path':          parent_path,
                    'epoch':                epoch+1,
                    'model_state_dict':     copy.deepcopy(self.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'model_config':         model_params,
                    'evaluation':           evaluation,
                    'augmentations':        augmentations}, 
                    os.path.join(path, model_name))