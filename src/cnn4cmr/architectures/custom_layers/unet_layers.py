import torch
import torch.nn as nn


##########
## Unet ##
##########
#####################################################################
## https://arxiv.org/pdf/1505.04597.pdf                            ##
## U-Net: Convolutional Networks for Biomedical Image Segmentation ##
#####################################################################
class UnetDoubleConv(nn.Module): # 2x(conv -> b_norm -> relu)
    def __init__(self, in_ch, out_ch, d_rate=0.1, act=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
        self.act1    = act
        self.b_norm1 = nn.BatchNorm2d(out_ch)
        self.drop    = torch.nn.Dropout(p=d_rate, inplace=True)
        self.conv2   = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act2    = act
        self.b_norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.b_norm1(self.act1(self.conv1(x)))
        x = self.drop(x)
        x = self.b_norm2(self.act2(self.conv2(x)))
        return x

def test_UnetDoubleConv():
    x = torch.randn(1,3,100,100)
    model = DoubleConv(3, 32, d_rate=0.1, act=nn.LeakyReLU(negative_slope=0.01, inplace=True))
    print('Input: ', x.shape); print('Output: ', model(x).shape); print(model)
    print()
    summary(model, (3, 100, 100))


class UnetDown(nn.Module): # Downscale followed by double conv
    def __init__(self, in_ch, out_ch, d_rate, act=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super().__init__()
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d_conv = UnetDoubleConv(in_ch, out_ch, d_rate, act=nn.LeakyReLU(negative_slope=0.01, inplace=True))

    def forward(self, x):
        return self.d_conv(self.pool(x))

def test_UnetDown():
    x = torch.randn(1,3,100,100)
    model = DoubleConv(3, 32, d_rate=0.1, act=nn.LeakyReLU(negative_slope=0.01, inplace=True))
    print('Input: ', x.shape); print('Output: ', model(x).shape); print(model)
    #print()
    #summary(model, (3, 100, 100)) # dont know how for two inputs


class UnetUp(nn.Module): # Upscaling then double conv
    # describe the left_ch, lower_ch here: 
    def __init__(self, left_ch, lower_ch, out_ch, drop_r=0.1, act=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super().__init__()
        self.up     = nn.ConvTranspose2d(lower_ch, left_ch, kernel_size=2, stride=2, padding=0) # upsamples by two
        self.d_conv = UnetDoubleConv(2*left_ch, out_ch)

    def forward(self, x_left, x_lower):
        x_lower = self.up(x_lower)
        x       = torch.cat([x_left, x_lower], dim=1)
        return self.d_conv(x)