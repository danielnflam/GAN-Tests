import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as vtransforms
from typing import Type, Any, Callable, Union, List, Optional
import blocks
####################
# Utility Functions
####################
def Identity(x):
    return x

####################
# Model from the paper
# Zhou, Z., Zhou, L., & Shen, K. (2020). Dilated conditional GAN for bone suppression in chest radiographs with enforced semantic features. Medical Physics, 47(12), 6207–6215. https://doi.org/https://doi.org/10.1002/mp.14371
####################

class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, normType="BatchNorm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (4,4)
        self.stride = (2,2)
        self.use_bias = use_bias
        self.normType = normType
        
        # components
        self.Conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                     padding=1, dilation=1, groups=1, bias=self.use_bias, padding_mode='zeros')
        if self.normType == "BatchNorm":
            self.norm = nn.BatchNorm2d(self.out_channels, affine=False)
        if self.normType == "InstanceNorm":
            self.norm = nn.InstanceNorm(self.out_channels, affine=False)
        
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        out = self.Conv(x)
        out = self.norm(out)
        out = self.lrelu(out)
        return out
    
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_bias=True, normType="BatchNorm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (4,4)
        self.stride = (1,1)
        self.dilation = dilation
        self.use_bias = use_bias
        self.normType = normType
        
        # components
        self.dilatedConv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                     padding=1, dilation=self.dilation, groups=1, bias=self.use_bias, padding_mode='zeros')
        if self.normType == "BatchNorm":
            self.norm = nn.BatchNorm2d(self.out_channels, affine=False)
        if self.normType == "InstanceNorm":
            self.norm = nn.InstanceNorm(self.out_channels, affine=False)
        
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        out = self.dilatedConv(x)
        out = self.norm(out)
        out = self.lrelu(out)
        return out

class StandardDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, dropoutType="normal", normType = "BatchNorm", reluType="normal"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size=(4,4)
        self.stride=(2,2)
        self.use_bias = use_bias
        self.normType = normType
        self.dropoutType = dropoutType
        self.reluType=reluType
        
        # Components
        self.dilatedDeconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 
                                                padding=1, output_padding=0, groups=1, bias=self.use_bias, dilation=1, padding_mode='zeros')
        
        if self.normType == "BatchNorm":
            self.norm = nn.BatchNorm2d(self.out_channels, affine=False)
        if self.normType == "InstanceNorm":
            self.norm = nn.InstanceNorm(self.out_channels, affine=False)
            
        if self.dropoutType == "normal":
            self.dropout = nn.Dropout(p=0.5)
        if self.dropoutType == "ADL":
            self.dropout = blocks.ADL(drop_rate=0.5, gamma=0.9)
        
        if self.reluType=="leaky":
            self.lrelu = nn.LeakyReLU(0.2)
        if self.reluType=="normal":
            self.lrelu = nn.ReLU()
        
        
    def forward(self, x):
        out = self.dilatedDeconv(x)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.lrelu(out)
        return out
class DilatedDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation , use_bias=True, dropoutType="normal", normType = "BatchNorm", reluType="normal"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size=(4,4)
        self.stride=(1,1)
        self.use_bias = use_bias
        self.normType = normType
        self.dropoutType = dropoutType
        self.reluType = reluType
        # Components
        self.dilatedDeconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 
                                                padding=1, output_padding=0, groups=1, bias=self.use_bias, dilation=self.dilation, padding_mode='zeros')
        
        if self.normType == "BatchNorm":
            self.norm = nn.BatchNorm2d(self.out_channels, affine=False)
        if self.normType == "InstanceNorm":
            self.norm = nn.InstanceNorm(self.out_channels, affine=False)
            
        if self.dropoutType == "normal":
            self.dropout = nn.Dropout(p=0.5)
        if self.dropoutType == "ADL":
            self.dropout = blocks.ADL(drop_rate=0.5, gamma=0.9)
        
        if self.reluType=="leaky":
            self.lrelu = nn.LeakyReLU(0.2)
        if self.reluType=="normal":
            self.lrelu = nn.ReLU()
        
    def forward(self, x):
        out = self.dilatedDeconv(x)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.lrelu(out)
        return out

class Generator(nn.Module):
    def __init__(self, input_array_shape, initial_channels_out=64, use_bias=True, normType="BatchNorm", dropoutType="normal", reluType="normal"):
        super().__init__()
        self.input_array_shape = input_array_shape
        self.initial_channels_out=initial_channels_out
        self.use_bias = use_bias
        self.normType = normType
        self.dropoutType = dropoutType
        self.reluType = reluType
        
        print(self.input_array_shape[1])
        self.conv1 = StandardConvBlock( in_channels=self.input_array_shape[1], out_channels=self.initial_channels_out*(2**0), use_bias=self.use_bias, normType=self.normType)
        self.dconv2 = DilatedConvBlock( in_channels=self.initial_channels_out*(2**0), out_channels=self.initial_channels_out*(2**1), dilation=2, use_bias=self.use_bias, normType=self.normType)
        self.dconv3 = DilatedConvBlock( in_channels=self.initial_channels_out*(2**1), out_channels=self.initial_channels_out*(2**2), dilation=4, use_bias=self.use_bias, normType=self.normType)
        self.dconv4 = DilatedConvBlock( in_channels=self.initial_channels_out*(2**2), out_channels=self.initial_channels_out*(2**3), dilation=8, use_bias=self.use_bias, normType=self.normType)
        self.dconv5 = DilatedConvBlock( in_channels=self.initial_channels_out*(2**3), out_channels=self.initial_channels_out*(2**3), dilation=16, use_bias=self.use_bias, normType=self.normType)
        self.dconv6 = DilatedConvBlock( in_channels=self.initial_channels_out*(2**3), out_channels=self.initial_channels_out*(2**3), dilation=32, use_bias=self.use_bias, normType=self.normType)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.dconv2(out1)
        out3 = self.dconv3(out2)
        out4 = self.dconv4(out3)
        out5 = self.dconv5(out4)
        out6 = self.dconv6(out5)
        return out6
        