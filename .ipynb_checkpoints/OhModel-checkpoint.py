import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as vtransforms
from torch.autograd import Variable

from typing import Type, Any, Callable, Union, List, Optional
import blocks
from torch.nn import init
import functools

#####################
# Custom Generator and Discriminator for Oh et al.'s model
# Oh and Yun. 2018. Oh, D. Y., & Yun, I. D. (2018). Learning Bone Suppression from Dual Energy Chest X-rays using Adversarial Networks. https://arxiv.org/abs/1811.02628
#####################
# Residual Block:
class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, reluType="normal"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.reluType = reluType
        
        self.conv1_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                             kernel_size=1, stride=2, padding=0, bias=self.use_bias)
        
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=3, stride=2, padding = 1, bias=self.use_bias)
        self.BN1 = nn.BatchNorm2d(self.out_channels)
        if self.reluType == "normal":
            self.relu = nn.ReLU()
        if self.reluType == "leaky":
            self.relu = nn.LeakyReLU(0.2)
        
        self.conv3_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
                             stride=1, padding=1, bias=self.use_bias)
        self.BN2 = nn.BatchNorm2d(self.out_channels)
    def forward(self, x):
        out_skip = self.conv1_1(x)
        out = self.conv3_1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        
        out = out + out_skip
        out = self.BN2(out)
        out = self.relu(out)
        return out

class DeconvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, reluType="normal"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.reluType = reluType
        self.convTranspose2d_padding = 1
        
        #self.conv1_1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, 
        #                     kernel_size=1, stride=2, padding=0, output_padding=1, bias=self.use_bias)
        self.conv1_1 = blocks.UpsampleConvolution(in_channels=self.in_channels, out_channels=self.out_channels, upsample_scale_factor=2, kernel_size=1, stride=1, padding=0, bias=self.use_bias)
        #self.conv3_1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, 
        #                         kernel_size=3, stride=2, padding = 1, output_padding=1, bias=self.use_bias)
        self.conv3_1 = blocks.UpsampleConvolution(in_channels=self.in_channels, out_channels=self.out_channels, upsample_scale_factor=2, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.BN1 = nn.BatchNorm2d(self.out_channels)
        if self.reluType == "normal":
            self.relu = nn.ReLU()
        if self.reluType == "leaky":
            self.relu = nn.LeakyReLU(0.2)
        
        self.conv3_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
                             stride=1, padding=1, bias=self.use_bias)
        self.BN2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        # Skip is correct
        out_skip = self.conv1_1(x)
        
        out = self.conv3_1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        
        out = out + out_skip
        out = self.BN2(out)
        out = self.relu(out)
        return out

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction_ratio=16, use_bias=True, reluType="normal"):
        super().__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.reduction_ratio = reduction_ratio
        self.use_bias = use_bias
        self.reluType = reluType
        
        # RESIDUAL BLOCK
        if self.reluType == "normal":
            self.relu = nn.ReLU()
        if self.reluType == "leaky":
            self.relu = nn.LeakyReLU(0.2)
        
        self.conv_skip = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(self.out_channels, self.out_channels)
        self.conv3_2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(self.out_channels, self.out_channels)
        
        self.linear1 = nn.Linear(in_features=self.out_channels, out_features=self.out_channels//self.reduction_ratio)
        self.linear2 = nn.Linear(in_features=self.out_channels//self.reduction_ratio, out_features=self.out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Residual Block
        out_skip = self.conv_skip(x)
        out = self.conv3_1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = out + out_skip
        out = self.norm2(out)
        out_after_residual = self.relu(out)
        
        # SQUEEZE + EXCITATION
        out = torch.mean(out_after_residual,[2,3]) # global average pooling
        
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        
        out = out.unsqueeze(-1).unsqueeze(-1)
        
        out = out*out_after_residual
        return out
        
    
class Generator(nn.Module):
    def __init__(self, input_array_shape, reluType="normal", use_bias=True):
        super().__init__()
        self.input_array_shape = input_array_shape
        self.num_ini_filters = 64
        self.reluType = reluType
        self.use_bias = use_bias
        
        in_channels = input_array_shape[1]
        self.encblk1 = ConvResBlock(in_channels, self.num_ini_filters, use_bias=self.use_bias, reluType=self.reluType)
        self.encblk2 = ConvResBlock(self.num_ini_filters, self.num_ini_filters, use_bias=self.use_bias, reluType=self.reluType)
        self.encblk3 = ConvResBlock(self.num_ini_filters, self.num_ini_filters*2, use_bias=self.use_bias, reluType=self.reluType)
        self.encblk4 = ConvResBlock(self.num_ini_filters*2, self.num_ini_filters*2, use_bias=self.use_bias, reluType=self.reluType)
        self.encblk5 = ConvResBlock(self.num_ini_filters*2, self.num_ini_filters*4, use_bias=self.use_bias, reluType=self.reluType)
        self.encblk6 = ConvResBlock(self.num_ini_filters*4, 320, use_bias=self.use_bias, reluType=self.reluType)
        
        self.SQblk = SqueezeExcitationBlock(320, 320,
                                            reduction_ratio=16, use_bias=self.use_bias, reluType=self.reluType)
        self.decblk6 = DeconvResBlock(320, self.num_ini_filters*4, use_bias=self.use_bias, reluType=self.reluType)
        self.decblk5 = DeconvResBlock(self.num_ini_filters*4, self.num_ini_filters*2, use_bias=self.use_bias, reluType=self.reluType)
        self.decblk4 = DeconvResBlock(self.num_ini_filters*2, self.num_ini_filters*2, use_bias=self.use_bias, reluType=self.reluType)
        self.decblk3 = DeconvResBlock(self.num_ini_filters*2, self.num_ini_filters*1, use_bias=self.use_bias, reluType=self.reluType)
        self.decblk2 = DeconvResBlock(self.num_ini_filters*1, self.num_ini_filters*1, use_bias=self.use_bias, reluType=self.reluType)
        self.decblk1 = DeconvResBlock(self.num_ini_filters*1, in_channels, use_bias=self.use_bias, reluType=self.reluType)
        
        print("Oh Model Generator thought to use summation skip connection.")
        
    def forward(self, x):
        out1 = self.encblk1(x)
        out2 = self.encblk2(out1)
        out3 = self.encblk3(out2)
        out4 = self.encblk4(out3)
        out5 = self.encblk5(out4)
        
        out6 = self.encblk6(out5)
        
        out = self.SQblk(out6)
        #print(out.shape)
        out = self.decblk6(out)
        
        # Using Summation Skip Connection
        out = out + out5
        out = self.decblk5(out)
        out = out + out4
        out = self.decblk4(out)
        out = out + out3
        out = self.decblk3(out)
        out = out + out2
        out = self.decblk2(out)
        out = out + out1
        out = self.decblk1(out)
        return out

class Discriminator_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_bias=True, reluType="normal"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 2
        self.padding = padding
        self.use_bias = use_bias
        self.reluType = reluType
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.use_bias)
        self.norm = nn.BatchNorm2d(self.out_channels)
        if self.reluType == "normal":
            self.relu = nn.ReLU()
        if self.reluType == "leaky":
            self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_array_shape, num_kernels, kernel_dims, use_bias = True, reluType="normal"):
        super().__init__()
        
        self.input_array_shape = input_array_shape
        self.initial_out_channel = 32
        self.use_bias = use_bias
        self.reluType = reluType
        self.num_kernels = num_kernels
        self.kernel_dims = kernel_dims
        
        in_c = self.input_array_shape[1]
        out_c = self.initial_out_channel
        self.encodeBlk1 = Discriminator_ConvBlock(in_c, out_c, 3, 1, self.use_bias, self.reluType)
        self.encodeBlk2 = Discriminator_ConvBlock(out_c, out_c, 3, 1, self.use_bias, self.reluType)
        self.encodeBlk3 = Discriminator_ConvBlock(out_c, out_c*2, 3, 1, self.use_bias, self.reluType)
        self.encodeBlk4 = Discriminator_ConvBlock(out_c*2, out_c*2, 3, 1, self.use_bias, self.reluType)
        self.encodeBlk5 = Discriminator_ConvBlock(out_c*2, out_c*4, 3, 1, self.use_bias, self.reluType)
        self.encodeBlk6 = Discriminator_ConvBlock(out_c*4, out_c*4, 3, 1, self.use_bias, self.reluType)
        self.encodeBlk7 = Discriminator_ConvBlock(out_c*4, out_c*8, 3, 1, self.use_bias, self.reluType)
        
        # Flatten array
        array_size = [self.input_array_shape[0], out_c*8, input_array_shape[2]//128, input_array_shape[3]//128]
        self.flatten = nn.Flatten()
        feature_num = array_size[1]*array_size[2]*array_size[3]
        
        self.miniBatchDisc = blocks.MiniBatchDiscrimination(feature_num, self.num_kernels, self.kernel_dims, mean=False)
        
        self.fc = nn.Linear(feature_num + self.num_kernels, 1)
        
    def forward(self, x):
        out = self.encodeBlk1(x)
        out = self.encodeBlk2(out)
        out = self.encodeBlk3(out)
        out = self.encodeBlk4(out)
        out = self.encodeBlk5(out)
        out = self.encodeBlk6(out)
        out = self.encodeBlk7(out)
        out = self.flatten(out)
        
        out = self.miniBatchDisc(out)
        out = self.fc(out)
        return out
        

        
        