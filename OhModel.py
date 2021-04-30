import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as vtransforms
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
    def __init__(self, in_channels, out_channels, use_bias=False, reluType="normal"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.reluType = reluType
        
        self.conv1_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                             kernel_size=1, stride=2, padding=0, bias=self.use_bias)
        
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=4, stride=2, padding = 1, bias=self.use_bias)
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
    def __init__(self, in_channels, out_channels, use_bias=False, reluType="normal"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.reluType = reluType
        
        self.conv1_1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, 
                             kernel_size=1, stride=2, padding=0, bias=self.use_bias)
        
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 
                                 kernel_size=3, stride=2, padding = 1, bias=self.use_bias)
        self.BN1 = nn.BatchNorm2d(self.out_channels)
        if self.reluType == "normal":
            self.relu = nn.ReLU()
        if self.reluType == "leaky":
            self.relu = nn.LeakyReLU(0.2)
        
        self.conv3_2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3,
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