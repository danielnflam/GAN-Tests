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
# Pix2Pix by Isola
####################
class Pix2Pix_Encoder_Block(nn.Module):
    """
    Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5967–5976. https://doi.org/10.1109/CVPR.2017.632
    """
    def __init__(self, _in_channels, _out_channels, _kernel_size=(4,4), _stride=(2,2), _padding=(1,1), _dilation=(1,1), _normType="BatchNorm", use_bias=True):
        super().__init__()
        self.in_channels = _in_channels
        self.out_channels = _out_channels
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding
        self.dilation_rate = _dilation
        self.normType = _normType
        # Downsampling
        self.conv2d_1 = nn.Conv2d(
                            in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation_rate,
                            bias=use_bias)
        # Norms
        if self.normType is not None:
            if self.normType == 'BatchNorm':
                self.norm = nn.BatchNorm2d(num_features=self.out_channels, affine=False)
            if self.normType == 'InstanceNorm':
                self.norm = nn.InstanceNorm2d(num_features=self.out_channels, affine=False)
        # ReLU
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2d_1(x)
        if self.normType is not None:
            out = self.norm(out)
        out = self.relu(out)
        return out

class Pix2Pix_DecoderBlock(nn.Module):
    """
    Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5967–5976. https://doi.org/10.1109/CVPR.2017.632
    """
    def __init__(self, _in_channels, _out_channels, _kernel_size=(4,4), _stride=(2,2), _padding=(1,1), _dilation=(1,1), _normType="BatchNorm", _dropoutType = "normal", _dropRate=0.5, use_bias=True):
        super().__init__()
        self.in_channels = _in_channels
        self.out_channels = _out_channels
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding
        self.dilation_rate = _dilation
        self.normType = _normType
        self.dropoutType = _dropoutType
        self.dropRate = _dropRate
        self.upsampleConv = nn.ConvTranspose2d(
                            in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation_rate,
                            bias=use_bias)
        
        # Norms
        if self.normType is not None:
            if self.normType == 'BatchNorm':
                self.norm = nn.BatchNorm2d(num_features=self.out_channels, affine=False)
            if self.normType == 'InstanceNorm':
                self.norm = nn.InstanceNorm2d(num_features=self.out_channels, affine=False)
        # ReLU
        self.relu = nn.ReLU()
        
        # Dropout
        if self.dropoutType is not None:
            if self.dropoutType == "normal":
                self.dropout = nn.Dropout(p=self.dropRate, inplace=False)
            if self.dropoutType == "ADL":
                self.dropout = blocks.ADL(drop_rate=self.dropRate, gamma=0.9)
        
    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        out = self.upsampleConv(x)
        if self.normType is not None:
            out = self.norm(out)
        if self.dropoutType is not None:
            out = self.dropout(out)
        out = torch.cat( (out, skip_x), axis=1)
        out = self.relu(out)
        return out

class Generator_Pix2Pix(nn.Module):
    def __init__(self, input_array_shape, _first_out_channels = 64, _normType="BatchNorm", _dropoutType = "normal", _dropRate=0.5, _outputType="Sigmoid"):
        super().__init__()
        self.first_out_channels = _first_out_channels
        self.input_array_shape = input_array_shape
        self.outputType = _outputType
        # Encoder
        self.enc1 = Pix2Pix_Encoder_Block( _in_channels=self.input_array_shape[1], _out_channels=self.first_out_channels, _normType=None)
        self.enc2 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels, _out_channels=self.first_out_channels*(2**1))
        self.enc3 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*(2**1), _out_channels=self.first_out_channels*(2**2))
        self.enc4 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*(2**2), _out_channels=self.first_out_channels*(2**3))
        self.enc5 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*(2**3), _out_channels=self.first_out_channels*(2**3))
        self.enc6 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*(2**3), _out_channels=self.first_out_channels*(2**3))
        self.enc7 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*(2**3), _out_channels=self.first_out_channels*(2**3))
        input_spatial = (int(self.input_array_shape[2]*(0.5**7)), int(self.input_array_shape[3]*(0.5**7)) )
        # Bridge
        #same_padding = (input_spatial[0]//2 - 1 + 4//2 , input_spatial[1]//2 - 1 + 4//2)
        self.bridge1 = nn.Conv2d(in_channels=self.first_out_channels*(2**3),
                                 out_channels=self.first_out_channels*(2**3),
                                kernel_size=(4,4),
                                stride=(2,2),
                                padding=(1,1), #same_padding,
                                dilation=(1,1),
                                bias=True)
        self.bridge2 = nn.ReLU()
        
        # Decoder.
        self.dec7 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**3), _out_channels=self.first_out_channels*(2**3))
        self.dec6 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**4), _out_channels=self.first_out_channels*(2**3))
        self.dec5 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**4), _out_channels=self.first_out_channels*(2**3))
        self.dec4 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**4), _out_channels=self.first_out_channels*(2**3), _dropoutType=None)
        self.dec3 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**4), _out_channels=self.first_out_channels*(2**2), _dropoutType=None)
        self.dec2 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**3), _out_channels=self.first_out_channels*(2**1), _dropoutType=None)
        self.dec1 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*(2**2), _out_channels=self.first_out_channels*(2**0), _dropoutType=None)
        
        # Output
        input_spatial = input_array_shape[2:4]
        #same_padding = (input_spatial[0]//2 - 1 + 4//2 , input_spatial[1]//2 - 1 + 4//2 )
        
        self.output_conv = nn.ConvTranspose2d(
                            in_channels=self.first_out_channels*(2**1),
                            out_channels=self.input_array_shape[1],
                            kernel_size=(4,4),
                            stride=(2,2),
                            padding=(1,1),
                            dilation=(1,1),
                            bias=True)
        if self.outputType == "Tanh":
            self.outImage = nn.Tanh()
        if self.outputType == "Sigmoid":
            self.outImage = nn.Sigmoid()
        
    def forward(self, x: Tensor) -> Tensor:
        # Encode
        out1 = self.enc1(x)
        out2 = self.enc2(out1)
        out3 = self.enc3(out2)
        out4 = self.enc4(out3)
        out5 = self.enc5(out4)
        out6 = self.enc6(out5)
        out7 = self.enc7(out6)
        
        #Bridge
        out = self.bridge1(out7)
        out = self.bridge2(out)
        
        # Decode
        out = self.dec7(out, out7)
        out = self.dec6(out, out6)
        out = self.dec5(out, out5)
        out = self.dec4(out, out4)
        out = self.dec3(out, out3)
        out = self.dec2(out, out2)
        out = self.dec1(out, out1)
        
        # Output
        out = self.output_conv(out)
        out = self.outImage(out)
        return out

class Discriminator_Pix2Pix(nn.Module):
    """
    The 70x70 PatchGAN from Isola et al.
    Implementation guided by code from:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L538
    
    LOGITS output -- use BCEWithLogitsLoss
    
    Paper:
    Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5967–5976. https://doi.org/10.1109/CVPR.2017.632
    """
    def __init__(self, _input_array_size, _first_out_channels=64, _normType="BatchNorm", spectral_normalize=False):
        super().__init__()
        self.input_array_size = _input_array_size
        self.first_out_channels = _first_out_channels
        self.normType = _normType
        
        if self.normType == "BatchNorm":
            normlayer = nn.BatchNorm2d
            use_bias = False
        if self.normType == "InstanceNorm":
            normlayer = nn.InstanceNorm2d
            use_bias = True
        if spectral_normalize:
            self.normalization_function = nn.utils.spectral_norm
        else:
            self.normalization_function = Identity
        
        self.conv1 = self.normalization_function(nn.Conv2d(in_channels=self.input_array_size[1],
                                out_channels=self.first_out_channels,
                                kernel_size=(4,4),
                                stride=(2,2),
                                padding=(1,1), #same_padding,
                                dilation=(1,1),
                                bias=use_bias))
        
        _out_channels2 = self.first_out_channels*2
        self.conv2 = self.normalization_function(nn.Conv2d(in_channels=self.first_out_channels,
                                out_channels=_out_channels2,
                                kernel_size=(4,4),
                                stride=(2,2),
                                padding=(1,1), #same_padding,
                                dilation=(1,1),
                                bias=use_bias))
        self.BN2 = normlayer(num_features=_out_channels2, affine=True)
        
        _out_channels3 = self.first_out_channels*(2**2)
        self.conv3 = self.normalization_function(nn.Conv2d(in_channels=_out_channels2,
                                out_channels=_out_channels3,
                                kernel_size=(4,4),
                                stride=(2,2),
                                padding=(1,1), #same_padding,
                                dilation=(1,1),
                                bias=use_bias))
        self.BN3 = normlayer(num_features=_out_channels3, affine=True)
        
        _out_channels4 = self.first_out_channels*(2**3)
        self.conv4 = self.normalization_function(nn.Conv2d(in_channels=_out_channels3,
                                out_channels=_out_channels4,
                                kernel_size=(4,4),
                                stride=1,
                                padding=(1,1), #same_padding,
                                dilation=(1,1),
                                bias=use_bias))
        self.BN4 = normlayer(num_features=_out_channels4, affine=True)
        
        # Final
        # This is from the pytorch implementation
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L538
        self.conv_final = self.normalization_function(nn.Conv2d(in_channels=_out_channels4, out_channels=1,
                                   kernel_size=4, stride=1, padding=1, bias=True))
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out = self.BN3(out)
        out = self.lrelu(out)
        out = self.conv4(out)
        out = self.BN4(out)
        out = self.lrelu(out)
        out = self.conv_final(out)
        return out
    