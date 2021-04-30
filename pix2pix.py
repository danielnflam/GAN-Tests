import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as vtransforms
from typing import Type, Any, Callable, Union, List, Optional
import blocks
from torch.nn import init
import functools


####################
# Utility Functions
####################
def Identity(x):
    return x

####################
# Pix2Pix by Isola
####################

class UnetGenerator(nn.Module):
    """
    Create a Unet-based generator.
    Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L436
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer [nn.BatchNorm2d, nn.InstanceNorm2d]
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
        
#####################
# Rewritten for better interpretability
#####################
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
                self.norm = nn.BatchNorm2d(num_features=self.out_channels, affine=True)
            if self.normType == 'InstanceNorm':
                self.norm = nn.InstanceNorm2d(num_features=self.out_channels, affine=True)
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
    def __init__(self, _in_channels, _out_channels, _kernel_size=(4,4), _stride=(2,2), _padding=(1,1), _dilation=(1,1), _normType="BatchNorm", use_bias=True, _dropoutType = "normal", _dropRate=0.5):
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
                self.norm = nn.BatchNorm2d(num_features=self.out_channels, affine=True)
            if self.normType == 'InstanceNorm':
                self.norm = nn.InstanceNorm2d(num_features=self.out_channels, affine=True)
        # ReLU
        self.relu = nn.ReLU()
        
        # Dropout
        if self.dropoutType is not None:
            if self.dropoutType == "normal":
                self.dropout = nn.Dropout(p=self.dropRate, inplace=False)
            if self.dropoutType == "ADL":
                self.dropout = blocks.ADL(drop_rate=self.dropRate, gamma=0.9)
        
    def forward(self, x: Tensor, skip_tensor: Tensor) -> Tensor:
        
        out = self.upsampleConv(x)
        #print("after upsample: " + str(out.shape))
        if self.normType is not None:
            out = self.norm(out)
        if self.dropoutType is not None:
            out = self.dropout(out)
        out = torch.cat((out, skip_tensor), 1)
        #print("after cat: " + str(out.shape))
        out = self.relu(out)
        return out

    
class Custom_Written_Generator(nn.Module):
    def __init__(self, input_array_shape, _first_out_channels = 64, _normType="BatchNorm", _dropoutType = "normal", _dropRate=0.5, _outputType="Tanh"):
        super().__init__()
        self.first_out_channels = _first_out_channels
        self.input_array_shape = input_array_shape
        self.outputType = _outputType
        self.normType = _normType
        if self.normType == "BatchNorm":
            _use_bias = False
        if self.normType == "InstanceNorm":
            _use_bias = True
        
        # INPUT
        self.convInput = nn.Conv2d(in_channels=self.input_array_shape[1], out_channels=self.first_out_channels,
                            kernel_size=4, stride=2, padding=1, dilation=1,
                            bias=_use_bias)
        self.lrelu = nn.LeakyReLU(0.2)
        # ENCODER
        self.enc1 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels, _out_channels=self.first_out_channels*2, _normType=self.normType, use_bias=_use_bias)
        self.enc2 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*2, _out_channels=self.first_out_channels*4, _normType=self.normType, use_bias=_use_bias)
        self.enc3 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*4, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        self.enc4 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*8, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        self.enc5 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*8, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        self.enc6 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*8, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        self.enc7 = Pix2Pix_Encoder_Block( _in_channels=self.first_out_channels*8, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        input_spatial = (int(self.input_array_shape[2]*(0.5**7)), int(self.input_array_shape[3]*(0.5**7)) )
        # Bridge
        #same_padding = (input_spatial[0]//2 - 1 + 4//2 , input_spatial[1]//2 - 1 + 4//2)
        self.bridgeConv = nn.Conv2d(in_channels=self.first_out_channels*8,
                                 out_channels=self.first_out_channels*8,
                                kernel_size=4,
                                stride=2,
                                padding=1, #same_padding,
                                dilation=1,
                                bias=_use_bias)
        self.bridgeRelu = nn.ReLU()
        
        # Decoder.
        self.dec7 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*8, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        
        self.dec6 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*16, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        self.dec5 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*16, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias)
        self.dec4 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*16, _out_channels=self.first_out_channels*8, _normType=self.normType, use_bias=_use_bias, _dropoutType=None)
        self.dec3 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*16, _out_channels=self.first_out_channels*4, _normType=self.normType, use_bias=_use_bias, _dropoutType=None)
        self.dec2 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*8, _out_channels=self.first_out_channels*2, _normType=self.normType, use_bias=_use_bias, _dropoutType=None)
        self.dec1 = Pix2Pix_DecoderBlock( _in_channels=self.first_out_channels*4, _out_channels=self.first_out_channels, _normType=self.normType, use_bias=_use_bias, _dropoutType=None)
        
        # Output
        input_spatial = input_array_shape[2:4]
        #same_padding = (input_spatial[0]//2 - 1 + 4//2 , input_spatial[1]//2 - 1 + 4//2 )
        
        self.output_conv = nn.ConvTranspose2d(
                            in_channels=self.first_out_channels*2,
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
        out = self.convInput(x)
        skip1 = self.lrelu(out)
        skip2 = self.enc1(skip1)
        skip3 = self.enc2(skip2)
        skip4 = self.enc3(skip3)
        skip5 = self.enc4(skip4)
        skip6 = self.enc5(skip5)
        skip7 = self.enc6(skip6)
        
        # Bridge
        out = self.bridgeConv(skip7)
        out = self.bridgeRelu(out)
        
        # Decode
        out = self.dec7(out, skip7)
        out = self.dec6(out, skip6)
        out = self.dec5(out, skip5)
        out = self.dec4(out, skip4)
        out = self.dec3(out, skip3)
        out = self.dec2(out, skip2)
        out = self.dec1(out, skip1)
        
        # Output
        out = self.output_conv(out)
        out = self.outImage(out)
        
        return out
""""""

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
    