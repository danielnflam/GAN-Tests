# IMPLEMENT THE RESNET
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torchvision.transforms as vtransforms
from typing import Type, Any, Callable, Union, List, Optional
import os, random, sys, time, pathlib
from operator import add

def Identity(x):
    return x

class MiniBatchDiscrimination(nn.Module):
    """
    source: https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491
    paper: Salimans et al. 2016. Improved Methods for Training GANs
    """
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims), requires_grad=True) #  Tensor that is to be considered a module parameter.
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1) # concatenate output with the input features
        return x

class ImageBuffer():
    """
    Attain a significant performance improvement by using a saved buffer of previously generated images.
    1) Take k-generated samples from the mini-batch in the i'th generation.
    2) Randomly shuffle data in the buffer
    3) Pop and concatenate the batch with the buffer sample
    
    """
    def __init__(self, pool_size, buffer_out_rate=0.5):
        """
        Inputs:
            pool_size, buffer_out_rate=0.5
        """
        self.poolSize = pool_size
        if self.poolSize > 0:  # create an empty pool
            self.numImagesInPoolCurrent = 0
            self.buffer_out_rate = buffer_out_rate
            self.images = []
    def query(self, images):
        """
        Outputs an image from the image buffer by chance (self.buffer_out_rate).
        Inputs:
            images: a [N x C x H x W] torch tensor.
        """
        
        # If image pool is size 0, identity function
        if self.poolSize == 0:
            return images
        
        # Else: build a list of tensors for the buffer
        return_images = []
        for image in images:
            # image.shape has [C x H x W]
            image = torch.unsqueeze(image.data, 0) # image.data is assumed to have dimensions [C x H x W]
            if self.numImagesInPoolCurrent < self.poolSize:   # if the buffer is not full; keep inserting current images to the buffer
                self.numImagesInPoolCurrent += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p < self.buffer_out_rate:  
                    # if buffer_out_rate = 0.8, 80% of the time the input image is swapped for a buffer-held image
                    random_id = random.randint(0, self.poolSize - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
        

    

class ADL(nn.Module):
    """
    From the work done by Choe et al. 2019.
    Inputs for initialisation: 
    drop_rate: the proportion at which the drop mask is selected instead of the importance map.  Default: 0.75 (i.e. use the drop mask 75% of the time).
    gamma: the ratio of maximum intensity of the self-attention map, at which the threshold is set to determine the importance map/drop mask.  Default: 0.9 (should be set depending on network)
    
    Source:
    Choe, J., & Shim, H. (2019). Attention-Based Dropout Layer for Weakly Supervised Object Localization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2219–2228.
    Source: https://github.com/clovaai/wsolevaluation
    """
    def __init__(self, drop_rate=0.75, gamma=0.9):
        super().__init__()
        self.drop_rate = drop_rate
        self.gamma = gamma
    
    def forward(self, x: Tensor) -> Tensor:
        """
        x has units NxCxHxW
        """
        # Evaluation mode:
        if not self.training:
            return x
        # Training mode:
        attention_map = torch.mean(x, dim=1, keepdim=True)
        drop_mask = self.calculate_drop_mask(attention_map)
        importance_map = torch.sigmoid(attention_map)
        selected_map = self.select_map(drop_mask, importance_map)
        
        return torch.mul(selected_map, x)
    
    def select_map(self, drop_mask, importance_map) -> Tensor:
        randNumber = torch.rand([], dtype=torch.float32) + self.gamma
        binaryNum = randNumber.floor()
        return (1.-binaryNum)*importance_map + binaryNum*drop_mask
    
    def calculate_drop_mask(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        maxAtt, _ = torch.max(x.view(batch_size,-1), dim=1, keepdim=True)  # maxAtt calculated for each batch individually.
        threshold = self.gamma * maxAtt
        threshold = threshold.view(batch_size,1,1,1) # reshape into NxCxHxW
        drop_mask = (x < threshold).float()
        return drop_mask
    
    def extra_repr(self):
        """Information Function"""
        return "ADL Drop Rate={}, ADL Gamma={}".format(
            self.drop_rate, self.gamma)

class UpsampleConvolution(nn.Module):
    """
    Instead of using ConvTranspose2d, which may lead to weird checkerboard artifacts, use a separate upsample and conv2d 1x1 operation.
    Here is a generic combination of the two.
    """
    def __init__(self, in_channels, out_channels, upsample_size=None, upsample_scale_factor=None, upsample_mode="nearest", upsample_align_corners=False,
                kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        # upsample
        if upsample_size is not None:
            if kernel_size > 1:
                a = upsample_size
                b = (2*(kernel_size//2) , 2*(kernel_size//2))
            self.size=list(map(add, a, b))
        else:
            self.size = None
        self.scale_factor=upsample_scale_factor
        self.mode=upsample_mode
        self.align_corners=upsample_align_corners
        # conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation=dilation
        self.groups=groups
        self.bias=bias
        self.padding_mode=padding_mode
        
        # blocks
        if self.mode == "nearest":
            #print("Upsampling Mode: " + str(self.mode))
            self.align_corners = None
        
        self.upsample = nn.Upsample(size=self.size, scale_factor = self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.padding_mode)
    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample(x)
        out = self.conv(out)
        return out

######################################
# Diakogiannis ResUNet-A
# Not in use
######################################
    
class PSPPooling_miniBlock(nn.Module):
    """
    PSP Pooling miniBlock.
    
    Source of architecture:
    Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020). ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. 
    ISPRS Journal of Photogrammetry and Remote Sensing, 162, 94–114. https://doi.org/10.1016/j.isprsjprs.2020.01.013
    
    Source of original paper:
    Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid Scene Parsing Network. 
    2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 6230–6239. https://doi.org/10.1109/CVPR.2017.660
    """
    def __init__(self, _in_channels, _output_size, _kernel_size, _stride, _padding, _dilation, _pyramid_levels):
        super().__init__()
        
        self.in_channels = _in_channels
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding
        self.dilation = _dilation
        self.output_size = _output_size
        
        self.maxPool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,padding=self.padding, dilation=self.dilation)
        self.upSample = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=None)
        self.dimensionalReduction = Conv2DN(_in_channels = self.in_channels, _out_channels = self.in_channels//_pyramid_levels, _kernel_size=(1,1), _stride=(1, 1), _padding=(0,0), _dilation_rate=(1,1), _norm_type='BatchNorm')
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.maxPool(x)
        out = self.upSample(out)
        out = self.dimensionalReduction(out)
        return out

class PSPPooling(nn.Module):
    """
    PSP Pooling.
    On forward step:
    INPUT and OUTPUT tensors are the same size.
    
    
    INPUT when initialising class:
    _tensor_array_shape : (N, C, H, W)
    
    Source of architecture:
    Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020). ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. 
    ISPRS Journal of Photogrammetry and Remote Sensing, 162, 94–114. https://doi.org/10.1016/j.isprsjprs.2020.01.013
    
    Source of original paper:
    Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid Scene Parsing Network. 
    2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 6230–6239. https://doi.org/10.1109/CVPR.2017.660
    
    
    """
    def __init__(self, _tensor_array_shape):
        super().__init__()
        
        self.in_channels = _tensor_array_shape[1]
        self.output_size = (_tensor_array_shape[2] , _tensor_array_shape[3])
        self.pyramid_levels = 4
        
        self.miniBlock_out_channels = (self.in_channels//self.pyramid_levels)
        self.concat_channels = self.miniBlock_out_channels*self.pyramid_levels
        
        kernel_sizes = ((_tensor_array_shape[2] , _tensor_array_shape[3]) ,
                        (_tensor_array_shape[2]//2 , _tensor_array_shape[3]//2),
                        (_tensor_array_shape[2]//4 , _tensor_array_shape[3]//4),
                        (_tensor_array_shape[2]//8 , _tensor_array_shape[3]//8))
        
        strides=((1,1),
                 kernel_sizes[1],
                 kernel_sizes[2],
                 kernel_sizes[3])
        
        paddings=((0),(0),(0),(0))
        dilations=((1),(1),(1),(1))
        
        self.pooling_1 = PSPPooling_miniBlock(_in_channels=self.in_channels, _output_size=self.output_size, _kernel_size=kernel_sizes[0], _stride=strides[0], _padding=paddings[0], _dilation=dilations[0], _pyramid_levels=self.pyramid_levels)
        self.pooling_2 = PSPPooling_miniBlock(_in_channels=self.in_channels, _output_size=self.output_size, _kernel_size=kernel_sizes[1], _stride=strides[1], _padding=paddings[1], _dilation=dilations[1], _pyramid_levels=self.pyramid_levels)
        self.pooling_3 = PSPPooling_miniBlock(_in_channels=self.in_channels, _output_size=self.output_size, _kernel_size=kernel_sizes[2], _stride=strides[2], _padding=paddings[2], _dilation=dilations[2], _pyramid_levels=self.pyramid_levels)
        self.pooling_4 = PSPPooling_miniBlock(_in_channels=self.in_channels, _output_size=self.output_size, _kernel_size=kernel_sizes[3], _stride=strides[3], _padding=paddings[3], _dilation=dilations[3], _pyramid_levels=self.pyramid_levels)
        
        self.finalConv2DN = Conv2DN(_in_channels = self.concat_channels, _out_channels = self.in_channels, _kernel_size=(1,1), _stride=(1, 1), _padding=(0,0), _dilation_rate=(1,1), _norm_type='BatchNorm')
        
    def forward(self, x: Tensor) -> Tensor:
        
        out1 = self.pooling_1(x)
        out2 = self.pooling_2(x)
        out3 = self.pooling_3(x)
        out4 = self.pooling_4(x)
        
        # concat
        out = torch.cat((out1,out2,out3,out4),dim=1)
        out = self.finalConv2DN(out)
        return out
    
    
class ResUNet_A_miniBlock(nn.Module):
    """
    This describes a miniblock in Fig 1b) of the paper.  The use of atrous convolutions was found by Diakogiannis et al. 'almost doubles the convergence rate'.
    
    Adapted by Daniel NF Lam from MXNet to Pytorch, with reference to:
    https://github.com/feevos/resuneta/blob/master/nn/BBlocks/resnet_blocks.py
    
    Default values follow the paper shown below.
    
    Paper:
    Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020). ResUNet-a: A deep learning 
    framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry 
    and Remote Sensing, 162, 94–114. https://doi.org/10.1016/j.isprsjprs.2020.01.013
    
    Convolution here uses atrous convolution.
    """
    def __init__( self, _in_channels: int,  _kernel_size=(3,3) , _dilation_rate=(1,1), _stride=(1,1), _norm_type='BatchNorm', **kwargs):
        super().__init__()
        
        self.in_channels = _in_channels  #input & output of res block has to have the same size
        self.out_channels = _in_channels #input & output of res block has to have the same size
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate
        self.stride = _stride
        if (_norm_type == 'BatchNorm'):
            self.norm = nn.BatchNorm2d
        elif (_norm_type == 'InstanceNorm'):
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        
        # PADDING for SAME CONVOLUTIONS (i.e. input in-plane size == output in-plane size)
        p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1)/2 
        p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1)/2 
        p = (int(p0),int(p1))
        
        # DEFINING THE LAYERS TO BE USED IN THIS BLOCK
        # LAYERS are objects, not functions.
        # Made here for ease of use & repeatability, reducing reused code
        self.BN = self.norm(num_features=self.in_channels, affine=True)
        self.conv2d = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=p,
            dilation=self.dilation_rate,
            bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # Forward function defines the network structure
        out = self.BN(x)
        out = self.relu(out)
        out = self.conv2d(out)

        out = self.BN(out)
        out = self.relu(out)
        out = self.conv2d(out)
        
        return out
    
class Conv2DN(nn.Module):
    """
    This class describes the CONV2DN blocks used in the listed paper.
    
    Paper:
    Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020). ResUNet-a: A deep learning 
    framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry 
    and Remote Sensing, 162, 94–114. https://doi.org/10.1016/j.isprsjprs.2020.01.013
    
    """
    def __init__( self, _in_channels: int, _out_channels: int, _kernel_size=(1,1), _stride=(1, 1), _padding=(0,0), _dilation_rate=(1,1), _norm_type='BatchNorm', **kwargs ):
        super().__init__()
        self.in_channels = _in_channels
        self.out_channels = _out_channels
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.dilation_rate = _dilation_rate
        self.padding = _padding
        
        if (_norm_type == 'BatchNorm'):
            self.norm = nn.BatchNorm2d
        elif (_norm_type == 'InstanceNorm'):
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        self.conv2d = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation_rate,
            bias=False)
        
        self.BN = self.norm(num_features=self.out_channels, affine=True)
        
    def forward(self,x: Tensor) -> Tensor:
        out = self.conv2d(x)
        out = self.BN(out)
        return out
    
class ResUNet_A_Block_4(nn.Module):
    """
    A multi-scale residual block that uses atrous convolutions at different dilation rates to examine different scales.
    
    Based on Diakogiannis et al.'s work on ResUNet-A in 2019:
    doi: 10.1016/j.isprsjprs.2020.01.013
    
    Which is itself based on Zhang et al.'s 2017 work:
    doi: 10.1109/LGRS.2018.2802944.
    """
    def __init__( self, _in_channels: int,  _kernel_size , _dilation_rates , _stride=(1,1) ,  _norm_type='BatchNorm', **kwargs):
        super().__init__()
        self.in_channels = _in_channels  #input & output of res block has to have the same size
        self.kernel_size = _kernel_size
        self.dilation_rate_acrossBlocks = _dilation_rates
        self.stride = _stride
        
        if (_norm_type == 'BatchNorm'):
            self.norm = nn.BatchNorm2d
        elif (_norm_type == 'InstanceNorm'):
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        d = self.dilation_rate_acrossBlocks[0]
        self.miniBlock1 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        d = self.dilation_rate_acrossBlocks[1]
        self.miniBlock2 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        d = self.dilation_rate_acrossBlocks[2]
        self.miniBlock3 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        d = self.dilation_rate_acrossBlocks[3]
        self.miniBlock4 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        
        # identity map
        out = x # residual
        out = out + self.miniBlock1(x)
        out = out + self.miniBlock2(x)
        out = out + self.miniBlock3(x)
        out = out + self.miniBlock4(x)
        return out

class ResUNet_A_Block_3(nn.Module):
    """
    A multi-scale residual block that uses atrous convolutions at different dilation rates to examine different scales.
    
    Based on Diakogiannis et al.'s work on ResUNet-A in 2019:
    doi: 10.1016/j.isprsjprs.2020.01.013
    
    Which is itself based on Zhang et al.'s 2017 work:
    doi: 10.1109/LGRS.2018.2802944.
    """
    def __init__( self, _in_channels: int,  _kernel_size , _dilation_rates , _stride=(1,1) ,  _norm_type='BatchNorm', **kwargs):
        super().__init__()
        self.in_channels = _in_channels  #input & output of res block has to have the same size
        self.kernel_size = _kernel_size
        self.dilation_rate_acrossBlocks = _dilation_rates
        self.stride = _stride
        
        if (_norm_type == 'BatchNorm'):
            self.norm = nn.BatchNorm2d
        elif (_norm_type == 'InstanceNorm'):
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        d = self.dilation_rate_acrossBlocks[0]
        self.miniBlock1 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        d = self.dilation_rate_acrossBlocks[1]
        self.miniBlock2 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        d = self.dilation_rate_acrossBlocks[2]
        self.miniBlock3 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        
        # identity map
        out = x
        out = out + self.miniBlock1(x)
        out = out + self.miniBlock2(x)
        out = out + self.miniBlock3(x)
        return out
class ResUNet_A_Block_1(nn.Module):
    """
    A multi-scale residual block that uses atrous convolutions at different dilation rates to examine different scales.
    
    Based on Diakogiannis et al.'s work on ResUNet-A in 2019:
    doi: 10.1016/j.isprsjprs.2020.01.013
    
    Which is itself based on Zhang et al.'s 2017 work:
    doi: 10.1109/LGRS.2018.2802944.
    """
    def __init__( self, _in_channels: int,  _kernel_size , _dilation_rates=(1) , _stride=(1,1) ,  _norm_type='BatchNorm', **kwargs):
        super().__init__()
        self.in_channels = _in_channels  #input & output of res block has to have the same size
        self.kernel_size = _kernel_size
        self.dilation_rate_acrossBlocks = _dilation_rates
        self.stride = _stride
        
        """if (_norm_type == 'BatchNorm'):
            self.norm = nn.BatchNorm2d
        elif (_norm_type == 'InstanceNorm'):
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError"""
        
        d = self.dilation_rate_acrossBlocks[0]
        self.miniBlock1 = ResUNet_A_miniBlock( self.in_channels,  _kernel_size=self.kernel_size , _dilation_rate=(d,d), _stride=self.stride, _norm_type=_norm_type, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        
        # identity map
        out = x
        out = out + self.miniBlock1(x)
        return out
    

class DownSample(nn.Module):
    """
    Convolutional NN to reduce the in-plane dimensions by a factor of 2 each, and increase channels by a factor of 2.
    Default values follow the paper set on Diakogiannis et al.
    From Diakogiannis et al.
    doi: 10.1016/j.isprsjprs.2020.01.013
    """
    def __init__(self, _in_channels, _factor=2, _kernel_size=(1,1), _stride=(2, 2), _padding=(0,0), _dilation_rate=(1,1), **kwargs ):
        super().__init__()
        
        self.in_channels = _in_channels;
        self.factor = _factor
        self.out_channels = _in_channels*_factor
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.dilation_rate = _dilation_rate
        self.padding = _padding
        
        self.conv_layer = nn.Conv2d(
                            in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation_rate,
                            bias=False)
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_layer(x)
        return out

    
class UpSampleAndHalveChannels(nn.Module):
    """
    Doubles the spatial dimensions (H,W) but halves the number of channels.
    Inverse of the DownSample function in blocks.py
    
    From Diakogiannis et al.
    doi: 10.1016/j.isprsjprs.2020.01.013
    """
    def __init__(self, _in_channels, _factor=2):
        super().__init__()
        
        self.in_channels = _in_channels
        self.factor = _factor
        
        self.upSample = nn.Upsample(scale_factor=self.factor, mode='bilinear', align_corners=None)
        
        self.halveChannels = nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.in_channels//self.factor,
                            kernel_size=(1,1),
                            stride=1,
                            padding=0,
                            dilation=1,
                            bias=False)
    def forward(self, x: Tensor) -> Tensor:
        out = self.upSample(x)
        out = self.halveChannels(out)
        return out
        
class Combine(nn.Module):
    """
    
    """
    def __init__(self, _in_channels):
        super().__init__()
        
        self.in_channels_per_tensor = _in_channels
        self.relu = nn.ReLU()
        self.conv2dn = Conv2DN( _in_channels = self.in_channels_per_tensor*2, _out_channels=self.in_channels_per_tensor, _kernel_size=(1,1), _stride=(1, 1), _padding=(0,0), _dilation_rate=(1,1), _norm_type='BatchNorm')
        
    def forward(self, decoder_tensor: Tensor, skip_tensor: Tensor) -> Tensor:
        # Upsample
        out = self.relu(decoder_tensor)
        out = torch.cat((out, skip_tensor), axis=1)
        out = self.conv2dn(out)
        return out
        
        
class Encoder_ResUNet_A_d7(nn.Module):
    def __init__(self, _input_channels, _input_array_shape, _norm_type='BatchNorm', _ADL_drop_rate=0.75, _ADL_gamma=0.9):
        super().__init__();
        
        self.initial_channels = _input_channels # Initial number of filters out from conv_first_normed
        self.initial_spatial = (_input_array_shape[2], _input_array_shape[3]) # (H,W)
        """
        ResUNet Encoder Section from Diakogiannis et al.
        doi: 10.1016/j.isprsjprs.2020.01.013
        """
        
        
        _out_channels_1 = self.initial_channels*2**(0)
        self.conv_first_normed = Conv2DN(_input_array_shape[1], _out_channels_1,
                                              _kernel_size=(1,1),
                                              _norm_type = _norm_type)
        
        self.EncResBlk1 = ResUNet_A_Block_4( _in_channels=_out_channels_1,  _kernel_size=(3,3), _dilation_rates=[1,3,15,31], _norm_type=_norm_type)
        self.ADL1 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        self.DnSmpl = DownSample(_in_channels=_out_channels_1, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        spatial = tuple(map(lambda num: num *(0.5**1), self.initial_spatial))
        _out_channels_2 = self.initial_channels*2**(1)
        
        self.EncResBlk2 = ResUNet_A_Block_4(_in_channels=_out_channels_2,  _kernel_size=(3,3), _dilation_rates=[1,3,15,31], _norm_type=_norm_type)
        self.ADL2 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        self.DnSmp2 = DownSample(_in_channels=_out_channels_2, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        spatial = tuple(map(lambda num: num *(0.5**2), self.initial_spatial))
        _out_channels_3 = self.initial_channels*2**(2)
        
        self.EncResBlk3 = ResUNet_A_Block_3( _in_channels=_out_channels_3,  _kernel_size=(3,3), _dilation_rates=[1,3,15], _norm_type=_norm_type)
        self.ADL3 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        self.DnSmp3 = DownSample(_in_channels=_out_channels_3, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        spatial = tuple(map(lambda num: num *(0.5**3), self.initial_spatial))
        _out_channels_4 = self.initial_channels*2**(3)
        
        self.EncResBlk4 = ResUNet_A_Block_3( _in_channels=_out_channels_4,  _kernel_size=(3,3), _dilation_rates=[1,3,15], _norm_type=_norm_type)
        self.ADL4 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        self.DnSmp4 = DownSample(_in_channels=_out_channels_4, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        spatial = tuple(map(lambda num: num *(0.5**4), self.initial_spatial))
        _out_channels_5 = self.initial_channels*2**(4)
        
        self.EncResBlk5 = ResUNet_A_Block_1( _in_channels=_out_channels_5,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        self.ADL5 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        self.DnSmp5 = DownSample(_in_channels=_out_channels_5, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        spatial = tuple(map(lambda num: num *(0.5**5), self.initial_spatial))
        _out_channels_6 = self.initial_channels*2**(5)
        
        self.EncResBlk6 = ResUNet_A_Block_1( _in_channels=_out_channels_6,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        self.ADL6 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        self.DnSmp6 = DownSample(_in_channels=_out_channels_6, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        spatial = tuple(map(lambda num: num *(0.5**6), self.initial_spatial))
        _out_channels_7 = self.initial_channels*2**(6)
        
        self.EncResBlk7 = ResUNet_A_Block_1( _in_channels=_out_channels_7,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        self.ADL7 = ADL(drop_rate=_ADL_drop_rate, gamma=_ADL_gamma)
        
        self.output_array_size = tuple(map(lambda x: int(x), (0, _out_channels_7, spatial[0], spatial[1])))
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Encoder Section
        """
        out = self.conv_first_normed(x)
        out = self.EncResBlk1(out)
        out = self.ADL1(out)
        out = self.DnSmpl(out)
        out = self.EncResBlk2(out)
        out = self.ADL2(out)
        out = self.DnSmp2(out)
        out = self.EncResBlk3(out)
        out = self.ADL3(out)
        out = self.DnSmp3(out)
        out = self.EncResBlk4(out)
        out = self.ADL4(out)
        out = self.DnSmp4(out)
        out = self.EncResBlk5(out)
        out = self.ADL5(out)
        out = self.DnSmp5(out)
        out = self.EncResBlk6(out)
        out = self.ADL6(out)
        out = self.DnSmp6(out)
        out = self.EncResBlk7(out)
        out = self.ADL7(out)
        return out
        

class MultiScale_Classifier(nn.Module):
    """
    Consists of 2 parts:
    
    Multi-Scale ResUNet encoder from Diakogiannis et al. until the bridge part.
    doi: 10.1016/j.isprsjprs.2020.01.013
    
    Classifier section from Wang et al. COVID classification from CT.
    doi: 10.1183/13993003.00775-2020
    """
    def __init__(self, _input_channels, _input_array_shape, _classifier_out_channels=64, _norm_type='BatchNorm', _ADL_drop_rate=0.75, _ADL_gamma=0.9):
        super().__init__();
        
        self.initial_channels = _input_channels # Initial number of filters output from the encoder's first layer
        self.input_array_shape = _input_array_shape
        self.classifier_out_channels = _classifier_out_channels
        self.norm_type = _norm_type
        self.ADL_drop_rate = _ADL_drop_rate
        self.ADL_gamma = _ADL_gamma
        
        """
        ResUNet Encoder Section from Diakogiannis et al.
        doi: 10.1016/j.isprsjprs.2020.01.013
        """
        
        self.enc = Encoder_ResUNet_A_d7(_input_channels=self.initial_channels, _input_array_shape=self.input_array_shape,
                                        _norm_type=self.norm_type, 
                                        _ADL_drop_rate=self.ADL_drop_rate, _ADL_gamma=self.ADL_gamma)
        _out_channels = self.enc.output_array_size[1]
        """
        Classifier Section from Wang et al. 2020. A fully automatic deep learning system for COVID-19 diagnostic and prognostic analysis.
        DOI: 10.1183/13993003.00775-2020
        """
        # Max Pool
        self.MaxPool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm = nn.BatchNorm2d(num_features=_out_channels, affine=False)
        self.relu = nn.ReLU()
        self.conv2dense = nn.Conv2d(in_channels=_out_channels, out_channels=_classifier_out_channels, kernel_size=(1,1),stride=(1,1),padding=0, dilation=1, bias=False)
        self.GlobalAvgPool2d = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dense_classifier = nn.Linear(in_features=_classifier_out_channels, out_features=1, bias=True)
        #self.output_activation = nn.Sigmoid()
    def forward(self, x: Tensor) -> int:
        """
        The input tensor x (Torch Tensor) contains both the real/fake image and the conditioning image, concatenated in the channel axis
        """
        out = self.enc(x)
        
        """
        Classifier Section
        """        
        out = self.MaxPool(out)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.conv2dense(out)
        out = self.GlobalAvgPool2d(out)
        out = self.flatten(out)
        out = self.dense_classifier(out)
        #out = self.output_activation(out)
        return out

class Generator_ResUNet_A(nn.Module):
    def __init__(self, _input_channels, _input_array_shape, _norm_type='BatchNorm', _ADL_drop_rate=0.75, _ADL_gamma=0.9):
        super().__init__()
        
        # ENCODER
        self.enc = Encoder_ResUNet_A_d7(_input_channels, _input_array_shape, _norm_type=_norm_type, _ADL_drop_rate=_ADL_drop_rate, _ADL_gamma=_ADL_gamma)
        _tensor_array_shape = self.enc.output_array_size
        initial_spatial = (_tensor_array_shape[2], _tensor_array_shape[3])
        
        # BRIDGE
        self.bridge = PSPPooling(_tensor_array_shape)
        
        #DECODER
        self.upsh1 = UpSampleAndHalveChannels(_in_channels=_tensor_array_shape[1], _factor=2)
        out_channels_8 = int(_tensor_array_shape[1]*0.5**(1))
        spatial = tuple(map(lambda num: num *(2**1), initial_spatial))
        self.comb1 = Combine(_in_channels=out_channels_8)
        self.DecResBlk1 = ResUNet_A_Block_1(_in_channels=out_channels_8,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        
        self.upsh2 = UpSampleAndHalveChannels(_in_channels=out_channels_8, _factor=2)
        out_channels_9 = int(_tensor_array_shape[1]*0.5**(2))
        spatial = tuple(map(lambda num: num *(2**2), initial_spatial))
        self.comb2 = Combine(_in_channels=out_channels_9)
        self.DecResBlk2 = ResUNet_A_Block_1(_in_channels=out_channels_9,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        
        self.upsh3 = UpSampleAndHalveChannels(_in_channels=out_channels_9, _factor=2)
        out_channels_10 = int(_tensor_array_shape[1]*0.5**(3))
        spatial = tuple(map(lambda num: num *(2**3), initial_spatial))
        self.comb3 = Combine(_in_channels=out_channels_10)
        self.DecResBlk3 = ResUNet_A_Block_1(_in_channels=out_channels_10,  _kernel_size=(3,3), _dilation_rates=[1, 3, 15], _norm_type=_norm_type)
        
        self.upsh4 = UpSampleAndHalveChannels(_in_channels=out_channels_10, _factor=2)
        out_channels_11 = int(_tensor_array_shape[1]*0.5**(4))
        spatial = tuple(map(lambda num: num *(2**4), initial_spatial))
        self.comb4 = Combine(_in_channels=out_channels_11)
        self.DecResBlk4 = ResUNet_A_Block_1(_in_channels=out_channels_11,  _kernel_size=(3,3), _dilation_rates=[1, 3, 15], _norm_type=_norm_type)
        
        self.upsh5 = UpSampleAndHalveChannels(_in_channels=out_channels_11, _factor=2)
        out_channels_12 = int(_tensor_array_shape[1]*0.5**(5))
        spatial = tuple(map(lambda num: num *(2**5), initial_spatial))
        self.comb5 = Combine(_in_channels=out_channels_12)
        self.DecResBlk5 = ResUNet_A_Block_1(_in_channels=out_channels_12,  _kernel_size=(3,3), _dilation_rates=[1, 3, 15, 31], _norm_type=_norm_type)
        
        self.upsh6 = UpSampleAndHalveChannels(_in_channels=out_channels_12, _factor=2)
        out_channels_13 = int(_tensor_array_shape[1]*0.5**(6))
        spatial = tuple(map(lambda num: num *(2**6), initial_spatial))
        self.comb6 = Combine(_in_channels=out_channels_13)
        self.DecResBlk6 = ResUNet_A_Block_1(_in_channels=out_channels_13,  _kernel_size=(3,3), _dilation_rates=[1, 3, 15, 31], _norm_type=_norm_type)
        
        self.comb_last = Combine(_in_channels=out_channels_13)
        
        self.PSPpool_last = PSPPooling((0, out_channels_13, int(spatial[0]), int(spatial[1]) ) )
        
        self.conv2d_final = nn.Conv2d(in_channels=out_channels_13, out_channels=1, kernel_size=(1,1),stride=(1,1),padding=0, dilation=1, bias=False)
        self.output_image = nn.Tanh()
    def forward(self, x: Tensor) -> Tensor:
        """
        Encoder Section
        """
        out1 = self.enc.conv_first_normed(x)
        out2 = self.enc.EncResBlk1(out1)
        out = self.enc.ADL1(out2)
        out = self.enc.DnSmpl(out)
        out4 = self.enc.EncResBlk2(out)
        out = self.enc.ADL2(out4)
        out = self.enc.DnSmp2(out)
        out6 = self.enc.EncResBlk3(out)
        out = self.enc.ADL3(out6)
        out = self.enc.DnSmp3(out)
        out8 = self.enc.EncResBlk4(out)
        out = self.enc.ADL4(out8)
        out = self.enc.DnSmp4(out)
        out10 = self.enc.EncResBlk5(out)
        out = self.enc.ADL5(out10)
        out = self.enc.DnSmp5(out)
        out12 = self.enc.EncResBlk6(out)
        out = self.enc.ADL6(out12)
        out = self.enc.DnSmp6(out)
        out = self.enc.EncResBlk7(out)
        out = self.enc.ADL7(out)
        
        out = self.bridge(out)
        
        
        out = self.upsh1(out)
        out = self.comb1(out, out12)
        out = self.DecResBlk1(out)
        
        out = self.upsh2(out)
        out = self.comb2(out, out10)
        out = self.DecResBlk2(out)
        
        out = self.upsh3(out)
        out = self.comb3(out, out8)
        out = self.DecResBlk3(out)
        
        out = self.upsh4(out)
        out = self.comb4(out, out6)
        out = self.DecResBlk4(out)
        
        out = self.upsh5(out)
        out = self.comb5(out, out4)
        out = self.DecResBlk5(out)
        
        out = self.upsh6(out)
        out = self.comb6(out, out2)
        out = self.DecResBlk6(out)
        
        out = self.comb_last(out, out1)
        out = self.PSPpool_last(out)
        out = self.conv2d_final(out)
        out = self.output_image(out)
        return out

####################################
# Deep Residual U-Net
# Zhang, Z., Liu, Q., & Wang, Y. (2018). Road Extraction by Deep Residual U-Net. IEEE Geoscience and Remote Sensing Letters, 15(5), 749–753. https://doi.org/10.1109/LGRS.2018.2802944
####################################
class ResUNet_block(nn.Module):
    def __init__(self, _in_channels, _out_channels, _kernel_size, _stride, _padding, _reluType, _normType="BatchNorm"):
        super().__init__()
        self.in_channels = _in_channels
        self.out_channels = _out_channels
        self.kernel_size=_kernel_size
        self.stride=_stride
        self.padding = _padding
        self.reluType = _reluType
        self.normType = _normType
        
        # BN
        if self.normType=="BatchNorm":
            self.norm = nn.BatchNorm2d(num_features=self.in_channels, affine=False)
        if self.normType == "InstanceNorm":
            self.norm = nn.InstanceNorm2d(num_features=self.in_channels, affine=False)
            
        # ReLU
        if self.reluType == "normal":
            self.relu = nn.ReLU()
        if self.reluType == "leaky":
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        # Conv2d
        self.conv2d = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                                dilation=1, bias=True)
    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv2d(out)
        return out
    
class ResUNet_shortcut(nn.Module):
    def __init__(self, _input_tensor_channels, _output_channels, _stride, _normType="BatchNorm"):
        super().__init__()
        self.shortcut_conv = nn.Conv2d(in_channels=_input_tensor_channels, out_channels=_output_channels,
                                kernel_size=1, stride=_stride, padding=0, 
                                dilation=1, bias=False)
        if self.normType=="BatchNorm":
            self.shortcut_norm = nn.BatchNorm2d(num_features=_output_channels, affine=False)
        if self.normType == "InstanceNorm":
            self.shortcut_norm = nn.InstanceNorm2d(num_features=_output_channels, affine=False)
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.shortcut_conv(x)
        out = self.shortcut_norm(out)
        return out

class Generator_ResUNet(nn.Module):
    """
    Zhang, Z., Liu, Q., & Wang, Y. (2018). Road Extraction by Deep Residual U-Net. IEEE Geoscience and Remote Sensing Letters, 15(5), 749–753. https://doi.org/10.1109/LGRS.2018.2802944
    """
    def __init__(self, input_array_shape, _first_out_channels=64, _reluType="leaky", _normType="BatchNorm"):
        super().__init__()
        self.first_out_channels = _first_out_channels
        self.input_array_shape = input_array_shape
        self.reluType = _reluType
        self.normType = _normType
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=input_array_shape[1], out_channels=self.first_out_channels,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                                dilation=1, bias=False)
        self.convblock12 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        
        self.convblock21 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock22 = ResUNet_block(_in_channels=self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut2 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**0),
                                          _output_channels=self.first_out_channels*(2**1), _stride=2, _normType=self.normType)
        
        self.convblock31 = ResUNet_block(_in_channels=self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock32 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut3 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**1),
                                          _output_channels=self.first_out_channels*(2**2), _stride=2, _normType=self.normType)
        # Bridge
        self.convblockB1 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**3),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblockB2 = ResUNet_block(_in_channels=self.first_out_channels*(2**3),
                                         _out_channels=self.first_out_channels*(2**3),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcutB = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**2),
                                          _output_channels=self.first_out_channels*(2**3), _stride=2, _normType=self.normType)
        
        # Decoder
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.convblock51 = ResUNet_block(_in_channels=self.first_out_channels*(2**3)+self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock52 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut5 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**3)+self.first_out_channels*(2**2),
                                          _output_channels=self.first_out_channels*(2**2), _stride=1, _normType=self.normType)
        
        self.convblock61 = ResUNet_block(_in_channels=self.first_out_channels*(2**2)+self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock62 = ResUNet_block(_in_channels=self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut6 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**2)+self.first_out_channels*(2**1),
                                          _output_channels=self.first_out_channels*(2**1), _stride=1, _normType=self.normType)
        self.convblock71 = ResUNet_block(_in_channels=self.first_out_channels*(2**1)+self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock72 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut7 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**1)+self.first_out_channels*(2**0),
                                          _output_channels=self.first_out_channels*(2**0), _stride=1, _normType=self.normType)
        self.output_conv = nn.Conv2d(in_channels=self.first_out_channels*(2**0), out_channels=input_array_shape[1],
                                kernel_size=(1,1), stride=(1,1), padding=(0,0), 
                                dilation=1, bias=False)
        self.output_activation = nn.Sigmoid()
        
        
    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        out = self.conv1(x)
        out = self.convblock12(out)
        out1 = out + x
        
        out = self.convblock21(out1)
        out = self.convblock22(out)
        shortcut = self.shortcut2(out1)
        out2 = out + shortcut
        
        out = self.convblock31(out2)
        out = self.convblock32(out)
        shortcut = self.shortcut3(out2)
        out3 = out + shortcut
        
        # Bridge
        out = self.convblockB1(out3)
        out = self.convblockB2(out)
        shortcut = self.shortcutB(out3)
        out = out + shortcut
        
        # Decoder
        out = self.upSample(out)
        out5 = torch.cat((out, out3), axis=1)
        out = self.convblock51(out5)
        out = self.convblock52(out)
        shortcut = self.shortcut5(out5)
        out = out + shortcut
        
        out = self.upSample(out)
        out6 = torch.cat((out, out2), axis=1)
        out = self.convblock61(out6)
        out = self.convblock62(out)
        shortcut = self.shortcut6(out6)
        out = out + shortcut
        
        out = self.upSample(out)
        out7 = torch.cat((out, out1), axis=1)
        out = self.convblock71(out7)
        out = self.convblock72(out)
        shortcut = self.shortcut7(out7)
        out = out + shortcut
        
        out = self.output_conv(out)
        out = self.output_activation(out)
        return out


###############################
# Special Blocks
# i.e. Custom Nets
###############################

class Generator_ResUNet_modified(nn.Module):
    """
    Use the ResUNet as a starting point, then add attention modules, etc.
    Added:
    1) 2021-04-13: dropout in decoder, like Pix2Pix, after the first 3 conv layers -- use '_dropoutType="normal"' to activate.
    2) 2021-04-13: attention-dropout layer implemented -- switch _dropoutType to "ADL" to activate.
    """
    def __init__(self, input_array_shape, _first_out_channels=64, _reluType="leaky", _normType = "BatchNorm", _dropoutType="ADL", _drop_rate=0.5, _output_activation="Sigmoid"):
        super().__init__()
        self.first_out_channels = _first_out_channels
        self.input_array_shape = input_array_shape
        self.reluType = _reluType
        self.dropoutType = _dropoutType
        self.outputActivationType = _output_activation
        self.normType = _normType
        # Dropouts
        if self.dropoutType == "ADL":
            self.dropout1 = ADL(drop_rate=_drop_rate, gamma=0.9)
            self.dropout2 = ADL(drop_rate=_drop_rate, gamma=0.9)
            self.dropout3 = ADL(drop_rate=_drop_rate, gamma=0.9)
        if self.dropoutType == "normal":
            self.dropout1 = nn.Dropout(p=_drop_rate)
            self.dropout2 = nn.Dropout(p=_drop_rate)
            self.dropout3 = nn.Dropout(p=_drop_rate)
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=input_array_shape[1], out_channels=self.first_out_channels,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                                dilation=1, bias=False)
        self.convblock12 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        
        self.convblock21 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock22 = ResUNet_block(_in_channels=self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut2 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**0),
                                          _output_channels=self.first_out_channels*(2**1), _stride=2, _normType=self.normType)
        
        self.convblock31 = ResUNet_block(_in_channels=self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock32 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut3 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**1),
                                          _output_channels=self.first_out_channels*(2**2), _stride=2)
        # Bridge
        self.convblockB1 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**3),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblockB2 = ResUNet_block(_in_channels=self.first_out_channels*(2**3),
                                         _out_channels=self.first_out_channels*(2**3),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcutB = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**2),
                                          _output_channels=self.first_out_channels*(2**3), _stride=2, _normType=self.normType)
        
        # Decoder
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.convblock51 = ResUNet_block(_in_channels=self.first_out_channels*(2**3)+self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock52 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut5 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**3)+self.first_out_channels*(2**2),
                                          _output_channels=self.first_out_channels*(2**2), _stride=1, _normType=self.normType)
        
        self.convblock61 = ResUNet_block(_in_channels=self.first_out_channels*(2**2)+self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock62 = ResUNet_block(_in_channels=self.first_out_channels*(2**1),
                                         _out_channels=self.first_out_channels*(2**1),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut6 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**2)+self.first_out_channels*(2**1),
                                          _output_channels=self.first_out_channels*(2**1), _stride=1, _normType=self.normType)
        self.convblock71 = ResUNet_block(_in_channels=self.first_out_channels*(2**1)+self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.convblock72 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType, _normType=self.normType)
        self.shortcut7 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**1)+self.first_out_channels*(2**0),
                                          _output_channels=self.first_out_channels*(2**0), _stride=1, _normType=self.normType)
        self.output_conv = nn.Conv2d(in_channels=self.first_out_channels*(2**0), out_channels=input_array_shape[1],
                                kernel_size=(1,1), stride=(1,1), padding=(0,0), 
                                dilation=1, bias=False)
        
        if self.outputActivationType == "Sigmoid":
            self.output_activation = nn.Sigmoid()
        if self.outputActivationType == "Tanh":
            self.output_activation = nn.Tanh()
        
    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        out = self.conv1(x)
        out = self.convblock12(out)
        out1 = out + x
        
        out = self.convblock21(out1)
        out = self.convblock22(out)
        shortcut = self.shortcut2(out1)
        out2 = out + shortcut
        
        out = self.convblock31(out2)
        out = self.convblock32(out)
        shortcut = self.shortcut3(out2)
        out3 = out + shortcut
        
        # Bridge
        out = self.convblockB1(out3)
        out = self.convblockB2(out)
        shortcut = self.shortcutB(out3)
        out = out + shortcut
        
        # Decoder
        out = self.upSample(out)
        out5 = torch.cat((out, out3), axis=1)
        out = self.convblock51(out5)
        out = self.dropout1(out)
        out = self.convblock52(out)
        out = self.dropout2(out)
        shortcut = self.shortcut5(out5)
        out = out + shortcut
        
        out = self.upSample(out)
        out6 = torch.cat((out, out2), axis=1)
        out = self.convblock61(out6)
        out = self.dropout3(out)
        out = self.convblock62(out)
        shortcut = self.shortcut6(out6)
        out = out + shortcut
        
        out = self.upSample(out)
        out7 = torch.cat((out, out1), axis=1)
        out = self.convblock71(out7)
        out = self.convblock72(out)
        shortcut = self.shortcut7(out7)
        out = out + shortcut
        
        out = self.output_conv(out)
        out = self.output_activation(out)
        return out

class Generator_ResUNet_PixelShuffle(nn.Module):
    """
    A ResUNet that uses PixelShuffle to upsample.  This was found to provide poor results in pre-training.
    """
    def __init__(self, input_array_shape, _first_out_channels=64, _reluType="leaky", _dropoutType="ADL", _drop_rate=0.5, _output_activation="Sigmoid"):
        super().__init__()
        self.first_out_channels = _first_out_channels
        self.input_array_shape = input_array_shape
        self.reluType = _reluType
        self.dropoutType = _dropoutType
        self.outputActivationType = _output_activation
        
        # Dropouts
        if self.dropoutType == "ADL":
            self.dropout1 = ADL(drop_rate=_drop_rate, gamma=0.9)
            self.dropout2 = ADL(drop_rate=_drop_rate, gamma=0.9)
            self.dropout3 = ADL(drop_rate=_drop_rate, gamma=0.9)
        if self.dropoutType == "normal":
            self.dropout1 = nn.Dropout(p=_drop_rate)
            self.dropout2 = nn.Dropout(p=_drop_rate)
            self.dropout3 = nn.Dropout(p=_drop_rate)
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=input_array_shape[1], out_channels=self.first_out_channels,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                                dilation=1, bias=False)
        self.convblock12 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        
        self.convblock21 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType)
        self.convblock22 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.shortcut2 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**0),
                                          _output_channels=self.first_out_channels*(2**2), _stride=2)
        
        self.convblock31 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**4),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType)
        self.convblock32 = ResUNet_block(_in_channels=self.first_out_channels*(2**4),
                                         _out_channels=self.first_out_channels*(2**4),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.shortcut3 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**2),
                                          _output_channels=self.first_out_channels*(2**4), _stride=2)
        # Bridge
        self.convblockB1 = ResUNet_block(_in_channels=self.first_out_channels*(2**4),
                                         _out_channels=self.first_out_channels*(2**6),
                                         _kernel_size=(3,3), _stride=(2,2), _padding=(1,1),
                                         _reluType=self.reluType)
        self.convblockB2 = ResUNet_block(_in_channels=self.first_out_channels*(2**6),
                                         _out_channels=self.first_out_channels*(2**6),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        # Decoder
        self.upSample = nn.PixelShuffle(upscale_factor=2)
        
        self.convblock51 = ResUNet_block(_in_channels=self.first_out_channels*(2**4)+self.first_out_channels*(2**4),
                                         _out_channels=self.first_out_channels*(2**4),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.convblock52 = ResUNet_block(_in_channels=self.first_out_channels*(2**4),
                                         _out_channels=self.first_out_channels*(2**4),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.shortcut5 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**4)+self.first_out_channels*(2**4),
                                          _output_channels=self.first_out_channels*(2**4), _stride=1)
        
        self.convblock61 = ResUNet_block(_in_channels=self.first_out_channels*(2**2)+self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.convblock62 = ResUNet_block(_in_channels=self.first_out_channels*(2**2),
                                         _out_channels=self.first_out_channels*(2**2),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.shortcut6 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**2)+self.first_out_channels*(2**2),
                                          _output_channels=self.first_out_channels*(2**2), _stride=1)
        
        self.convblock71 = ResUNet_block(_in_channels=self.first_out_channels*(2**0)+self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.convblock72 = ResUNet_block(_in_channels=self.first_out_channels*(2**0),
                                         _out_channels=self.first_out_channels*(2**0),
                                         _kernel_size=(3,3), _stride=(1,1), _padding=(1,1),
                                         _reluType=self.reluType)
        self.shortcut7 = ResUNet_shortcut(_input_tensor_channels=self.first_out_channels*(2**0)+self.first_out_channels*(2**0),
                                          _output_channels=self.first_out_channels*(2**0), _stride=1)
        self.output_conv = nn.Conv2d(in_channels=self.first_out_channels*(2**0), out_channels=input_array_shape[1],
                                kernel_size=(1,1), stride=(1,1), padding=(0,0), 
                                dilation=1, bias=False)
        
        if self.outputActivationType == "Sigmoid":
            self.output_activation = nn.Sigmoid()
        if self.outputActivationType == "Tanh":
            self.output_activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        out = self.conv1(x)
        out = self.convblock12(out)
        out1 = out + x
        
        out = self.convblock21(out1)
        out = self.convblock22(out)
        shortcut = self.shortcut2(out1)
        out2 = out + shortcut
        
        out = self.convblock31(out2)
        out = self.convblock32(out)
        shortcut = self.shortcut3(out2)
        out3 = out + shortcut
        
        # Bridge
        out = self.convblockB1(out3)
        out = self.convblockB2(out)
        
        
        # Decoder
        out = self.upSample(out)
        out5 = torch.cat((out, out3), axis=1)
        out = self.convblock51(out5)
        out = self.dropout1(out)
        out = self.convblock52(out)
        out = self.dropout2(out)
        shortcut = self.shortcut5(out5)
        out = out + shortcut
        
        out = self.upSample(out)
        out6 = torch.cat((out, out2), axis=1)
        out = self.convblock61(out6)
        out = self.dropout3(out)
        out = self.convblock62(out)
        shortcut = self.shortcut6(out6)
        out = out + shortcut
        
        out = self.upSample(out)
        out7 = torch.cat((out, out1), axis=1)
        out = self.convblock71(out7)
        out = self.convblock72(out)
        shortcut = self.shortcut7(out7)
        out = out + shortcut
        
        out = self.output_conv(out)
        out = self.output_activation(out)
        return out
    
############################################################
# MEDGAN
# U-net blocks as generator
# 1) Perceptual loss instead of pixel-distance loss
# 2) use VGG19 network for feature extraction
############################################################
