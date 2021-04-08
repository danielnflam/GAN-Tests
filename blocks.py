# IMPLEMENT THE RESNET
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

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
        
    def forward(self, x: Tensor) -> int:
        """
        Encoder Section
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
        
        return out