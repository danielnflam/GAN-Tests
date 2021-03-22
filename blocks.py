# IMPLEMENT THE RESNET
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

class ADL(nn.Module):
    """
    From the work done by Choe et al. 2019.
    Choe, J., & Shim, H. (2019). Attention-Based Dropout Layer for Weakly Supervised Object Localization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2219–2228.
    Source: https://github.com/clovaai/wsolevaluation
    """
    def __init__(self, drop_rate, gamma):
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
        
        selected_map
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
        
        if (_norm_type == 'BatchNorm'):
            self.norm = nn.BatchNorm2d
        elif (_norm_type == 'InstanceNorm'):
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
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
        
class MultiScale_Classifier(nn.Module):
    """
    Consists of 2 parts:
    
    Multi-Scale ResUNet encoder from Diakogiannis et al. until the bridge part.
    doi: 10.1016/j.isprsjprs.2020.01.013
    
    Classifier section from Wang et al. COVID classification from CT.
    doi: 10.1183/13993003.00775-2020
    """
    def __init__(self, _input_channels, _classifier_out_channels=64, _norm_type='BatchNorm'):
        super().__init__();
        
        self.initial_channels = _input_channels # Initial number of filters 
        
        """
        ResUNet Encoder Section from Diakogiannis et al.
        doi: 10.1016/j.isprsjprs.2020.01.013
        """
        _out_channels = self.initial_channels*2**(1)
        self.conv_first_normed = Conv2DN(self.initial_channels, _out_channels,
                                              _kernel_size=(1,1),
                                              _norm_type = _norm_type)
        
        self.EncResBlk1 = ResUNet_A_Block_4( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1,3,15,31], _norm_type=_norm_type)
        self.DnSmpl = DownSample(_out_channels, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        _out_channels = self.initial_channels*2**(2)
        
        self.EncResBlk2 = ResUNet_A_Block_4( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1,3,15,31], _norm_type=_norm_type)
        self.DnSmp2 = DownSample(_out_channels, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        _out_channels = self.initial_channels*2**(3)
        
        self.EncResBlk3 = ResUNet_A_Block_3( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1,3,5], _norm_type=_norm_type)
        self.DnSmp3 = DownSample(_out_channels, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        _out_channels = self.initial_channels*2**(4)
        
        self.EncResBlk4 = ResUNet_A_Block_3( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1,3,5], _norm_type=_norm_type)
        self.DnSmp4 = DownSample(_out_channels, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        _out_channels = self.initial_channels*2**(5)
        
        self.EncResBlk5 = ResUNet_A_Block_1( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        self.DnSmp5 = DownSample(_out_channels, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        _out_channels = self.initial_channels*2**(6)
        
        self.EncResBlk6 = ResUNet_A_Block_1( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1], _norm_type=_norm_type)
        self.DnSmp6 = DownSample(_out_channels, _kernel_size=(1,1) , stride=(2,2), padding=(0,0),_norm_type=_norm_type)
        _out_channels = self.initial_channels*2**(7)
        
        self.EncResBlk7 = ResUNet_A_Block_4( _out_channels,  _kernel_size=(3,3), _dilation_rates=[1,3,15,31], _norm_type=_norm_type)
        
        """
        Classifier Section from Wang et al. 2020. A fully automatic deep learning system for COVID-19 diagnostic and prognostic analysis.
        DOI: 10.1183/13993003.00775-2020
        """
        # Max Pool
        self.MaxPool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.BN = nn.BatchNorm2d(num_features=_out_channels, affine=False)
        self.relu = nn.ReLU()
        self.conv2dense = nn.Conv2d(in_channels=_out_channels, out_channels=_classifier_out_channels, kernel_size=(1,1),stride=(1,1),padding=0, dilation=1, bias=False)
        self.GlobalAvgPool2d = nn.AdaptiveAvgPool2d((1,1))
        self.dense_classifier = nn.Linear(in_features=_classifier_out_channels, out_features=1, bias=True)
        
    def forward(self, x: Tensor) -> int:
        """
        Encoder Section
        """
        out = self.conv_first_normed(x)
        out = self.EncResBlk1(out)
        out = self.DnSmpl(out)
        out = self.EncResBlk2(out)
        out = self.DnSmp2(out)
        out = self.EncResBlk3(out)
        out = self.DnSmp3(out)
        out = self.EncResBlk4(out)
        out = self.DnSmp4(out)
        out = self.EncResBlk5(out)
        out = self.DnSmp5(out)
        out = self.EncResBlk6(out)
        out = self.DnSmp6(out)
        out = self.EncResBlk7(out)
        
        """
        Classifier Section
        """        
        out = self.MaxPool(out)
        out = self.BN(out)
        out = self.relu(out)
        out = self.conv2dense(out)
        out = self.GlobalAvgPool2d(out)
        out = torch.flatten(out)
        out = self.dense_classifier(out)
        return out
