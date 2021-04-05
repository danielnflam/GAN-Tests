import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, sys, time, datetime
from torchvision.transforms.functional import InterpolationMode


"""
File contains transformations as callable classes.

"""

class ToTensor(object):
    """
    Torch transforms generally work on Torch tensor datasets best.
    This transform will turn a numpy ndarray into a torch tensor.
    
    No init -- this is a standard function.
    """
    def __init__(self):
        self.desc = "Numpy ndarray to Torch Tensor"
    def __call__(self, image):
        image = image[np.newaxis, np.newaxis,:]
        return torch.from_numpy(image)

class StandardiseSize(object):
    """
    Rescale the image in a sample to a given size.
    """
    def __init__(self, output_size):
        """
        Inputs:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, input_image):
        """
        Inputs:
            input_image (Torch Tensor): a 4D input image [B x C x H x W]
                                        B is batch number, C is channels, H is number of rows, W is number of columns
        """
        image = input_image
        
        if len(image.shape) == 4:
            h, w = image.shape[2:4]
        elif len(image.shape) == 2:
            h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        
        if len(image.shape)==4:
            resize = transforms.Resize((new_h, new_w), interpolation=InterpolationMode.NEAREST)        
            out = resize(image)
        elif len(image.shape)==2:
            out = transform.resize(image, (new_h, new_w), order=0)
        return out
class RandomCrop