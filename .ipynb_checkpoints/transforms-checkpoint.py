import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, sys, time, datetime


"""
File contains transformations as callable classes.

"""

class ToTensor(object):
    """
    Torch transforms generally work on Torch tensor datasets best.
    This transform will turn a numpy ndarray into a torch tensor.
    
    No init -- this is a standard function.
    """
    def __init__(self, sample_keys):
        self.sample_keys_images = sample_keys
    def __call__(self, sample):
        """
        Input:
            sample (dict): the dictionary containing the images to be transformed
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            image = image.astype(np.float32)
            sample[key_idx] = torch.from_numpy(image[np.newaxis,:])
            
        return sample

class NormaliseBetweenPlusMinus1(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            # Rescale to between 0 and 1
            image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
            # Rescale to between -1 and 1
            image = (image*2 - 1)
            sample[key_idx] = image
        return sample

class RandomIntensityFlip(object):
    # Black becomes white and white becomes black
    def __init__(self, sample_keys_images, probability=0.5):
        self.probability = probability
        self.sample_keys_images = sample_keys_images
        
    def __call__(self, sample):
        if np.random.rand(1) < self.probability:
            for key_idx in self.sample_keys_images:
                image = sample[key_idx]
                max_image = np.amax(image)                
                flipped_image = np.abs(max_image - image)
                sample[key_idx] = flipped_image
        
        return sample
    
class IntensityJitter(object):
    """
    Scale the intensity of the input image randomly by a factor that is randomly chosen between the rescale_factor_limits.
    
    Images input are numpy NDARRAYS
    """
    def __init__(self, sample_keys_images, rescale_factor_limits=(0.5,1.0)):
        self.sample_keys_images = sample_keys_images
        self.rescale_factor_limits = rescale_factor_limits
    def __call__(self, sample):
        # Generate the same factor for all images denoted by the sample_keys
        factor = np.random.rand(1)*(max(self.rescale_factor_limits) - min(self.rescale_factor_limits)) + min(self.rescale_factor_limits)
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            sample[key_idx] = factor*image
        return sample

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    This effectively resamples the image to fit that given output size.
    
    Input integer decides the number of column pixels.
    
    To keep the aspect ratio the same, use an integer as the input when initialising this object.
    """
    def __init__(self, output_size, sample_keys_images, sample_key_PixelSize):
        """
        Inputs:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, that will be the number of column pixels, 
            and the number of row pixels is determined from the aspect ratio
            sample_keys_images (list or tuple): list of strings representing the keys to images in the sample_dictionary
            sample_key_PixelSize (string): string for the key holding the PixelSize values in the sample dict.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.sample_keys_images = sample_keys_images
        self.sample_key_PixelSize = sample_key_PixelSize
        
    def __call__(self, sample):
        """
        Inputs:
            sample (dict): the dictionary containing the images to be transformed
                            Images should be numpy arrays in the format: [H x W]
                            B is batch number, C is channels, H is number of rows, W is number of columns
                            The "ToTensor" function can do this.
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            PixelSize = sample[self.sample_key_PixelSize]
            
            if len(image.shape) == 2:
                h, w = image.shape[:2]
                
            if isinstance(self.output_size, int):
                new_h, new_w = self.output_size* h / w, self.output_size
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            if len(image.shape)==2:
                # Skimage transform
                out = transform.resize(image, (new_h, new_w), order=0)
            
            sample[key_idx] = out
            
            # Output the rescaled PixelSize
            if PixelSize == None:
                out_PixelSize = None
            else:
                out_PixelSize = (PixelSize[0]*(h/new_h), PixelSize[1]*(w/new_w))
            sample[self.sample_key_PixelSize] = out_PixelSize
        return sample

class RescalingNormalisation(object):
    def __init__(self, sample_keys_images):
        """
        Inputs:
            sample_keys_images (list or tuple): list of strings representing the keys to images in the sample_dictionary
        """
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        """
        Inputs:
            sample (dict): the dictionary containing the images to be transformed
                            Images should be numpy arrays in the format [H x W]
                            B is batch number, C is channels, H is number of rows, W is number of columns
                            The "ToTensor" function can do this.
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            sample[key_idx] = (image - np.amin(image))/(np.amax(image) - np.amin(image))
        return sample
        
class ImageComplement(object):
    """
    Inverse the grayscale image.  White becomes black and vice versa.
    """
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        """
        Image in sample_dict assumed to be a numpy array [H x W]
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            sample[key_idx] = np.abs(np.amax(image)-image) # Dark is low intensity, bright is high intensity
        return sample
    
class Random180(object):
    """
    Randomly flip image via horizontal axis and then vertical axis.
    This ensures that the heart points towards the left side of the body.
    Probability: the % chance that the flip occurs.  Probability = 0.6 means that a flip occurs 60% of the time.
    """
    def __init__(self, sample_keys_images, probability=0.5):
        self.sample_keys_images = sample_keys_images
        self.probability = probability
    def __call__(self, sample):
        if np.random.rand(1) < self.probability:
            for key_idx in self.sample_keys_images:
                image = sample[key_idx]
                image = np.flip(image, 0)
                image = np.flip(image, 1)
                sample[key_idx] = image
        
        return sample