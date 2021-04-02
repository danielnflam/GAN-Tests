import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys, time, datetime

"""
File contains transformations as callable classes.

"""

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
    def __call__(self, sample_dict, sample_key):
        """
        sample_dict: output from the dataloader
        sample_key (string): key
        """
        image = sample_dict[sample_key]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):