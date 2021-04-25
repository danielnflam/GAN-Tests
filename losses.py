import torch
import numpy as np
import os, sys, time, datetime, random, math
import torch.nn as nn
import pytorch_msssim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# Perceptual Loss
"""
Uses a discriminator network to transform the image into a feature vector.
Find the mean L1 distance between feature maps of translated & true images.
"""

# Gusarev Loss
def criterion_Gusarev(testImage, referenceImage, alpha=0.84):
    """
    Gusarev et al. 2017. Deep learning models for bone suppression in chest radiographs.  IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology.
    """
    mseloss = nn.MSELoss() # L2 used for easier optimisation

    msssim = pytorch_msssim.MSSSIM(window_size=11, size_average=True, channel=1, normalize="relu")
    msssim_loss = 1 - msssim(testImage, referenceImage)
    total_loss = (1-alpha)*mseloss(testImage, referenceImage) + alpha*msssim_loss
    return total_loss


# IF USING BCEWithLogitsLoss, do NOT use Sigmoid as output activation for Discriminator
def criterion_Pix2Pix_WithLogitsLoss(input_image, ground_truth_image, input_label, ground_truth_label, reg_lambda=100):
    L_cGAN = nn.BCEWithLogitsLoss(reduction="mean")
    L_l1 = nn.L1Loss(reduction="mean")
    out = L_cGAN(input_label , ground_truth_label) + reg_lambda*L_l1(input_image , ground_truth_image)
    return out

# Pixel Distance Loss
def criterion_L1Loss(input_image, ground_truth_image):
    L_l1 = nn.L1Loss(reduction="mean")
    return L_l1(input_image , ground_truth_image)

# Adversarial Loss
def criterion_BCEWithLogitsLoss(input_label, ground_truth_label):
    L_cGAN = nn.BCEWithLogitsLoss(reduction="mean")
    out = L_cGAN(input_label, ground_truth_label)
    return out
def criterion_BCELoss(input_label, ground_truth_label):
    L_cGAN = nn.BCELoss(reduction="mean")
    out = L_cGAN(input_label, ground_truth_label)
    return out   

def Wasserstein_G(discriminator_output_for_generated_images):
    return -discriminator_output_for_generated_images.mean()
def Wasserstein_D(discriminator_output_for_real_images, discriminator_output_for_generated_images):
    return (discriminator_output_for_generated_images - discriminator_output_for_real_images).mean()