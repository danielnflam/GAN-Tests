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

def criterion_TotalVariation(tensor):
    img_height = tensor.size()[2]
    img_width = tensor.size()[3]
    a = torch.square(tensor[:, :, :img_height - 1, :img_width - 1] - tensor[:, :, 1:, :img_width           - 1] )    
    b = torch.square(tensor[:, :, :img_height - 1, :img_width - 1] - tensor[:, :, :img_height -           1, 1:])    
    
    return torch.sum((a+b)**0.5,(-1,-2,-3))

def criterion_StyleReconstruction_layer(representation_fake, representation_real, reduction="mean"):
    N_f = representation_fake.size()[0]
    C_f = representation_fake.size()[1]
    H_f = representation_fake.size()[2]
    W_f = representation_fake.size()[3]
    
    N_r = representation_real.size()[0]
    C_r = representation_real.size()[1]
    H_r = representation_real.size()[2]
    W_r = representation_real.size()[3]
    
    # reshape representations into C x HW tensors
    representation_fake = torch.reshape(representation_fake,(N_f,C_f,H_f*W_f))
    representation_real = torch.reshape(representation_real,(N_r,C_r,H_r*W_r))
    def calculate_Gram_Matrix(representation, C, H, W):
        return torch.matmul(representation, torch.transpose(representation,-1,-2))/(C*H*W)
    G_fake = calculate_Gram_Matrix(representation_fake, C_f, H_f, W_f)
    G_real = calculate_Gram_Matrix(representation_real, C_r, H_r, W_r)
    G_difference = G_real - G_fake
    
    # Frobenius loss squared:
    matrix = torch.matmul(G_difference, torch.transpose(torch.conj(G_difference),-1,-2))
    print(matrix.shape)
    out = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) 
    return out

def criterion_Perceptual_layer(representation_fake, representation_real, reduction="mean"):
    """Extract feature representations from layers of the discriminator.  Use for 4D tensors only (NxCxHxW).  Output is size N"""
    
    diff = torch.abs(representation_fake - representation_real)
    if reduction=="mean":
        out = torch.mean(diff,-1)
        out = torch.mean(out,-1)
        out = torch.mean(out,-1)
    elif reduction=="sum":
        out = torch.sum(diff,-1)
        out = torch.sum(out,-1)
        out = torch.sum(out,-1)
    return out

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

def criterion_MSELoss(input_image, target_image):
    loss = nn.MSELoss()
    return loss(input_image, target_image)