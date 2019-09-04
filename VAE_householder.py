# -*- coding: utf-8 -*-
"""
VAE with Householder normalizing flow.

Householder transformation is volume-preserving, so the type of distribution
does not change after the transformation. In the case of VAE the distribution
after transformation is still Gaussian but crucially with full covariance matrix.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import argparse
import math

# Output directory
output_dir = 'test/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

###############################
# Hyperparameters
###############################

# Dataset
NUM_TRAIN  = 50000
NUM_VAL    = 5000
batch_size = 128

# Number of units in each hidden layer (excl. latent code layer)
h_len = 120

# Length of latent code (number of units in latent code layer)
z_len = 100

# Normalizing flow
normalizing_flow = 'householder'
num_flow = 10

# Training
num_epochs = 100
lr         = 3e-4

###############################
# Helper functions
###############################
        
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images  = np.reshape(images, [images.shape[0], -1])
    sqrtn   = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs  = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    plt.show()
    
    return fig

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current model graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

###############################
# Dataset
###############################

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    
    Arguments:
        num_samples: Number of desired datapoints
        start:       offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


# MNIST dataset: 60,000 training, 10,000 test images.
# We'll take NUM_VAL of the training examples and place them into a validation dataset.
NUM_TRAIN  = 55000
NUM_VAL    = 5000

# Training set
mnist_train = dset.MNIST('C:/datasets/MNIST',
                         train=True, download=True,
                         # Converts (H x W x C) image in the range [0, 255] to a
                         # torch.FloatTensor of shape (C x H x W) in the range [0, 1].
                         transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
# Validation set
mnist_val = dset.MNIST('C:/datasets/MNIST',
                       train=True, download=True,
                       transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
# Test set
mnist_test = dset.MNIST('C:/datasets/MNIST',
                       train=False, download=True,
                       transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# Input dimensions
train_iter     = iter(loader_train)
images, labels = train_iter.next()
_, C, H, W     = images.size()
in_dim         = H * W          # Flatten

###############################
# Model
###############################

class VAE(nn.Module):
    def __init__(self, in_dim, h_len, z_len, normalizing_flow, num_flow):
        super().__init__()
                
        self.in_dim   = in_dim
        self.h_len    = h_len
        self.z_len    = z_len
        self.flow     = normalizing_flow
        self.num_flow = num_flow

        # Fully connected layers
        self.fc1   = nn.Linear(self.in_dim, self.h_len)
        self.fc2_1 = nn.Linear(self.h_len, self.z_len)    # 'Mean' layer
        self.fc2_2 = nn.Linear(self.h_len, self.z_len)    # 'Log variance' layer
        self.fc3   = nn.Linear(self.z_len, self.h_len)
        self.fc4   = nn.Linear(self.h_len, self.in_dim)
        
        # Fully connected layer for Householder vector
        if (normalizing_flow == 'householder'):
            self.fc_hhv = nn.ModuleList()
            self.fc_hhv.append(nn.Linear(self.h_len, self.z_len))
            for t in range(1, self.num_flow):
                self.fc_hhv.append(nn.Linear(self.z_len, self.z_len))

    def encoder(self, x):
        h1     = F.relu(self.fc1(x))
        mean   = self.fc2_1(h1)
        logvar = self.fc2_2(h1)
        return mean, logvar, h1
    
    def reparameterize(self, mean, logvar):
        sd  = torch.exp(0.5 * logvar)   # Standard deviation
        # We'll assume the posterior is a multivariate Gaussian
        eps = torch.randn_like(sd)      
        z   = eps.mul(sd).add(mean)
        return z
    
    def hh_transform(self, v_nxt, z):
        # Householder transform.
        # Vector-vector product of v. Conceptually, v should be a column vector
        # (ignoring the batch dimension) so the result of v.v_transformed is a
        # matrix (i.e. not a dot product).
        #  - v_nxt size = [batch_size, h_len]
        #  - v_nxt.unsqueeze(2) size = [batch_size, h_len, 1]
        #  - v_nxt.unsqueeze(1) size = [batch_size, 1, h_len]
        #  - v_mult size = [batch_size, h_len, h_len]
        v_mult = torch.matmul(v_nxt.unsqueeze(2), v_nxt.unsqueeze(1))
        # L2 norm squared of v, size = [batch_size, 1]
        v_l2_sq = torch.sum(v_nxt * v_nxt, 1).unsqueeze(1)
        
        z_nxt = z - 2 * (torch.matmul(v_mult, z.unsqueeze(2)).squeeze(2)) / v_l2_sq

        return z_nxt
    
    def householder_flow(self, h, z0):
        if (self.num_flow > 0):
            v    = [torch.zeros_like(h, requires_grad=True)] * self.num_flow
            z    = [torch.zeros_like(z0, requires_grad=True)] * self.num_flow
            v[0] = h
            z[0] = z0
            for t in range(1, self.num_flow):
                v[t] = self.fc_hhv[t-1](v[t-1])
                z[t] = self.hh_transform(v[t], z[t-1])
            return z[-1] 
        else:
            return z0
    
    def decoder(self, z):
        h2 = F.relu(self.fc3(z))
        # For binarised image use sigmoid function to get the probability
        x  = torch.sigmoid(self.fc4(h2))
        return x

    # Note this takes flattened images as input
    def forward(self, x_flat):
        mean, logvar, h = self.encoder(x_flat)
        z0 = self.reparameterize(mean, logvar)
        
        if (self.flow == 'householder'):
            z = self.householder_flow(h, z0)
        
        x_recon = self.decoder(z)
        
        return x_recon, mean, logvar, z0, z

###############################
# Loss function
###############################

# The Evidence Lower Bound (ELBO) gives the negative loss, so
# minimising the loss maximises the ELBO
def loss_fn(x_original, x_recon, mean, logvar, z_start, z_end):
    # Reconstruction loss
    # Each pixel is a Bernoulli variable (black and white image), so we use
    # binary cross entropy. For each batch we sum the losses from every image
    # in that batch.
    recon_loss = F.binary_cross_entropy(x_recon, x_original, reduction='sum')

    # Log of prior distribution p(z) at the end of flow (zero-mean Gaussian with
    # full covariance matrix). Constant log(2*pi) has been dropped since it'll
    # get cancelled out in KL divergence anyway.
    log_p = -0.5 * torch.sum(torch.pow(z_end, 2), 1)
    # Log of approx. posterior. q(z|x) at the start of flow (Gaussian with diagonal
    # covariance matrix). Constant log(2*pi) has been dropped.
    log_q = -0.5 * torch.sum(logvar + torch.pow(z_start - mean, 2) / logvar.exp(), 1)
    # KL divergence
    # REVISIT: KL divergence should be the expectation of (log_q - log_p) w.r.t. q(z)
    # REVISIT: The official implementation seems to use Monte Carlo estimate of the
    # REVISIT: expectation with a sample size of 1. (This may sound very inaccurate
    # REVISIT: but the reconstruction error is also using Monte Carlo estimate of
    # REVISIT: expectation with just 1 sample)
    KL_loss = torch.sum((log_q - log_p))
    
    return recon_loss + KL_loss
    
###############################
# Main
###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae       = VAE(in_dim, h_len, z_len, normalizing_flow, num_flow).to(device)
optimiser = optim.Adam(vae.parameters(), lr=lr)

def train(epoch):
    vae.train()
    loss_train = 0
    for batch, (x, _) in enumerate(loader_train):
        # Reshape (flatten) input images
        x_flat = x.view(x.size(0), -1).to(device)
        
        x_recon, mean, logvar, z_start, z_end = vae(x_flat)
        loss        = loss_fn(x_flat, x_recon, mean, logvar, z_start, z_end)
        loss_train += loss.item()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    print('Epoch {} avg. training loss: {:.3f}'.format(epoch, loss_train / len(loader_train.dataset)))

def validation(epoch):
    vae.eval()
    loss_val = 0
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader_val):
            # Reshape (flatten) input images
            x_flat = x.view(x.size(0), -1).to(device)

            x_recon, mean, logvar, z_start, z_end = vae(x_flat)
            loss      = loss_fn(x_flat, x_recon, mean, logvar, z_start, z_end)
            loss_val += loss.item()
            
    print('Epoch {} validation loss: {:.3f}'.format(epoch, loss_val / len(loader_val.dataset)))

for epoch in range(num_epochs):
    train(epoch)
    validation(epoch)
    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(128, z_len).to(device)
            # REVISIT: The decoder output produced by sigmoid function is the
            # REVISIT: mean of Bernoulli distribution for each pixel. Strictly
            # REVISIT: speaking, we should sample from this distribution to get
            # REVISIT: binarised images, but below it is treated as gray scale
            # REVISIT: pixel value in the range [0, 1] instead.
            sample = vae.decoder(z)
            imgs_numpy = sample.cpu().numpy()
            fig = show_images(imgs_numpy[0:16])
            # Save image to disk
            fig.savefig('{}/{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)