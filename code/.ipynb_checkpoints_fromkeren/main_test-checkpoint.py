import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse
from torch.utils.data import DataLoader, Subset, random_split
import os

import torch.nn as nn
import torch.optim as optim
import MNIST.autoencoder as autoencoder

from utils import plot_tsne

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    freeze_seeds(42)

    # normalize data - prevents exploding/vanishing gradients
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust based on dataset
    ])

    dataset_path = '/datasets/cv_datasets/data'
    batch_size = 128

    # Load MNIST from path
    ds_train = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transform)
    ds_val = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transform)

    # Split training into 80% train and 20% test
    train_size = int(0.8 * len(ds_train))  
    test_size = len(ds_train) - train_size
    ds_train, ds_test = random_split(ds_train, [train_size, test_size])

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = autoencoder.Encoder().to(device)
    decoder = autoencoder.Decoder().to(device)
    autoencoder_model = autoencoder.Autoencoder(encoder, decoder).to(device)

    loss_fn = nn.L1Loss() # MAE loss 
    optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)
    num_epochs = 20

    trainer = autoencoder.autoencoderTrainer(autoencoder_model, dl_train, dl_test, loss_fn, optimizer, num_epochs, device)
    trainer.trainAutoencoder()
    plot_tsne(encoder, dl_test, device)

    