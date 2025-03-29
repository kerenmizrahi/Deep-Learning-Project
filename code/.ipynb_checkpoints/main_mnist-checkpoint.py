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
from utils import plot_tsne

import autoencoders 
import Trainers
import classifiers as clf


def mnist_121(dl_train, dl_val, dl_test, device):

    print("Initializing Autoencoder...\n")
    encoder = autoencoders.Encoder_mnist().to(device)
    decoder = autoencoders.Decoder_mnist().to(device)
    #autoencoder_model = autoencoder.Autoencoder(encoder, decoder).to(device)

    print("Training AutoEncoder...\n")
    loss_fn = nn.L1Loss() # mean absolute error (reconstruction error) 
    #optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),  lr=0.001)
    num_epochs = 20

    
    #trainer = autoencoder.autoencoderTrainer(autoencoder_model, dl_train, dl_test,
     #                                        loss_fn, optimizer, num_epochs, device)
    trainer = Trainers.autoencoderTrainer(encoder, decoder, dl_train, dl_test, loss_fn, optimizer, num_epochs, device)
    trainer.trainAutoencoder()

    print("Initializing classifier (using pre-trained encoder)...")
    classifier = clf.Classifier(hidden_layers=[512, 256, 128, 64], dropout_rate=0.1).to(device)

    print("Training Classifier (fully-supervised training + frozen encoder)...")
    hyperparams = {
        'loss_fn': torch.nn.CrossEntropyLoss(),
        'optimizer' : optim.Adam,
        'weight_decay' : 0.0,
        'learning_rate': 0.001,  
        'num_epochs': 5,            
        }

        freeze_encoder = True

    trainer = clf.clfTrainer(classifier, encoder, dl_train, dl_test,
                             hyperparams, freeze_encoder, device)
    train_acc, test_acc = trainer.trainClassifier()

    print("Evaluate Classifier on Validation...")
    val_loss, val_accuracy = trainer.evalClassifier(dl_val)
    
    