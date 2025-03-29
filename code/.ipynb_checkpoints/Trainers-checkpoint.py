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


def freeze_encoder(encoder):
    # Freeze encoder parameters
    for param in encoder.encoder.parameters():
        param.requires_grad = False  


# self-supervised
class autoencoderTrainer(nn.Module):
    def __init__(self, encoder, decoder, dl_train, dl_test, loss_fn, optimizer, num_epochs, device):
    # def __init__(self, model, dl_train, dl_test, loss_fn, optimizer, num_epochs, device):
        super().__init__()
    
        #self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.loss_fn = loss_fn  
        self.optimizer = optimizer 
        self.num_epochs = num_epochs
        self.device = device

     
    def trainAutoencoder(self):    
        
        for epoch in range(self.num_epochs):
            #self.model.train(True)
            #self.model.encoder.train()
            #self.model.decoder.train()
            self.encoder.train()
            self.decoder.train()
            train_loss = 0.0

            for img, _ in self.dl_train:
                img = img.to(self.device)
                #self.optimizer.zero_grad()
                encoded_img = self.encoder(img)
                #encoded_img = encoded_img.view(encoded_img.size(0), -1) !!!!
                reconstructed_img = self.decoder(encoded_img)
                '''
                reconstructed_img = self.model(img)  # Forward: encoder + decoder
                '''
                # FOR DEBUG: print(reconstructed_img.shape)
                #latent = self.model.encoder(img)
                #reconstructed_img = self.model.decoder(latent)
                loss = self.loss_fn(reconstructed_img, img)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() 

            
            train_loss /= len(self.dl_train)  # Average loss 

            # evaluate the model on test set - Set model to evaluation mode
            # self.model.eval()
            '''
            self.model.encoder.eval()
            self.model.decoder.eval()
            '''
            self.encoder.eval()
            self.decoder.eval()
            test_loss = 0.0
   
            with torch.no_grad(): 
                for img, _ in self.dl_test: 
                    img = img.to(self.device)
                    encoded_img = self.encoder(img)
                    encoded_img = encoded_img.view(encoded_img.size(0), -1)
                    reconstructed_img = self.decoder(encoded_img)
                    '''
                    reconstructed_img = self.model(img)  # Forward: encoder + decoder
                    '''
                    loss = self.loss_fn(reconstructed_img, img)
                    test_loss += loss.item()
                 
            test_loss /= len(self.dl_test)   

            print(f"Epoch {epoch + 1}:")
            print(f"    train reconstruction error: {train_loss:.4f}")
            print(f"    Test reconstruction error: {test_loss:.4f}")
  

        print(f"\n reconstruction error (mean absolute error, for last epoch): {test_loss:.4f}")
        