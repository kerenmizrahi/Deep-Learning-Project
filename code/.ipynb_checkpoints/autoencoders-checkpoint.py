import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import plot_tsne
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image


class Encoder_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        #  N batches
        #  MNIST input shape: (N, 1, 28, 28)   
        self.encoder = nn.Sequential(
            # (N, 1, 28, 28)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # (N, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (N, 256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Dropout(p=0.5), 
            nn.Flatten(), # (N, 256*28*28)
            
            nn.Linear(256 * 28 * 28, 128), # (N, 128)
            nn.ReLU(),
        )

    
    def forward(self, x):
        return self.encoder(x)
    

class Decoder_mnist(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            # (N, 128)
            nn.Linear(128, 256 * 28 * 28),  # (N, 256*28*28)
            nn.ReLU(),
            nn.Unflatten(1, (256, 28, 28)), # (N, 64, 28, 28)
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), # (N, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1), # (N, 1, 28, 28)
            nn.Tanh()
            
        )
    
    def forward(self, x):
        return self.decoder(x)
    


class Encoder_cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        #  N batches
        #  CIFAR10 input shape: (N, 3, 32, 32)
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=3, padding=1), # (N,64, 11, 11)
            #nn.ReLU(True),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2), #(N, 64, 5, 5)
            nn.Conv2d(64, 3, 3, stride=2, padding=1), #(N, 3, 3, 3)
            nn.GELU(),
            nn.MaxPool2d(2, stride=1), # (N, 3, 2, 2)
            nn.Flatten(), # (N, 64*2*2)
            nn.Linear(3 * 2 * 2, 128), 
            nn.GELU(),
        )
        
        
        BACKUP #2
        '''
        self.encoder = nn.Sequential(
            # (N, 3, 32, 32)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # (N, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (N, 256, 28, 28)
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Dropout(p=0.5), 
            nn.Flatten(), # (N, 256*28*28)
            
            nn.Linear(256 * 32 * 32, 128), # (N, 128)
            nn.GELU(),
        )
        
        '''
        
        BACKUP #1
        self.encoder = nn.Sequential(
            # (N, 3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # (N, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (N, 32, 16, 16)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (N, 64, 8, 8)
            
            #added------------------
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (N, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            #nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # (N, 512, 8, 8)
            #nn.ReLU(),
            #nn.BatchNorm2d(512),
            #-----------------------
            # (N, 64, 8, 8)
            #nn.Dropout(p=0.5), 
            nn.Flatten(),
            # (N, 64*8*8)
            # I changed 64 ->128 ->256
            nn.Linear(256 * 8 * 8, 128),
            #nn.Linear(512 * 16 * 16, 128),
            # (N, 128)
            nn.ReLU(),
        )
        '''

    
    def forward(self, x):
        return self.encoder(x)

   
class Decoder_cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        #  N batches

        self.decoder = nn.Sequential(
            nn.Linear(128, 3 * 2 * 2),
            nn.Unflatten(1, (3, 2, 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 64, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Tanh()
        )
        
        '''
        BACKUP
        self.decoder = nn.Sequential(
            # (N, 128)
            nn.Linear(128, 64 * 8 * 8),
            # (N, 64*8*8)
            #nn.Linear(128, 128 * 8 * 8),
            #nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            # (N, 128, 8, 8)
            #nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            # (N, 64, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            # (N, 32, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            # (N, 3, 32, 32)
        )
        

        BACKUP #2 
        self.decoder = nn.Sequential(
            # (N, 128)
            nn.Linear(128, 256 * 8 * 8), # (128, 256 * 8 * 8)
            nn.LeakyReLU(),
            nn.Unflatten(1, (256, 8, 8)), # (N, 256, 8, 8)
    

            #nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), # (N, 256, 8, 8)
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(256),
    
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 8, 8)
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
    
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (N, 64, 16, 16)
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
    
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output should be (N, 3, 32, 32)
            nn.Tanh()  # Normalize output to [-1, 1] - images are normalized
        )
        
        '''
    
    def forward(self, x):
        return self.decoder(x)

    

       
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
       
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
