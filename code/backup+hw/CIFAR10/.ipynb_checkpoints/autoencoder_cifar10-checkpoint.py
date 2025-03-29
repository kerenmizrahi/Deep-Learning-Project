import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import plot_tsne
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #  N batches
        #  CIFAR10 input shape: (N, 3, 32, 32)

        self.encoder = nn.Sequential(
            # (N, 3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # (N, 32, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (N, 32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            # (N, 64, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (N, 64, 8, 8)
            #added------------------
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            # (N, 128, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            # (N, 256, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            # (N, 512, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #-----------------------
            # (N, 64, 8, 8)
            #nn.Dropout(p=0.5), 
            nn.Flatten(),
            # (N, 64*8*8)
            # I changed 64 ->128 ->256 ->512
            nn.Linear(512 * 8 * 8, 128),
            #nn.Linear(512 * 16 * 16, 128),
            # (N, 128)
            nn.ReLU(),
        )

    
    def forward(self, x):
        return self.encoder(x)

   
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #  N batches
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
        '''
        self.decoder = nn.Sequential(
            # (N, 128)
            nn.Linear(128, 512 * 8 * 8), # (128, 512 * 8 * 8)
            nn.LeakyReLU(),
            nn.Unflatten(1, (512, 8, 8)), # (N, 512, 8, 8)
    
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), # (N, 256, 8, 8)
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
    
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 8, 8)
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
    
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (N, 64, 16, 16)
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
    
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output should be (N, 3, 32, 32)
            nn.Tanh()  # Normalize output to [-1, 1] - images are normalized
        )
        
    
    def forward(self, x):
        return self.decoder(x)
    
'''            
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
       
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''    
        
# --------------------------- contrastive learning - SimCLR (task 1.2.3) -------------------------------

class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # (N, 3, 32, 32)
        self.base_encoder = models.resnet18(pretrained=False)
        # CONTINUE #
        
        

    def forward(self, x):
        # IMPLEMENT # 
        return x

class PairDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset 
        self.transform = transform    
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        view1 = self.transform(image) if self.transform else image
        view2 = self.transform(image) if self.transform else image
        return view1, view2, label



class NTXentLoss(torch.nn.Module):
    def __init__(self, tmp=0.5):
        super().__init__()
        self.tmp = tmp
    
    def forward(self, z1, z2):
        # N batches
        # z1 = (N , latent_space = 128)  encoded view1
        # z2 = (N , latent_space = 128)  encoded view2
    
        # cosine similarity - matrix of similarity between each pair 
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        similarity_mat = torch.matmul(z1, z2.T)  # (pair_batch_size, pair_batch_size) = (2N, 2N)
        similarity_mat /= self.tmp

        # recude val of diagonal =similarity between a 2 views of same orig img
        i_mat = torch.eye(similarity_mat.size(0)).to(similarity_mat.device) * 1e8
        similarity_mat = similarity_mat - i_mat 

        # CHECK
        labels = torch.arange(similarity_mat.size(0)).to(similarity_mat.device) 
        labels[::2] += 1 
        labels[1::2] -= 1  
        
        loss = F.cross_entropy(similarity_mat, labels)
        return loss


        

def contrastive_train( encoder, pair_dl_train, pair_dl_test, loss_fn, optimizer, num_epochs, device):
    
    for epoch in range(num_epochs):
        total_loss = 0
        encoder.train() 

        for view1, view2, _ in pair_dl_train:
            #print(f"View1 type: {type(view1)}, View2 type: {type(view2)}")
            view1, view2 = view1.to(device), view2.to(device)
            z1 = encoder(view1)
            z2 = encoder(view2)
            optimizer.zero_grad()
            loss = loss_fn(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

           
        # ADD FOR PAIR_DL_TEST
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - train Loss: {total_loss / len(pair_dl_train):.4f}")
        
         




