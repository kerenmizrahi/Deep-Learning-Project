import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import plot_tsne
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #  N batches
        #  MNIST input shape: (N, 1, 28, 28)   

        self.encoder = nn.Sequential(
            # (N, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            # (N, 64, 28, 28)
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Flatten(),
            # (N, 64*28*28)
            nn.Linear(64 * 28 * 28, 128),
            # (N, 128)
            nn.ReLU(),
        )

    
    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # (N, 128)
            nn.Linear(128, 64 * 28 * 28),
            # (N, 64*7*7)
            nn.ReLU(),
            nn.Unflatten(1, (64, 28, 28)),
            # (N, 64, 28, 28)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), 
            # (N, 32, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            # (N, 1, 28, 28)
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


# --------------------- contrastive learning - SimCLR (task 1.2.3) ------------------------


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

        progress_bar = tqdm(pair_dl_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
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

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        # ADD FOR PAIR_DL_TEST
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - Contrastive Loss: {total_loss / len(pair_dl_train):.4f}")
        
         



