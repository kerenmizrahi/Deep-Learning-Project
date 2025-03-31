import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import plot_tsne
import torchvision.models as models
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

# --------------------- contrastive learning - SimCLR (task 1.2.3) ------------------------

class Encoder_ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.resnet = models.resnet18(weights=None)  # Load the standard ResNet-18
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # Change input layer
        self.resnet.maxpool = nn.Identity()  # Remove maxpool layer
        #self.resnet.fc = nn.Linear(512, num_classes)  # Change final layer for CIFAR-10
        
        self.projection_head = nn.Sequential(
            nn.Linear(512, 2*128),  # MobileNetV3-Small output is 576, changed 4->2
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(2*128, 128)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.projection_head(x)  
        return x
        

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, out_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        return x


class ResNet18Encoder(nn.Module):
    def __init__(self, out_dim=128, pretrained=False, device='cpu'):
        super().__init__()
        # Load ResNet-18 backbone
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Remove the final classification layer (fully connected layer)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Removes the FC layer

        # Projection head
        self.projection_head = ProjectionHead(in_dim=512, out_dim=out_dim)

        self.device = device

    def forward(self, x):
        x = x.to(next(self.parameters()).device)  # Ensure input is on the same device
        features = self.encoder(x)  # Shape: [batch_size, 512, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten to [batch_size, 512]
        projections = self.projection_head(features)  # Pass through Projection Head
        return projections  # Output of size [batch_size, 128]

      
class Encoder_cifar10(nn.Module):
    def __init__(self):
        super().__init__()
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
            #nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            # (N, 512, 8, 8)
            #nn.ReLU(),
            #nn.BatchNorm2d(512),
            #-----------------------
            # (N, 64, 8, 8)
            #nn.Dropout(p=0.5), 
            nn.Flatten(),
            # (N, 64*8*8)
            # I changed 64 ->128 ->256 ->512
            nn.Linear(256 * 8 * 8, 80),
            #nn.Linear(512 * 16 * 16, 128),
            # (N, 128)
            nn.ReLU(),
        )

    
    def forward(self, x):
        return self.encoder(x)


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
    
        # cosine similarity - matrix of similarity between each pair 
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        #out = torch.cat([z1, z2], dim=0)
        #out1, out2 = torch.split(out, z1.size(0), dim=0)
        #n_samples = len(out)
        
        similarity_mat = torch.matmul(z1, z2.T) 
        similarity_mat /= self.tmp
        
        #cov = torch.mm(out, out.t().contiguous())
        #sim = torch.exp(cov / self.tmp)

        # recude val of diagonal =similarity between a 2 views of same orig img
        i_mat = torch.eye(similarity_mat.size(0)).to(similarity_mat.device) * 1e8
        similarity_mat = similarity_mat - i_mat 

        # Mask out the diagonal (same image comparison)
        mask = torch.eye(similarity_mat.size(0), device=similarity_mat.device).bool()
        similarity_mat = similarity_mat.masked_fill(mask, -1e8)  # A large negative number

        #mask = ~torch.eye(n_samples, device=sim.device).bool()
        #neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        #pos = torch.exp(torch.sum(out1 * out2, dim=-1) / self.tmp)
        #pos = torch.cat([pos,pos], dim=0)
        
        # Create labels for positive pairs (same image)
        labels = torch.arange(similarity_mat.size(0), device=similarity_mat.device)
        
        # CHECK
        #labels = torch.arange(similarity_mat.size(0)).to(similarity_mat.device) 
        #labels[::2] += 1 
        #labels[1::2] -= 1  
        
        loss = F.cross_entropy(similarity_mat, labels)
        #epsilon = 1e-8
        #loss = -torch.log(pos / (neg + epsilon)).mean()
        #loss = -torch.log(pos / neg).mean()
        return loss


        

def contrastive_train( encoder, pair_dl_train, pair_dl_test, loss_fn, optimizer, num_epochs, device):
    
    for epoch in range(num_epochs):
        total_train_loss = 0
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

            total_train_loss += loss.item()

        encoder.eval()
        total_test_loss = 0
        with torch.no_grad():
            for view1, view2, _ in pair_dl_test:
                view1, view2 = view1.to(device), view2.to(device)
                z1 = encoder(view1)
                z2 = encoder(view2)
                loss = loss_fn(z1, z2)
                total_test_loss += loss.item()
        
        print(f"Epoch {epoch + 1}:")
        print(f"    train Contrastive Loss: {total_train_loss / len(pair_dl_train):.4f}")
        print(f"    test Contrastive Loss: {total_test_loss / len(pair_dl_test):.4f}")
        #print(f"Epoch [{epoch+1}/{num_epochs}] - Contrastive Loss: {total_loss / len(pair_dl_train):.4f}")



