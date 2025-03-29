import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader

import MNIST.autoencoder as autoencoderMNIST

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    #parser.add_argument('--self-supervised', action='store_true', default=False,
    #                   help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    
    # ADDED flag: --task 
    parser.add_argument('--task', type=str, choices=['1.2.1', '1.2.2', '1.2.3'], required=True,
                        help='Choose task: 1.2.1 (self-supervised, reconstruction objective), 1.2.2 (Classification-Guided Encoding), 1.2.3 (Structured Latent Spaces)')
    return parser.parse_args()
    

def run_1_2_1(dl_train, dl_val, dl_test, dataset_type, device):
    
    if dataset_type == "MNIST":
        
    
def run_1_2_2(dl_train, dl_val, dl_test, dataset_type, device):

    
def run_1_2_3(dl_train, dl_val, dl_test, dataset_type, device):


    
if __name__ == "__main__":

    # =====================
    # Load Data
    # =====================
    transform = transforms.Compose([
        transforms.ToTensor(),
        ## SEGEL: one possible convenient normalization: You don't have to use it
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    args = get_args()
    freeze_seeds(args.seed)
                                           
    if args.mnist:
        ds_train = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        ds_test = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
        dataset_type="MNIST"
    else:
        ds_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        ds_test = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        dataset_type="CIFAR10"
        
    ## SEGEL: When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation 

   # Split full_train_dataset into train (90%) and validation (10%)
    split_lengths = [int(len(dataset) * 0.9), int(len(dataset) * 0.1)]
    ds_train, ds_val = random_split(ds_train, split_lengths)

    # Create DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)


    '''
    SEGEL:
    #this is just for the example. Simple flattening of the image is probably not the best idea                                        
    encoder_model = torch.nn.Linear(32*32*3,args.latent_dim).to(args.device)
    decoder_model = torch.nn.Linear(args.latent_dim,32*32*3 if args.self_supervised else NUM_CLASSES).to(args.device) 

    sample = train_dataset[0][0][None].to(args.device) #This is just for the example - you should use a dataloader
    output = decoder_model(encoder_model(sample.flatten()))
    print(output.shape)
    '''

    
    # Run the selected task
    if args.task == "1.2.1":
        run_1_2_1(dl_train, dl_val, dl_test, dataset_type, args.device)
    elif args.task == "1.2.2":
        run_1_2_2(dl_train, dl_val, dl_test, dataset_type, args.device)
    elif args.task == "1.2.3":
        run_1_2_3(dl_train, dl_val, dl_test, dataset_type, args.device)
    else:
        raise ValueError(f"Unknown task: {args.task}"


'''
"READ ME":
example for how to run:

# Run Self-Supervised Autoencoding (1.2.1)
python main.py --task 1.2.1 --mnist --batch-size 64

# Run Classification-Guided Encoding (1.2.2)
python main.py --task 1.2.2 --mnist --batch-size 64

# Run Structured Latent Spaces (1.2.3)
python main.py --task 1.2.3 --mnist --batch-size 64
'''

