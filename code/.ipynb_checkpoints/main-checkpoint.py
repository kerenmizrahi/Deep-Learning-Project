import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, Subset, random_split
import main_cifar10 as run_cifar10
import main_mnist as run_mnist

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_args():   
    parser = argparse.ArgumentParser()
    #parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    #parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    #parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    #parser.add_argument('--self-supervised', action='store_true', default=False,
    #                   help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    
    # ADDED flag: --task 
    parser.add_argument('--task', type=str, choices=['1.2.1', '1.2.2', '1.2.3'], required=True,
                        help='Choose task: 1.2.1 , 1.2.2 , 1.2.3 ')
    return parser.parse_args()
    

def run_1_2_1(dl_train, dl_val, dl_test, dataset_type, device):
    
    if dataset_type == "MNIST":
        run_mnist.mnist_121(dl_train, dl_val, dl_test, device)
    else:
        run_cifar10.cifar10_121(dl_train, dl_val, dl_test, device)
        
    
def run_1_2_2(dl_train, dl_val, dl_test, dataset_type, device):

    if dataset_type == "MNIST":
        run_mnist.mnist_122(dl_train, dl_val, dl_test, device)
    else:
        run_cifar10.cifar10_122(dl_train, dl_val, dl_test, device)

    
def run_1_2_3(dl_train, dl_val, dl_test, dataset_type, device):
    pass


    
if __name__ == "__main__":

    args = get_args()
    #freeze_seeds(args.seed)
    freeze_seeds(116)
    batch_size = args.batch_size
                                           
    if args.mnist:
        transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  # MNIST has 1 channel
        ])
        ds_train = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        ds_validation = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
        dataset_type="MNIST"
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        ## SEGEL: one possible convenient normalization: You don't have to use it
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
        ds_train = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
        ds_validation = datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=transform)
        dataset_type="CIFAR10"
        
    ## SEGEL: When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation 
    # Split  80% train  20% test
    train_size = int(0.8 * len(ds_train))
    test_size = len(ds_train) - train_size
    ds_train, ds_test = random_split(ds_train, [train_size, test_size])

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_validation, batch_size=batch_size, shuffle=True)


    '''
    SEGEL:
    #this is just for the example. Simple flattening of the image is probably not the best idea                                        
    encoder_model = torch.nn.Linear(32*32*3,args.latent_dim).to(args.device)
    decoder_model = torch.nn.Linear(args.latent_dim,32*32*3 if args.self_supervised else NUM_CLASSES).to(args.device) 

    sample = train_dataset[0][0][None].to(args.device) #This is just for the example - you should use a dataloader
    output = decoder_model(encoder_model(sample.flatten()))
    print(output.shape)
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run the selected task
    if args.task == "1.2.1":
        run_1_2_1(dl_train, dl_val, dl_test, dataset_type, device)
    elif args.task == "1.2.2":
        run_1_2_2(dl_train, dl_val, dl_test, dataset_type, device)
    elif args.task == "1.2.3":
        run_1_2_3(dl_train, dl_val, dl_test, dataset_type, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")


'''
"READ ME":
example for how to run:

# Run 1.2.1
srun -c 2 --gres=gpu:1 python -u  main.py --mnist --task 1.2.1
python -u main.py --task 1.2.1 --mnist --batch-size 64

# Run 1.2.2
python  -u main.py --task 1.2.2 --mnist --batch-size 64

# Run 1.2.3
python -u main.py --task 1.2.3 --mnist --batch-size 64
'''

