# ==============
# Part 1.2.1
dataset_path = '/datasets/cv_datasets/data'

def part121_mnist_hyperparams():
    hypers = dict(
        #seed=0,
        data_path=dataset_path,
        batch_size=64,
        h_dim=128,    
        learn_rate=0.0002,
        betas=(0.9, 0.999),
    )
    return hypers
    

