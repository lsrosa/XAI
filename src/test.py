import torch
from models.cifar import Cifar 

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    cifar = Cifar(dataset='CIFAR100')
    cifar.load_data(
            batch_size = 64,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = 29
            )
    cifar.get_train_dataset()
    cifar.get_val_dataset()
    cifar.get_test_dataset()
    cifar.get_parameter_matrix()
