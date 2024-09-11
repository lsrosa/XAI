# Our stuff
from models.model_base import ModelBase

# General python stuff
from pathlib import Path as Path
from matplotlib import pyplot as plt
import numpy as np

# torch stuff
import torch
from torch.utils.data import random_split, DataLoader

# CIFAR from torchvision
from torchvision import transforms, datasets
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.utils import make_grid

class Cifar(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self)
        
        # use CIFAR10 by default
        if 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
        else:
            self.dataset = 'CIFAR10'
        print('dataset: %s'%self.dataset)

        if 'dataset_config' in kwargs:
            self.config = kwargs['dataset_config']
        else:
            if self.dataset == 'CIFAR10':
                self.config= {
                        'num_classes': 10,
                        'input_ch': 3,
                        'means': (0.424, 0.415, 0.384),
                        'stds': (0.283, 0.278, 0.284)
                        }
            elif self.dataset == '':
                self.config = {
                        'num_classes': 100,       
                        'input_ch': 3, 
                        'means': (0.438, 0.418, 0.377), 
                        'stds': (0.300, 0.287, 0.294)
                        }

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.classes = None
        self.loaders = None
        return
    
    def load_data(self, **kwargs):
        # parse parameteres
        batch_size = kwargs['batch_size']
        data_kwargs = kwargs['data_kwargs']
        seed = kwargs['seed']

        if 'data_augmentation' in kwargs:
            data_augmentation = kwargs['data_augmentation']
        else:
            data_augmentation=False
        
        if 'shuffle_train' in kwargs:
            shuffle_train = kwargs['suffle_train']
        else:
            shuffle_train = True
        
        if 'data_path' in kwargs:
            data_path =  kwargs['datapath']
        else:
            data_path = Path.cwd().parent/'data'

        # set torch seed
        torch.manual_seed(seed)

        # original dataset without augmentation
        original_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.config['means'], self.config['stds'])
        ])
        
        # Test dataset is loaded directly
        test_dataset = datasets.__dict__[self.dataset](
            root=data_path,
            train=False,
            transform=original_transform,
            download=True
        )
        
        # train data will be splitted for training and validation
        _train_data = datasets.__dict__[self.dataset]( 
            root=data_path,
            train=True,
            transform=transforms.Compose([transforms.ToTensor()]), #None, #original_transform,
            download=True
        )
        
        train_dataset, val_dataset = random_split(
            _train_data,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(seed)
        )
         
        dl = DataLoader(val_dataset)
        it = iter(dl)
        img, label = next(it)
        
        plt.figure()
        plt.imshow(np.transpose(make_grid(img), (1, 2, 0)))
        plt.savefig('aaa')
        input()
        val_dataset.dataset.transform = original_transform
        
        
        # Apply the transformation accoding to data augmentation 
        if data_augmentation:
            autoaugment_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10), 
                transforms.ToTensor(),
                transforms.Normalize(self.config['means'], self.config['stds'])
            ])
            train_dataset.dataset.transform = autoaugument_transform 
        else:
            train_dataset.dataset.transform = original_transform
     
        # Save datasets as objects in the class
        self.train_ds = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, **data_kwargs)
        self.val_ds = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)
        self.test_ds = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)

        self.classes = {i: class_name for i, class_name in enumerate(test_dataset.classes)}  
        
        self.loaders = {
            'train': self.train_ds,
            'val': self.val_ds,
            'test': self.test_ds,
            'classes': self.classes
            }

        return


    def get_train_dataset(self):
        return self.train_ds
    
    def get_val_dataset(self):
        return self.val_ds

    def get_test_dataset(self):
        return self.test_ds

    def get_parameter_matrix(self):
        print('ccc')

