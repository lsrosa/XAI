import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['SCIPY_USE_PROPACK'] = "True"
 
threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

import torchattacks

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

from pathlib import Path as Path
import abc 

# python stuff

from pathlib import Path as Path
from numpy.random import randint
from tqdm import tqdm

# Our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap
from adv_atk.attacks_base import fds, ftd
from coreVectors.coreVectors import CoreVectors 
from coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn

# from credibility import get_credibility

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights

use_cuda = torch.cuda.is_available()
cuda_index = 1 #torch.cuda.device_count() - 3
device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
print(f"Using {device} device")

#--------------------------------
# Dataset 
#--------------------------------
# model parameters
name_model = 'vgg16'
dataset = 'CIFAR100' 
seed = 29
bs = 64
data_path = '/srv/newpenny/dataset/CIFAR100'

ds = Cifar(dataset=dataset, data_path=data_path)
ds.load_data(
        batch_size = bs,
        data_kwargs = {'num_workers': 4, 'pin_memory': True},
        seed = seed,
        )


#--------------------------------
# Model 
#--------------------------------
pretrained = True
model_dir = '/srv/newpenny/XAI/models'
# model_name = f'{name_model}_pretrained={pretrained}_dataset={dataset}-'\
# f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'

model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
in_features = 4096
num_classes = len(ds.get_classes()) 
nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
model = ModelWrap(device=device)
model.set_model(model=nn, path=model_dir, name=model_name, verbose=True)

target_layers = [
            'classifier.0',
            # 'classifier.3',
            #'features.28'
            ]
model.set_target_layers(target_layers=target_layers, verbose=True)

direction = {'save_input':True, 'save_output':False}
model.add_hooks(**direction, verbose=False) 

dry_img, _ = ds._train_ds.dataset[0]
dry_img = dry_img.reshape((1,)+dry_img.shape)
model.dry_run(x=dry_img)

#--------------------------------
# CoreVectors 
#--------------------------------
loaders = ds.get_dataset_loaders()

cvs_name = 'corevectors'
cvs_path = Path.cwd()/f'../data/_corevectors_l/{dataset}'

corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        device = device
        )

# copy dataset to coreVect dataset
with corevecs as cv:
    cv.get_coreVec_dataset(
            loaders = loaders,
            verbose = True
            ) 
