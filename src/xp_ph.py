# python stuff
from pathlib import Path as Path
from numpy.random import randint

# Our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from peepholes.peepholes import Peepholes
from peepholes.svd_peepholes import peep_matrices_from_svds as parser_fn

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    # model parameters
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64

    ds = Cifar(dataset=dataset)

    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    pretrained = True
    model_dir = '/srv/newpenny/XAI/LM/models'
    model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
    f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'
    
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    num_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=True)

    layers_dict = {'classifier': [0, 3],
                  'features': [28]}
    model.set_target_layers(target_layers=layers_dict, verbose=True)
    print('target layers: ', model.get_target_layers()) 

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    svds_path = Path.cwd()/'../data/svds'
    svds_name = 'svds' 
    model.get_svds(model=model, path=svds_path, name=svds_name, verbose=True)
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
    
    #--------------------------------
    # Peepholes 
    #--------------------------------
    phs_name = 'peepholes'
    phs_dir = Path.cwd()/'../data/peepholes'
    peepholes = Peepholes(
            path = phs_dir,
            name = phs_name,
            )
    loaders = ds.get_dataset_loaders()

    # copy dataset to peepholes dataset
    peepholes.get_peep_dataset(
            loaders = loaders,
            verbose = True
            ) 

    peepholes.get_activations(
            model=model,
            loaders=loaders,
            verbose=True
            )
    
    peepholes.get_peepholes(
            model = model,
            peep_matrices = model._svds,
            parser = parser_fn,
            verbose = True
            )
    '''
    ranks = dict()
    for lk in model._target_layers:
        ranks[lk] = randint(1,5) 
    print(ranks)
    parser_kwargs = {'rank': ranks}
    '''
