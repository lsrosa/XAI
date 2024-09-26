from pathlib import Path as Path

from datasets.cifar import Cifar
from models.vgg import VGG 
from activations.activations import Activations

import torch
from torchvision.models import vgg16, VGG16_Weights

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
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
    
    l=ds.get_train_dataset()

    pretrained = True
    model_dir = '/srv/newpenny/XAI/LM/models'
    model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
    f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'
    
    model = VGG(device=device)
    model.load_checkpoint(path=model_dir, name=model_name, verbose=True)
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    num_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.set_model(model=nn)

    layers_dict = {'classifier': [0,3],
                  'features': [28]}
    model.set_target_layers(target_layers=layers_dict, verbose=True)
    print('target layers: ', model.get_target_layers()) 

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    svds_path = Path.cwd()/'../data/svds'
    svds_name = 'svds' 
    model.get_svds(path=svds_path, name=svds_name, verbose=True)
    for svd in model._svds.values():
        print(svd['U'].shape, svd['s'].shape, svd['Vh'].shape)
    quit()
    activations = Activations()
    loaders = ds.get_dataset_loaders()

    act_dir = Path.cwd()/'../data/activations'
    act_name = 'activations'
    act_loaders = activations.get_activations(
            path=act_dir,
            name=act_name,
            model=model,
            loaders=loaders,
            verbose=True
            )
