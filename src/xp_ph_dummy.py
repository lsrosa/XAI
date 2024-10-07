from pathlib import Path as Path

from datasets.cifar import Cifar
from models.vgg import VGG 
from activations.activations import Activations
from peepholes.peepholes import Peepholes

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
	model_name = 'dummy_model.pth'

	model = Dummy_model(input_size=25, hidden_size=35, output_size=10)
	torch.save(model.state_dict(), '../../data/banana.pt') 

		

	model.load_and_set(path=model_dir, name=model_name, model=nn, verbose=True) # da definire in ModelBase -> check param in input




	
    pretrained = True
    model_dir = '/srv/newpenny/XAI/LM/models'
    model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
    f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'   # questo path viene genrato nel load and save del dummy_model.py
    
    model = VGG(device=device) # instanzia dummy net wrap
    model.load_checkpoint(path=model_dir, name=model_name, verbose=True) #2 - qui tiro fuori i pesi dal path

# load check e set model -> fai funz unica (in ModelBase)
	
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)# crea vgg #1 Ha senso che prima venga dichiarata vgg16
	# nn poi chiamerà una funz -> model.Load_and_set(model=nn) in cui faccio #2 e #3 -> questa funz definita in model_base mi sembra
    in_features = 4096
    num_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.set_model(model=nn) #3 - qui sto inserendo i pesi dentro l'istanza del modello vgg16

    layers_dict = {'classifier': [0,3],
                  'features': [28]}
    model.set_target_layers(target_layers=layers_dict, verbose=True)
    print('target layers: ', model.get_target_layers()) 

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)
    
