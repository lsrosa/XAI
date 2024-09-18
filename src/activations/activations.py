# Out stuff
from activations.activationsDataset import ActivationsDataset

# General python stuff
from pathlib import Path as Path
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader 

class Activations():
    def __init__(self, **kwargs):
        # computed in get_activations()
        self._act_loaders = None
    
    def get_activations(self, **kwargs):
        """
        Computer the activations of a model.BaseModel. 
        We supppose the model hooks have been added using model.add_hooks().
    
        Parameters:
        -------
        path (str): folder to store the activations datasets
        name (str): file name to store the activations dataset. Will store a different file name 'file.<loader_key>' for wach loader passed in the loaders parameter. If the file exists, load it from memory
        model (ModelBase): the reference model, which generates the activation we are interested in
        loaders (Dict): dictionary containing torch.utils.data.DataLoaders. Activations will be computed for each dataloader.
        verbose (bool): If true, print some progress messages 
        Returns:
        -------
        activations_loader (Dict): a dictonary containing a torch.utils.data.DataLoader(activations.ActivationDataset) for each loader in the loaders parameter.
        """
        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        # create folder
        path.mkdir(parents=True, exist_ok=True)
            
        model = kwargs['model'] 
        loaders = kwargs['loaders']
        device = model.device 
        hooks = model.get_hooks()
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        
        # for saving the ACTivations_DataSetS
        _act_dss = {} 
       
        for loader_name in loaders:
            if verbose: print(f'\n ---- Getting activations for {loader_name}\n')
            file_path = path/(name.name+'.'+loader_name)
            
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                _act_dss[loader_name] = torch.load(file_path)
                continue
            
            _acts = ActivationsDataset(hooks.keys())

            # TODO: we should probably stack everything in a big tensor instead of doing this one data at a time
            for image, label in tqdm(loaders[loader_name]): 
                #print('\nimg, label:', image.shape, label.shape) 

                _image = image.to(device)
                with torch.no_grad():
                    y_predicted = model(_image)
                #print('predicted: ', y_predicted.shape) 
                
                _in_act = {}
                _out_act = {}
                for key in hooks.keys():
                    _in_act[key] = hooks[key].in_activations
                    _out_act[key] = hooks[key].out_activations
                    #print('activations: ', key, ' - ', _in_act[key].shape, _out_act[key].shape)
                predicted_labels = y_predicted.argmax(axis = 1).detach().cpu()
                result = predicted_labels == label
                
                _acts.add(
                        image = image,
                        label = label,
                        pred = predicted_labels,
                        result = result,
                        in_activations = _in_act, 
                        out_activations = _out_act 
                        )
                '''
                print('total data: ', _acts.n)
                print('total images: ', _acts.images.shape)
                print('total labels: ', _acts.labels.shape)
                print('total preds: ', _acts.preds.shape)
                print('total results: ', _acts.results.shape)
                for k in hooks.keys():
                    print(k, 'total in act: ', _acts.in_activations[k].shape)
                    print(k, 'total out act: ', _acts.out_activations[k].shape)
                '''
            
            _act_dss[loader_name] = _acts 

            # Save datasets into file
            if verbose: print(f'saving {file_path}')
            torch.save(_act_dss[loader_name], file_path)
        
        # Create dataloaders for each activationDataset
        self._act_loaders = {}
        for loader_name in loaders:
            bs = loaders[loader_name].batch_size
            if verbose: print('creating dataloader for: ', loader_name)
            self._act_loaders[loader_name] = DataLoader(
                    dataset = _act_dss[loader_name],
                    batch_size = bs
                    )

        return self._act_loaders

    def get_activations_loader(self):
        if not self._act_loaders:
            raise RuntimeError('No activations data. Please run activations.compute_activations() first.')
        return self._act_loaders

