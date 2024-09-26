# General python stuff
from pathlib import Path as Path
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import DataLoader 
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

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
         
        # for saving the ACTivations_TensorDictS
        _act_tds = {} 
       
        for loader_name in loaders:
            if verbose: print(f'\n ---- Getting activations for {loader_name}\n')
            file_path = path/(name.name+'.'+loader_name)
            bs = loaders[loader_name].batch_size
            
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                _act_tds[loader_name] = TensorDict.load_memmap(file_path)
                n_samples = len(_act_tds[loader_name])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(loaders[loader_name].dataset)
                if verbose: print('loader n_samples: ', n_samples) 
                _act_tds[loader_name] = TensorDict(batch_size=n_samples) #TODO: check device
                
                #------------------------
                # copy images and labels
                #------------------------
                
                # get shapes for pre-allocation
                _img, _label = loaders[loader_name].dataset[0]
                # pre-allocate
                _act_tds[loader_name]['image'] = MMT.empty(shape=torch.Size((n_samples,)+_img.shape)) 
                _act_tds[loader_name]['label'] = MMT.empty(shape=torch.Size((n_samples,))) 

                if verbose: print('Copying images and labels')
                for bn, data in enumerate(tqdm(loaders[loader_name])): 
                    images, labels = data
                    n_in = len(images)
                    _act_tds[loader_name][bn*bs:bn*bs+n_in] = {'image':images, 'label':labels}
            
                # Save datasets into file
                if verbose: print(f'saving {file_path}')
                _act_tds[loader_name].memmap(file_path)
       
            #------------------------------------------------
            # pre-allocate predictions, results, activations
            #------------------------------------------------
            
            # check if in and out activations exist
            if model._si and (not ('in_activations' in _act_tds[loader_name])):
                print('adding in act tensorDict')
                _act_tds[loader_name]['in_activations'] = TensorDict(batch_size=n_samples)
            elif verbose: print('In activations exist.')

            if model._so and (not ('out_activations' in _act_tds[loader_name])):
                print('adding out act tensorDict')
                _act_tds[loader_name]['out_activations'] = TensorDict(batch_size=n_samples)
            elif verbose: print('Out activations exist.')
            
            # check if layer in and out activations exist
            _layers_to_save = []
            for layer_key in model.get_target_layers():
                # allocate for input activations 
                if model._si and (not (layer_key in _act_tds[loader_name]['in_activations'])):
                    if verbose: print('allocating in act layer: ', layer_key)
                    act_shape = hooks[layer_key].in_shape
                    _act_tds[loader_name]['in_activations'][layer_key] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                    _layers_to_save.append(layer_key)

                # allocate for output activations 
                if model._so and (not (layer_key in _act_tds[loader_name]['out_activations'])):
                    if verbose: print('allocating out act layer: ', layer_key)
                    act_shape = hooks[layer_key].out_shape
                    _act_tds[loader_name]['out_activations'][layer_key] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                    _layers_to_save.append(layer_key)
            
            _layers_to_save = list(set(_layers_to_save))
            if verbose: print('Layers to save: ', _layers_to_save)
            if len(_layers_to_save) == 0:
                print(f'No new activations for {loader_name}, skipping')
                continue
            
            # to check if pred and results data exist 
            has_pred = 'pred' in _act_tds[loader_name]
            
            # allocate memory for pred and result
            if not has_pred:
                _act_tds[loader_name]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
                _act_tds[loader_name]['result'] = MMT.empty(shape=torch.Size((n_samples,)))
            
            # ---------------------------------------
            # compute predictions and get activations
            # ---------------------------------------
            
            # create a temp dataloader to iterate over images
            _dl = DataLoader(_act_tds[loader_name], batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
            
            if verbose: print('Computing activations')
            for bn, data in enumerate(tqdm(_dl)):
                n_in = data['image'].shape[0]
                imgs = data['image'].contiguous().to(device)
                with torch.no_grad():
                    y_predicted = model(imgs)
                
                # do not save predictions and results if it is already there
                if not has_pred:
                    predicted_labels = y_predicted.argmax(axis = 1).detach().cpu()
                    results = predicted_labels == data['label']
                    _act_tds[loader_name][bn*bs:bn*bs+n_in] = {'pred':predicted_labels, 'result':results}
                
                for layer_key in _layers_to_save:
                    if model._si:
                        _act_tds[loader_name][bn*bs:bn*bs+n_in] = {'in_activations': {layer_key: hooks[layer_key].in_activations}}
                    if model._so:
                        _act_tds[loader_name][bn*bs:bn*bs+n_in] = {'out_activations': {layer_key: hooks[layer_key].out_activations}}
            
            #-----------------------------------
            # Saving updates to memory
            #-----------------------------------
            if verbose: print(f'Saving {loader_name} to {file_path}.')
            _act_tds[loader_name].memmap(file_path)


        # Create dataloader for each TensorDicts 
        self._act_loaders = {}
        for loader_name in loaders:
            bs = loaders[loader_name].batch_size
            if verbose: print('creating dataloader for: ', loader_name)
            self._act_loaders[loader_name] = DataLoader(
                    dataset = _act_tds[loader_name],
                    batch_size = bs, 
                    collate_fn = lambda x: x
                    )
            '''
            for data in self._act_loaders[loader_name]:
                print(data['image'][5])
                print(data['label'][5])
                print(data['pred'][5])
                print(data['result'][5])
                for k in data['in_activations'].keys(): 
                    print(k, data['in_activations'][k][5])
                for k in data['out_activations'].keys(): 
                    print(k, data['out_activations'][k][5])
                break
            '''
        return self._act_loaders

    def get_activations_loader(self):
        if not self._act_loaders:
            raise RuntimeError('No activations data. Please run activations.compute_activations() first.')
        return self._act_loaders
