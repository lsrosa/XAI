# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def get_activations(self, **kwargs):
    model = kwargs['model'] 
    device = model.device 
    hooks = model.get_hooks()

    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False 

    if not self._peepds:
        raise RuntimeError('No data found. Please run get_peep_dataset() first.')

    for ds_key in self._peepds:
        if verbose: print(f'\n ---- Getting activations for {ds_key}\n')
        file_path = self._file_paths[ds_key]
        n_samples = self._n_samples[ds_key]       

        #------------------------------------------------
        # pre-allocate predictions, results, activations
        #------------------------------------------------
        
        # check if in and out activations exist
        if model._si and (not ('in_activations' in self._peepds[ds_key])):
            if verbose: print('adding in act tensorDict')
            self._peepds[ds_key]['in_activations'] = TensorDict(batch_size=n_samples)
        elif verbose: print('In activations exist.')

        if model._so and (not ('out_activations' in self._peepds[ds_key])):
            if verbose: print('adding out act tensorDict')
            self._peepds[ds_key]['out_activations'] = TensorDict(batch_size=n_samples)
        elif verbose: print('Out activations exist.')
        
        # check if layer in and out activations exist
        _layers_to_save = []
        for layer_key in model.get_target_layers():
            # allocate for input activations 
            if model._si and (not (layer_key in self._peepds[ds_key]['in_activations'])):
                if verbose: print('allocating in act layer: ', layer_key)
                act_shape = hooks[layer_key].in_shape
                self._peepds[ds_key]['in_activations'][layer_key] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _layers_to_save.append(layer_key)

            # allocate for output activations 
            if model._so and (not (layer_key in self._peepds[ds_key]['out_activations'])):
                if verbose: print('allocating out act layer: ', layer_key)
                act_shape = hooks[layer_key].out_shape
                self._peepds[ds_key]['out_activations'][layer_key] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _layers_to_save.append(layer_key)
        
        _layers_to_save = list(set(_layers_to_save))
        if verbose: print('Layers to save: ', _layers_to_save)
        if len(_layers_to_save) == 0:
            print(f'No new activations for {ds_key}, skipping')
            continue
        
        # to check if pred and results data exist 
        has_pred = 'pred' in self._peepds[ds_key]
        
        # allocate memory for pred and result
        if not has_pred:
            self._peepds[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            self._peepds[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))
        
        # ---------------------------------------
        # compute predictions and get activations
        # ---------------------------------------
        
        # create a temp dataloader to iterate over images
        _dl = DataLoader(self._peepds[ds_key], batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        
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
                self._peepds[ds_key][bn*bs:bn*bs+n_in] = {'pred':predicted_labels, 'result':results}
            
            for layer_key in _layers_to_save:
                if model._si:
                    self._peepds[ds_key][bn*bs:bn*bs+n_in] = {'in_activations': {layer_key: hooks[layer_key].in_activations}}
                if model._so:
                    self._peepds[ds_key][bn*bs:bn*bs+n_in] = {'out_activations': {layer_key: hooks[layer_key].out_activations}}
        
        #-----------------------------------
        # Saving updates to memory
        #-----------------------------------
        if verbose: print(f'Saving {ds_key} to {file_path}.')
        self._peepds[ds_key].memmap(file_path)

    return 

