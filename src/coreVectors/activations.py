# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def get_activations(self, **kwargs):
    model = self._model
    device = model.device 
    hooks = model.get_hooks()

    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32

    if not self._corevds:
        raise RuntimeError('No data found. Please run get_coreVec_dataset() first.')

    for ds_key in self._corevds:
        if verbose: print(f'\n ---- Getting activations for {ds_key}\n')
        file_path = self._file_paths[ds_key]
        n_samples = self._n_samples[ds_key]       

        #------------------------------------------------
        # pre-allocate predictions, results, activations
        #------------------------------------------------
        _td = self._corevds[ds_key]
        _td.unlock_()

        # check if in and out activations exist
        if model._si and (not ('in_activations' in _td)):
            if verbose: print('adding in act tensorDict')
            _td['in_activations'] = TensorDict(batch_size=n_samples)
        elif verbose: print('In activations exist.')

        if model._so and (not ('out_activations' in _td)):
            if verbose: print('adding out act tensorDict')
            _td['out_activations'] = TensorDict(batch_size=n_samples)
        elif verbose: print('Out activations exist.')
        
        # check if layer in and out activations exist
        _layers_to_save = []
        for lk in model.get_target_layers():
            # this set if there are layers to prevent double entries 
            _lts = None

            # allocate for input activations 
            if model._si and (not (lk in _td['in_activations'])):
                if verbose: print('allocating in act layer: ', lk)
                act_shape = hooks[lk].in_shape
                _td['in_activations'][lk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = lk

            # allocate for output activations 
            if model._so and (not (lk in _td['out_activations'])):
                if verbose: print('allocating out act layer: ', lk)
                act_shape = hooks[lk].out_shape
                _td['out_activations'][lk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = lk
             
            if _lts != None: _layers_to_save.append(_lts)
        
        if verbose: print('Layers to save: ', _layers_to_save)
        if len(_layers_to_save) == 0:
            self._corevds[ds_key].lock_()
            print(f'No new activations for {ds_key}, skipping')
            continue
        
        # to check if pred and results data exist 
        has_pred = 'pred' in _td 
        
        # allocate memory for pred and result
        if not has_pred:
            _td['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            _td['result'] = MMT.empty(shape=torch.Size((n_samples,)))
        
        #-----------------------------------
        # Saving allocation to memory
        #-----------------------------------
        if verbose: print(f'Saving {ds_key} to {file_path}.')
        self._corevds[ds_key] = _td.memmap_like(file_path, num_threads=n_threads)

        # ---------------------------------------
        # compute predictions and get activations
        # ---------------------------------------
        
        # create a temp dataloader to iterate over images
        _dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        
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
                self._corevds[ds_key]['pred'][bn*bs:bn*bs+n_in] = predicted_labels
                self._corevds[ds_key]['result'][bn*bs:bn*bs+n_in] = results
            
            for lk in _layers_to_save:
                if model._si:
                    self._corevds[ds_key]['in_activations'][lk][bn*bs:bn*bs+n_in] = hooks[lk].in_activations
                if model._so:
                    self._corevds[ds_key]['out_activations'][lk][bn*bs:bn*bs+n_in] = hooks[lk].out_activations

    return 

