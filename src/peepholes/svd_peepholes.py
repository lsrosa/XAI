# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

def peep_matrices_from_svds(svd, lk):
    return svd[lk]['Vh']

def get_peepholes(self, **kwargs):
    model = kwargs['model']
    device = model.device 
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    peep_matrices = kwargs['peep_matrices']
    

    parser = kwargs['parser'] if 'parser' in kwargs else lambda x, y:x[y]
    parser_kwargs = kwargs['parser_kwargs'] if 'parser_kwargs' in kwargs and 'parser' in kwargs else dict()
    
    if not self._peepds:
        raise RuntimeError('No data found. Please run get_peep_dataset() first.')

    for ds_key in self._peepds:
        if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
        file_path = self._file_paths[ds_key]
        n_samples = self._n_samples[ds_key]       
         
        # create peepholes TensorDict if needed 
        if not 'peepholes' in self._peepds[ds_key]:
            if verbose: print('adding peepholes tensorDict')
            self._peepds[ds_key]['peepholes'] = TensorDict(batch_size=n_samples)
        elif verbose: print('Peepholes TensorDict exists.')
        
        # check if layer in and out activations exist
        _layers_to_save = []
        for lk in model.get_target_layers():
            peep_m = parser(peep_matrices, lk, **parser_kwargs)
            print('peep_m shape: ', peep_m.shape)
            # allocate for input activations 
            if not (lk in self._peepds[ds_key]['peepholes']):
                if verbose: print('allocating peepholes for layer: ', lk)
                peep_size = peep_m.shape[0] 
                print('peep_size: ', peep_size)
                self._peepds[ds_key]['peepholes'][lk] = MMT.empty(shape=torch.Size((n_samples,)+(peep_size,)))
                _layers_to_save.append(lk)
        if verbose: print('Layers to save: ', _layers_to_save)
        if len(_layers_to_save) == 0:
            print(f'No new peepholes for {ds_key}, skipping')
            continue

        # ---------------------------------------
        # compute peepholes 
        # ---------------------------------------

        # create a temp dataloader to iterate over images
        _dl = DataLoader(self._peepds[ds_key], batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        
        if verbose: print('Computing peepholes')
        for lk in _layers_to_save:
            peep_m = parser(peep_matrices, lk, **parser_kwargs).to(device)
            if verbose: print(f'\n ---- Getting peepholes for {lk}\n')
            
            # TODO: make a generic get layer function
            # get layer
            parts = lk.split('.')
            layer = model._model._modules[parts[0]][int(parts[1])]
            if isinstance(layer, torch.nn.Linear):
                for bn, data in enumerate(tqdm(_dl)):
                    n_act = data['in_activations'][lk].shape[0]
                    acts = data['in_activations'][lk].contiguous()
                    acts_flat = acts.flatten(start_dim=1)
                    ones = torch.ones(n_act, 1)
                    _acts = torch.hstack((acts_flat, ones)).to(device)
                    phs = (peep_m@_acts.T).T
                    self._peepds[ds_key][bn*bs:bn*bs+n_act] = {'peepholes': {lk: phs}}
            if isinstance(layer, torch.nn.Conv2d):
                pad_mode = layer.padding_mode if layer.padding_mode != 'zeros' else 'constant'
                padding = _reverse_repeat_tuple(layer.padding, 2) 
                for bn, data in enumerate(tqdm(_dl)):
                    n_act = data['in_activations'][lk].shape[0]
                    acts = data['in_activations'][lk].contiguous()
                    acts_pad = pad(acts, pad=padding, mode=pad_mode)

                    acts_flat = acts_pad.flatten(start_dim=1)
                    ones = torch.ones(n_act, 1)
                    _acts = torch.hstack((acts_flat, ones)).to(device)
                    phs = (peep_m@_acts.T).T
                    self._peepds[ds_key][bn*bs:bn*bs+n_act] = {'peepholes': {lk: phs}}

        #-----------------------------------
        # Saving updates to memory
        #-----------------------------------
        if verbose: print(f'Saving {ds_key} to {file_path}.')
        self._peepds[ds_key].memmap(file_path)
