# Our stuff
from models.model_base import ModelBase, Hook 

# General python stuff
import numpy as np
from pathlib import Path as Path

# torch stuff
import torch

class VGG(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, **kwargs)
        
    def add_hooks(self, **kwargs):
        _si = kwargs['save_input'] if 'save_input' in kwargs else True 
        _so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        _hooks = {}
        for key in self._target_layers:
            if verbose: print('Adding hook to layer: ', key)
            parts = key.split('.')
            layer = self._model._modules[parts[0]][int(parts[1])]
            hook = Hook(save_input=_si, save_output=_so)
            handle = hook.register(layer)

            _hooks[key] = hook
        
        self._hooks = _hooks
        return 
     
    def get_svds(self, **kwargs):
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        # create folder
        path.mkdir(parents=True, exist_ok=True)

        _svds = {} 
        for lk in self._target_layers:
            if verbose: print(f'\n ---- Getting SVDs for {lk}\n')
            file_path = path/(name.name+'.'+lk)
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                _svds[lk] = torch.load(file_path)
                continue

            weight = self._state_dict[lk+'.weight']
            bias = self._state_dict[lk+'.bias']
            print(weight.shape, bias.shape, self._hooks[lk].layer)
            # get layer
            parts = lk.split('.')
            layer = self._model._modules[parts[0]][int(parts[1])]
            if isinstance(layer, torch.nn.Conv2d):
                print('conv layer')
                U = torch.rand(2)
                s = torch.rand(3)
                Vh = torch.rand(4)
            elif isinstance(layer, torch.nn.Linear):
                print('linear layer')
                W_ = torch.hstack((weight, bias.reshape(-1,1)))
                U, s, Vh = torch.linalg.svd(W_, full_matrices=False)
            _svds[lk] = {
                    'U': U.detach().cpu(),
                    's': s.detach().cpu(),
                    'Vh': Vh.detach().cpu()
                    }
            if verbose: print(f'saving {file_path}')
            torch.save(_svds[lk], file_path)

        self._svds = _svds
        return self._svds
