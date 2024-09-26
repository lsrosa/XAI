# Our stuff
from models.model_base import ModelBase, Hook 
from models.conv2d_to_sparse import conv2d_to_sparse as c2s

# General python stuff
import numpy as np
from pathlib import Path as Path

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

class VGG(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, **kwargs)
        
    def add_hooks(self, **kwargs):
        self._si = kwargs['save_input'] if 'save_input' in kwargs else True 
        self._so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        _hooks = {}
        for key in self._target_layers:
            if verbose: print('Adding hook to layer: ', key)
            parts = key.split('.')
            layer = self._model._modules[parts[0]][int(parts[1])]
            hook = Hook(save_input=self._si, save_output=self._so)
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
        
        _svds = TensorDict()

        file_path = path/(name.name)
        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            _svds = TensorDict.load_memmap(file_path)
        
        _layers_to_compute = []
        for lk in self._target_layers:
            if lk in _svds.keys():
                continue
            _layers_to_compute.append(lk)
        if verbose: print('Layers to compute SVDs: ', _layers_to_compute)
        
        for lk in _layers_to_compute:
            if verbose: print(f'\n ---- Getting SVDs for {lk}\n')

            weight = self._state_dict[lk+'.weight']
            bias = self._state_dict[lk+'.bias']
            print(weight.shape, bias.shape, self._hooks[lk].layer)
            # get layer
            parts = lk.split('.')
            layer = self._model._modules[parts[0]][int(parts[1])]
            if isinstance(layer, torch.nn.Conv2d):
                print('conv layer')
                in_shape = self._hooks[lk].in_shape
                stride = self._hooks[lk].layer.stride 
                padding = self._hooks[lk].layer.padding 
                _W_full = c2s(in_shape, weight, bias, stride=stride, padding=padding) 
                print(_W_full.shape)

                U = torch.rand(2)
                s = torch.rand(3)
                Vh = torch.rand(4)

            elif isinstance(layer, torch.nn.Linear):
                print('linear layer')
                W_ = torch.hstack((weight, bias.reshape(-1,1)))
                U, s, Vh = torch.linalg.svd(W_, full_matrices=False)
            _svds[lk] = TensorDict({
                    'U': MMT(U.detach().cpu()),
                    's': MMT(s.detach().cpu()),
                    'Vh': MMT(Vh.detach().cpu())
                    })
        
        if verbose: print(f'saving {file_path}')
        if len(_layers_to_compute) != 0:
            _svds.memmap(file_path)
        
        ''' 
        for k in _svds.keys():
            print('\n', k, ' - U: ', _svds[k]['U'])
            print(k, ' - s: ', _svds[k]['s'])
            print(k, ' - Vh: ', _svds[k]['Vh'])
        '''
        self._svds = _svds
        return self._svds
