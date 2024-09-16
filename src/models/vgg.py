# Our stuff
from models.model_base import ModelBase, Hook 

# General python stuff
from pathlib import Path as Path

# torch stuff
import torch

class VGG(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, **kwargs)
        
    def add_hooks(self, **kwargs):
        _si = kwargs['save_input'] if 'save_input' in kwargs else True 
        _so = kwargs['save_output'] if 'save_output' in kwargs else False 
         
        layers_dict = kwargs['layers_dict'] if 'layers_dict' in kwargs else None 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        hook_handles = {}
        hooks = {}
        
        # get unique set of keys, removing the '.weights' and '.bias'
        _keys = sorted(list(set([k.replace('.weight','').replace('.bias','') for k in self._state_dict.keys()])))

        for key in _keys:
            parts = key.split('.')
            
            module_name = parts[0]
            layer_number = int(parts[1])

            # Check is layers_dict is given, only add hooks for those layers 
            if layers_dict != None and ((module_name not in layers_dict) or (layer_number not in layers_dict[module_name])):
                if verbose: print(f'Skipping hook for: {module_name}[{layer_number}]')
                continue
            if verbose: print(f'Adding hook for: {module_name}[{layer_number}]')

            hook = Hook(save_input=_si, save_output=_so)
            layer = self._model._modules[module_name][layer_number]
            handle = layer.register_forward_hook(hook)
            hooks[key] = hook
            hook_handles[key] = handle
        
        self._hook_handles = hook_handles
        self._hooks = hooks
        return 
        
