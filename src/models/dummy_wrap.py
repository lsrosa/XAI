# Our stuff
from models.wrap_base import WrapBase, Hook 

# General python stuff
import numpy as np
from pathlib import Path as Path

class DummyWrap(WrapBase):
    def __init__(self, **kwargs):
        WrapBase.__init__(self, **kwargs)
    '''    
    def add_hooks(self, **kwargs):
        self._si = kwargs['save_input'] if 'save_input' in kwargs else True 
        self._so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        _hooks = {}
        for key in self._target_layers:
            if verbose: print('Adding hook to layer: ', key)

            # TODO: make a generic get layer function
            parts = key.split('.')
            layer = self._model._modules[parts[0]][int(parts[1])]
            hook = Hook(save_input=self._si, save_output=self._so)
            handle = hook.register(layer)

            _hooks[key] = hook
        
        self._hooks = _hooks
        return 
     '''
