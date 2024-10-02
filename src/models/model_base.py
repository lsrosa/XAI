# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

class Hook:
    def __init__(self, save_input=True, save_output=False):
        self.layer = None 
        self.handle = None

        self._si = save_input
        self._so = save_output
        
        self.in_shape = None
        self.out_shape = None

        self.in_activations = None 
        self.out_activations = None 
        return
    
    def register(self, layer):
        # check is already registered to a layer
        if self.layer or self.handle:
            self.handle.remove()
            self.handle = None
            self.layer = None
        
        self.layer = layer
        self.handle = layer.register_forward_hook(self)
        return self.handle
    
    def set_shapes(self):
        if self._si:
            self.in_shape = self.in_activations.shape[1:]
        if self._so:
            self.out_shape = self.out_activations.shape[1:]
        return

    def __call__(self, module, module_in, module_out):
        if self._si: 
            self.in_activations = module_in[0].detach().cpu()
        if self._so: 
            self.out_activations = module_out.detach().cpu()
        return

    def __str__(self):
        return f"\nInputs shape: {self.in_activations.shape}\nOutputs shape: {self.out_activations.shape}\n"

class ModelBase(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        # set in set_model()
        self._model = None

        # set in set_target_layers()
        self._target_layers = None 

        # computed in load_checkpoint()
        self._checkpoint = None
        self._state_dict = None
        
        # computed in add_hooks()
        self._hooks = None
        self._si = None 
        self._so = None 
    
        return

    def __call__(self, x):
        return self._model(x)

    def set_model(self, **kwargs):
        '''
        Set a nn as a model and apply the loaded checkpoint to it.
        Args:
        - model (torch.nn): Folder where the model is located.
        - name (str|Path): Model name.
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''

        if (not self._checkpoint) or (not self._state_dict):
            raise RuntimeError('No checkpoint available. Please run load_checkpoint() first.')
        _model = kwargs['model']    
        assert(issubclass(type(_model), torch.nn.Module))

        self._model = _model
        self._model.load_state_dict(self._state_dict) 
        self._model.to(self.device)
        self._model.eval()
        return

    def load_checkpoint(self, **kwargs):
        '''
        Load checkpoint information containing model parameters and training information from a file.
        
        Args:
        - path (str|Path): Folder where the model is located.
        - name (str|Path): Model name.
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        file = path/name

        self._checkpoint = torch.load(file, map_location=self.device)
        self._state_dict = self._checkpoint['state_dict']
        
        # see what is saved in the checkpoint (except for the state_dict)
        if verbose:
            print('\n-----------------\ncheckpoint\n-----------------')
            for k, v in self._checkpoint.items():
                if k != 'state_dict':
                    print(k, v)
                else:
                    print('state_dict keys: \n', v.keys(), '\n')
            print('-----------------\n')
        return
    
    def set_target_layers(self, **kwargs):
        '''
        Set target layers studied with peephole. Other functions will operate only for the layers specified here: add_hooks()

        Args:
        - target_layers (dict): keys are the module names as in the loaded state_dict. 
        '''
        tl = kwargs['target_layers'] if 'target_layers' in kwargs else None
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._state_dict:
            raise RuntimeError('No state_dict loaded. Please run load_checkpoint() first.')

        # get unique set of keys, removing the '.weights' and '.bias'
        _keys = sorted(list(set([k.replace('.weight','').replace('.bias','') for k in self._state_dict.keys()])))
        
        _tl = []
        
        if not tl:
            _tl = _keys
            if verbose: print('Targeting all layers.')
        else:
            for key in _keys:
                parts = key.split('.')
                module_name = parts[0]
                layer_number = int(parts[1])

                if ((module_name not in tl) or (layer_number not in tl[module_name])):
                    if verbose: print(f'Skipping layer: {module_name}[{layer_number}]')
                    continue
                if verbose: print(f'Adding layer: {module_name}[{layer_number}]')
                _tl.append(module_name+'.'+str(layer_number))

        self._target_layers = _tl
        return
    
    def dry_run(self, **kwargs):
        '''
        A dry run is used to collect information from the module, such as activation's sizes

        Args:
        - x (tensor) - one input for the model set with set_model().
        '''
        _img = kwargs['x'].to(self.device)
        self._model(_img)    
        
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run set_hooks() first.')

        for hk in self._hooks:
            self._hooks[hk].set_shapes()

        return

    def get_target_layers(self):
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        return self._target_layers

    @abc.abstractmethod
    def add_hooks(self, **kwargs):
        raise NotImplementedError()

    def get_hooks(self):
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run add_hooks() first.')
        return self._hooks
