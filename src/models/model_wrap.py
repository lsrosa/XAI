# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

#input dict and m - return 2 arrays
def flatten_dictionary(d, m):
    if not isinstance(d, dict): # leaf
        keys = [str(i) for i in d]
        layers = [m._modules[str(i)] for i in d]
        return keys, layers
    else:
        keys = []
        layers = []
        for k in d:
            _k, _l = flatten_dictionary(d[k], m._modules[k])
            _k = [k+'.'+ tempk for tempk in _k]
            keys += _k
            layers += _l
        return keys, layers


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

class ModelWrap(metaclass=abc.ABCMeta):

    from models.svd import get_svds

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
        
        # computed in get_svds()
        self._svds = None
        return

    def __call__(self, x):
        return self._model(x)


    def set_model(self, **kwargs):
        '''
        Set a nn as a model and apply the loaded checkpoint from a file.
        
        Args:
        - model (torch.nn): neural network
        - path (str|Path): Folder where the model is located.
        - name (str|Path): Model name.
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''
        # kwargs
        model = kwargs['model']
        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        file = path/name
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        # take the checkpoint and the state_dict from the saved file
        self._checkpoint = torch.load(file, map_location=self.device)
        self._state_dict = self._checkpoint['state_dict']
        
        # verbose - see what is saved in the checkpoint (except for the state_dict)
        if verbose:
            print('\n-----------------\ncheckpoint\n-----------------')
            for k, v in self._checkpoint.items():
                if k != 'state_dict':
                    print(k, v)
                else:
                    print('state_dict keys: \n', v.keys(), '\n')
            print('-----------------\n')
        
        # assign model    
        assert(issubclass(type(model), torch.nn.Module))
        
        self._model = model
        self._model.load_state_dict(self._state_dict) 
        self._model.to(self.device)
        self._model.eval()
        
        return

    def set_target_layers(self, **kwargs):
        '''
        Set target layers studied with peephole. Other functions will operate only for the layers specified here: add_hooks()

        Args:
        - target_layers (dict): keys are the module names as in the loaded state_dict. 
        '''
        tl_in = kwargs['target_layers'] if 'target_layers' in kwargs else None  #input dict
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._state_dict:
            raise RuntimeError('No state_dict loaded. Please run set_model() first.')

        k_out, l_out = flatten_dictionary(tl_in, self._model)
        
        # create the dict
        tl_out = dict(zip(k_out, l_out))
        
        # get unique set of keys, removing the '.weights' and '.bias'
        _keys = sorted(list(set([k.replace('.weight','').replace('.bias','') for k in self._state_dict.keys()])))

        # check if the keys you create are present in _keys
        for k in tl_out:
            assert(k in _keys)
        
        self._target_layers = tl_out
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

    def add_hooks(self, **kwargs):
        self._si = kwargs['save_input'] if 'save_input' in kwargs else True 
        self._so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        _hooks = {}
        for key in self._target_layers:
            if verbose: print('Adding hook to layer: ', key)

            layer = self._target_layers[key]
            hook = Hook(save_input=self._si, save_output=self._so)
            handle = hook.register(layer)

            _hooks[key] = hook
        
        self._hooks = _hooks
        return 

    def get_hooks(self):
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run add_hooks() first.')
        return self._hooks
