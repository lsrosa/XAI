# General python stuff
from pathlib import Path as Path
import abc  
import gc

# torch stuff
import torch

class Hook:
    def __init__(self, save_input=True, save_output=False):
        self._si = save_input
        self._so = save_output

        self.in_activations = None 
        self.out_activations = None 
        return

    def __call__(self, module, module_in, module_out):
        if self._si: 
            self.in_activations = module_in[0].detach().cpu()
        if self._so: 
            self.out_activations = module_out.detach().cpu()
        return

    def __str__(self):
        return f"\nInputs shape: {self.in_activations.shape}\nOutputs shape: {self.out_activations.shape}\n"
    '''
    def clear(self):
        del self.in_activations
        del self.out_activations
        gc.collect()

        self.in_activations = None
        self.out_activations = None 
        return
    '''

class ModelBase(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        # set in set_model()
        self._model = None

        # computed in load_checkpoint()
        self._checkpoint = None
        self._state_dict = None
        
        # computed in add_hooks()
        self._hook_handles = None
    
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

    @abc.abstractmethod
    def add_hooks(self, **kwargs):
        raise NotImplementedError()

    def get_hooks(self):
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run add_hooks() first.')
        if not self._hook_handles:
            raise RuntimeError('No hook handles available. Please run add_hooks() first.')
        return self._hooks, self._hook_handles
