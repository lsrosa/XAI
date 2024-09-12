# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

class ModelBase(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        # set in set_model()
        self._model = None

        # computed in load_checkpoint()
        self._checkpoint = None
        self._state_dict = None
        
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
    def statedict_2_layerskeys(self):
        raise NotImplementedError()
