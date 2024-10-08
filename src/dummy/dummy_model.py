# Our stuff
from models.model_base import ModelBase, Hook 

from pathlib import Path as Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class Dummy_model(nn.Module):
    def __init__(self, **kwargs):
		nn.Module.__init__(self)
		si = kwargs['input_size']
		output_size = kwargs['output_size']
		hidden_features = kwargs['hidden_features']
		self.nn1 = nn.Sequential()
		self.nn1.add_module(nn.Linear(si, hidden_features))
		self.nn1.add_module(nn.Linear(hidden_features, hidden_features))
		self.nn1.add_module(nn.Linear(hidden_features, output_size))

	# forward
	def forwad(self, x):
		x = F.relu(self.layer_1(x))
		x = F.relu(self.layer_2(x))
		x = F.relu(self.layer_out)


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
	
	    if (not self._checkpoint) or (not self._state_dict):
	        raise RuntimeError('No checkpoint available. Please run load_checkpoint() first.')
	    
	    return