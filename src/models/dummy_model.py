# Our stuff
from models.model_base import ModelBase, Hook 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Dummy_model(nn.Module):
    def __init__(self, **kwargs):
		nn.Module.__init__(self)
		si = kwargs['input_size']
		self.nn1 = nn.Sequential()
		self.nn1.add_module(nn.Linear(si, hidden_features))
		self.nn1.add_module(nn.Linear(hidden_features, hidden_features))
		self.nn1.add_module(nn.Linear(hidden_features, output_size))

	# forward
	def forwad(self, x):
		x = F.relu(self.layer_1(x))
		x = F.relu(self.layer_2(x))
		x = F.relu(self.layer_out)