# Our stuff
from models.model_base import ModelBase

# General python stuff
from pathlib import Path as Path

# torch stuff
import torch

class VGG(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, **kwargs)
        
    def statedict_2_layerskeys(self):
        sdk = self._state_dict.keys()
        print(sdk)

        lk = dict()
        for key in sdk:
            parts = key.split('.')
            if parts[2] == 'bias':
                continue

            module = parts[0]
            layer = int(parts[1])

            if module not in lk:
                lk[module] = []

            lk[module].append(layer)

        print('\nlayerskeys\n', lk)

        return lk
