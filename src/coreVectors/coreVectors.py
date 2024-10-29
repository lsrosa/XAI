# torch stuff
import torch
from torch.utils.data import DataLoader 
# generic python stuff
from pathlib import Path

class CoreVectors():
    from coreVectors.dataset import get_coreVec_dataset
    from coreVectors.activations import get_activations
    from coreVectors.svd_coreVectors import get_coreVectors

    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        # computed in get_activations()
        self._loaders = None

        # computed in get_coreVec_dataset()
        self._corevds = None
        self._n_samples = None
        self._file_paths = None
        return
    
    def get_dataloaders(self, **kwargs):
        batch_dict = kwargs['batch_dict'] if 'batch_dict' in kwargs else {key: 64 for key in self._peepds}
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        if self._loaders:
            if verbose: print('Loaders exist. Returning existing ones.')
            return self._loaders

        # Create dataloader for each coreV TensorDicts 
        _loaders = {}
        for ds_key in self._corevds:
            if verbose: print('creating dataloader for: ', ds_key)
            _loaders[ds_key] = DataLoader(
                    dataset = self._corevds[ds_key],
                    batch_size = bs, 
                    collate_fn = lambda x: x
                    )

        self._loaders = _loaders 
        return self._loaders