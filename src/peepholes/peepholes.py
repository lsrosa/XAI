# torch stuff
import torch
from torch.utils.data import DataLoader 
# generic python stuff
from pathlib import Path

class Peepholes():
    from peepholes.activations import get_activations
    from peepholes.dataset import get_peep_dataset 
    from peepholes.svd_peepholes import get_peepholes

    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        # computed in get_activations()
        self._loaders = None

        # computed in get_peep_dataset()
        self._peepds = None
        self._n_samples = None
        self._file_paths = None
        return
    
    def get_dataloaders(self, **kwargs):
        batch_dict = kwargs['batch_dict'] if 'batch_dict' in kwargs else {key: 64 for key in self._peepds}
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        if self._loaders:
            if verbose: print('Loaders exist. Returning existing ones.')
            return self._loaders

        # Create dataloader for each peep TensorDicts 
        _loaders = {}
        for ds_key in self._peepds:
            if verbose: print('creating dataloader for: ', ds_key)
            _loaders[ds_key] = DataLoader(
                    dataset = self._peepds[ds_key],
                    batch_size = batch_dict[ds_key], 
                    collate_fn = lambda x: x
                    )

        self._loaders = _loaders 
        return self._loaders
        
    def get_act_from_ds(self, model, portion):
        """
        Extract the activations specific for a set of layer and the labels for a specific dataset
        """
        dataloader = self._loaders[portion]
        if dataloader.batch_size == len(dataloader.dataset):
            activations = {}
            for data in dataloader:
                labels = data['label'].detach().cpu().numpy()
                for key,layer in model._target_layers.items():
                    if isinstance(layer, torch.nn.Linear):
                        activations[key] = data['in_activations'][key].detach().cpu().numpy()
                    if isinstance(layer, torch.nn.Conv2d):
                        flatten_act = data['in_activations'][key].flatten(start_dim=1)
                        activations[key] = flatten_act.detach().cpu().numpy()
        else:
            raise RuntimeError('set DataLoader batch_size equal to the length of the dataset')

        return activations, labels
