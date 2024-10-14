# torch stuff
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
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
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
                    batch_size = bs, 
                    collate_fn = lambda x: x
                    )
            '''
            for data in self._act_loaders[ds_key]:
                print(data['image'][5])
                print(data['label'][5])
                print(data['pred'][5])
                print(data['result'][5])
                for k in data['in_activations'].keys(): 
                    print(k, data['in_activations'][k][5])
                for k in data['out_activations'].keys(): 
                    print(k, data['out_activations'][k][5])
                break
            '''
        self._loaders = _loaders 
        return self._loaders
