# torch stuff
import torch
from torch.utils.data import DataLoader 
from tensordict import TensorDict, PersistentTensorDict

# generic python stuff
from pathlib import Path
from tqdm import tqdm
from functools import partial

class CoreVectors():
    from coreVectors.dataset import get_coreVec_dataset
    from coreVectors.activations import get_activations
    from coreVectors.svd_coreVectors import get_coreVectors

    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._model = kwargs['model'] if 'model' in kwargs else None  
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'  

        # computed in get_activations()
        self._loaders = None

        # computed in get_coreVec_dataset()
        self._corevds = None
        self._n_samples = None
        self._file_paths = None
        
        # set in normalize_corevectors() 
        self._norm_wrt = None
        self._norm_mean = None 
        self._norm_std = None 
        self._is_normalized = None
        
        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False
        return
     
    def normalize_corevectors(self, **kwargs):
        self.check_uncontexted()
        wrt = kwargs['wrt']
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64 
        
        if wrt not in self._corevds:
            raise RuntimeError(f'{wrt} not in data. Choose from {self._corevds.keys()}')
        
        file_path = self.path/(self.name.name+'.normalization')
        
        if file_path.exists():
            means, stds, is_normed, _wrt = torch.load(file_path)
            if _wrt != wrt:
                raise RuntimeError(f"Seems like there are corevectors normalized w.r.t. {_wrt}, which is different from the requested {wrt}. Unormalization and re-normalization is not implemented. Submit a PR if you contribute to it =). Doing nothing.")
        else:
            is_normed = {} 

        # check for layers to be normalized for each dataloader
        layers_to_norm = {}
        cnt = 0
        for ds_key in self._corevds:
            layers_to_norm[ds_key] = []
            if not ds_key in is_normed:
                is_normed[ds_key] = []
            for lk in self._model.get_target_layers():
                if not lk in is_normed[ds_key]:
                    layers_to_norm[ds_key].append(lk)
                    cnt += 1
        print('Layers to norm: ', layers_to_norm)

        if cnt == 0:
            if verbose: print('All corevectors seems to be normalized. Doing nothing')
            self._norm_mean = means
            self._norm_std = stds
            return
        elif verbose: print(f'New unormalized layers: {layers_to_norm}. Running normalization.')

        means = self._corevds[wrt]['coreVectors'].mean(dim=0)
        stds = self._corevds[wrt]['coreVectors'].std(dim=0)

        # TODO: It is a bit excessive to renormalize all layers again (including the ones already normalized). Gotta change the logic to normalize only the ones in `layers_to_norm` 
        for ds_key in self._corevds:
            if verbose: print(f'\n ---- Normalizing core vectors for {ds_key}\n')
            td = self._corevds[ds_key]['coreVectors']  
            dl = DataLoader(td, batch_size=bs, collate_fn=lambda x: x)
            for batch in tqdm(dl, disable=not verbose, total=len(dl)):
                n_in = len(batch) 
                batch = (batch - means)/stds

            is_normed[ds_key] = list(set(layers_to_norm[ds_key]).union(is_normed[ds_key])) 
        print('is normed: ', is_normed)

        if not file_path.exists(): torch.save((means, stds, is_normed, wrt), file_path)
        self._norm_mean = means
        self._norm_std = stds
        self._is_normalized = is_normed
        self._norm_wrt = wrt 

        return

    def get_dataloaders(self, **kwargs):
        self.check_uncontexted()
        
        _bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
        if isinstance(_bs, int):
            batch_dict = {key: _bs for key in self._corevds}
        elif isinstance(_bs, dict):
            batch_dict = _bs
        else:
            raise RuntimeError('Batch size should be a dict or an integer')

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
                    batch_size = batch_dict[ds_key], 
                    collate_fn = lambda x: x
                    )

        self._loaders = _loaders 
        return self._loaders
    
    def load_only(self, **kwargs):
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']

        _corevds = {}
        _n_samples = {}
        _file_paths = {}

        for loader_name in loaders:
            if verbose: print(f'\n ---- Getting data from {loader_name}\n')
            file_path = self.path/(self.name.name+'.'+loader_name)
            _file_paths[loader_name] = file_path
            
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            _corevds[loader_name] = PersistentTensorDict.from_h5(file_path, mode='r').to(self.device)

            n_samples = len(_corevds[loader_name])
            if verbose: print('loaded n_samples: ', n_samples)
        
        norm_file_path = self.path/(self.name.name+'.normalization')
        if norm_file_path.exists():
            if verbose: print('Loading normalization info.')
            # TODO: should save and load these as non-tensor within tensordict
            means, stds, is_normed, wrt = torch.load(norm_file_path)
            self._norm_mean = means 
            self._norm_std = stds
            self._is_normalized = is_normed
            self._norm_wrt = wrt
        else:
            if verbose: print('No normalization info found')

        # save computed data within the class
        self._file_paths = _file_paths
        self._n_samples = _n_samples
        self._corevds = _corevds

        return
    
    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._corevds == None:
            if verbose: print('no corevds to close. doing nothing.')
            return

        for loader_name in self._corevds:
            if verbose: print(f'closing {loader_name}')
            self._corevds[loader_name].close()
            
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
