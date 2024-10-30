# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from adv_atk.attacks_base import fds, ftd

def get_coreVec_dataset(self, **kwargs):
    self.check_uncontexted()
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    loaders = kwargs['loaders']
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32 
    key_list = kwargs['key_list'] if 'key_list' in kwargs else ['image', 'label']
    parser = kwargs['parser'] if 'parser' in kwargs else fds

    _n_samples = {}
    _file_paths = {}

    for ds_key in loaders:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        file_path = self.path/(self.name.name+'.'+ds_key)
        print(file_path)
        _file_paths[ds_key] = file_path
        bs = loaders[ds_key].batch_size
        
        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+').to(self.device)

            n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            n_samples = len(loaders[ds_key].dataset)
            if verbose: print('loader n_samples: ', n_samples) 
            self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], device=self.device, mode='w')
            
            #------------------------
            # copy images and labels
            #------------------------

            if verbose: print('Allocating data')

            for key in key_list:
                data = parser(next(iter(loaders[ds_key])), key)[0]
                
                # get shapes for pre-allocation
                if data.shape == torch.Size([]):
                    self._corevds[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,))) 
                else:
                    self._corevds[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+data.shape))
            ###

            if verbose: print('Copying images and labels')
            dl_in = loaders[ds_key]
            dl_t = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x:x)
            
            for data in tqdm(zip(dl_in, dl_t), disable=not verbose, total=len(dl_in)): 
                data_in, data_t = data
                for key in key_list:
                    value = parser(data_in, key)
                    data_t[key] = value
        
        _n_samples[ds_key] = n_samples
    
    # save computed data within the class
    self._file_paths = _file_paths
    self._n_samples = _n_samples

    return
