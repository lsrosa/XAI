# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def get_coreVec_dataset(self, **kwargs):
    self.check_uncontexted()
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    loaders = kwargs['loaders']

    _corevds = {}
    _n_samples = {}
    _file_paths = {}

    for ds_key in loaders:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        file_path = self.path/(self.name.name+'.'+ds_key)
        _file_paths[ds_key] = file_path
        bs = loaders[ds_key].batch_size
        
        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            _corevds[ds_key] = TensorDict.from_h5(file_path, mode='r+').to(self.device)

            n_samples = len(_corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            n_samples = len(loaders[ds_key].dataset)
            if verbose: print('loader n_samples: ', n_samples) 
            _corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], device=self.device, mode='w')
            
            #------------------------
            # copy images and labels
            #------------------------
            
            # get shapes for pre-allocation
            if verbose: print('Allocating data')
            _img, _label = loaders[ds_key].dataset[0]
            # pre-allocate
            _corevds[ds_key]['image'] = MMT.empty(shape=torch.Size((n_samples,)+_img.shape)) 
            _corevds[ds_key]['label'] = MMT.empty(shape=torch.Size((n_samples,))) 

            if verbose: print('Copying images and labels')
            dl_in = loaders[ds_key]
            dl_t = DataLoader(_corevds[ds_key], batch_size=bs, collate_fn=lambda x:x)
            for data in tqdm(zip(dl_in, dl_t), disable=not verbose, total=len(dl_in)): 
                data_in, data_t = data
                images, labels = data_in
                data_t['image'] = images
                data_t['label'] = labels
        
        _n_samples[ds_key] = n_samples
    
    # save computed data within the class
    self._file_paths = _file_paths
    self._n_samples = _n_samples
    self._corevds = _corevds
    return
