# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def get_coreVec_dataset(self, **kwargs):
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    loaders = kwargs['loaders']
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32 

    _corevds = {}
    _n_samples = {}
    _file_paths = {}

    for loader_name in loaders:
        if verbose: print(f'\n ---- Getting data from {loader_name}\n')
        file_path = self.path/(self.name.name+'.'+loader_name)
        _file_paths[loader_name] = file_path
        bs = loaders[loader_name].batch_size
        
        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            _corevds[loader_name] = TensorDict.load_memmap(file_path)
            _corevds[loader_name].lock_()
            n_samples = len(_corevds[loader_name])
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            n_samples = len(loaders[loader_name].dataset)
            if verbose: print('loader n_samples: ', n_samples) 
            #TODO: check device
            _td = TensorDict(batch_size=n_samples)
            
            #------------------------
            # copy images and labels
            #------------------------
            
            # get shapes for pre-allocation
            _img, _label = loaders[loader_name].dataset[0]
            # pre-allocate
            _td['image'] = MMT.empty(shape=torch.Size((n_samples,)+_img.shape)) 
            _td['label'] = MMT.empty(shape=torch.Size((n_samples,))) 

            if verbose: print('Allocating data')
            _corevds[loader_name] = _td.memmap_like(file_path, num_threads=n_threads)

            if verbose: print('Copying images and labels')
            for bn, data in enumerate(tqdm(loaders[loader_name])): 
                images, labels = data
                n_in = len(images)
                _corevds[loader_name]['image'][bn*bs:bn*bs+n_in] = images
                _corevds[loader_name]['label'][bn*bs:bn*bs+n_in] = labels
        
        _n_samples[loader_name] = n_samples
    
    # save computed data within the class
    self._file_paths = _file_paths
    self._n_samples = _n_samples
    self._corevds = _corevds
    return
