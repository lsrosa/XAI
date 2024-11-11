# python stuff
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader 

class Peepholes:
    def __init__(self, **kwargs):
        self.layer = kwargs['layer']
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._classifier = kwargs['classifier'] 

        # computed in get_peepholes
        self._phs = None
        
        # computed in get_dataloaders()
        self._loaders = None
        return

    def get_peepholes(self, **kwargs):
        '''
        Compute model probabilities from classifier probabilities and empirical posteriors.
        
        Args:
        - dataloader (DataLoader): Dataloader containing data to be parsed with the paser function set on __init__() 
        '''
        if self._classifier._empp == None:
            raise RuntimeError('No prediction probabilities. Please run classifier.compute_empirical_posteriors() first.')
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32
        _dls = kwargs['loaders']

        layer = self.layer 
        # device = kwargs['device']

        _phs = {} 

        for loader_name in _dls:
            bs = _dls[loader_name].batch_size
            if verbose: print(f'\n ---- Getting peepholes for {loader_name}\n')
            file_path = self.path/(self.name.name+'.'+loader_name)
           
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                _phs[loader_name] = TensorDict.load_memmap(file_path)
                _phs[loader_name].lock_()
                n_samples = len(_phs[loader_name])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(_dls[loader_name].dataset)
                if verbose: print('loader n_samples: ', n_samples) 
                _ph = TensorDict(batch_size=n_samples)
                _phs[loader_name] = _ph.memmap_like(file_path, num_threads=n_threads)   
            
            #-----------------------------------------
            # Pre-allocate peepholes
            #-----------------------------------------
            _td = _phs[loader_name]
            if not layer in _td:
                if verbose: print('allocating peepholes for layer: ', layer)
                _td.unlock_()
                _td[layer] = TensorDict(batch_size=n_samples)
                _td[layer]['peepholes'] = MMT.empty(shape=(n_samples, self._classifier.nl_model))
                _phs[loader_name] = _td.memmap_like(file_path, num_threads=n_threads)
             
                #----------------------------------------- 
                # computing peepholes
                #-----------------------------------------
                if verbose: print('\n ---- computing peepholes \n')
                for bn, batch in enumerate(tqdm(_dls[loader_name])):
                    n_in = len(batch)
                    cp = self._classifier.classifier_probabilities(batch=batch, verbose=verbose)
                    # print('cp %d: '%bn, cp)
                    _lp = cp@self._classifier._empp
                    _lp /= _lp.sum(dim=1, keepdim=True)
                    
                    _phs[loader_name][layer]['peepholes'][bn*bs:bn*bs+n_in] = _lp
            self._phs = _phs
        return 

    def get_scores(self, **kwargs):
        '''
        Compute scores (score_max and score_entropy) from precomputed peepholes.
        
        Args:
        - dataloader (DataLoader): Dataloader containing data to be parsed with the parser function set on __init__() 
        '''
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64 
        layer = self.layer 

        if self._phs == None:
            raise RuntimeError('No core vectors present. Please run get_peepholes() first.')

        for loader_name in self._phs:
            if verbose: print(f'\n ---- Getting scores for {loader_name}\n')
            file_path = self.path / (self.name.name + '.' + loader_name)
    
            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            _td = self._phs[loader_name]
            n_samples = len(_td)
            print(n_samples)
            if layer not in _td:
                raise ValueError(f"Peepholes for layer {layer} do not exist. Please run get_peepholes() first.")
            
            if 'peepholes' not in _td[layer]:
                raise ValueError(f"Peepholes do not exist in layer {layer}. Please run get_peepholes() first.")
            
            #-----------------------------------------
            # Check if scores already exist
            #-----------------------------------------
            if 'score_max' in _td[layer] and 'score_entropy' in _td[layer]:
                if verbose: print(f"Scores already computed for layer {layer}. Skipping computation.")
                continue 

            #-----------------------------------------
            # Pre-allocate scores
            #-----------------------------------------
            if verbose: print('Allocating scores for layer:', layer)
            _td.unlock_()
            _td[layer]['score_max'] = MMT.empty(shape=(n_samples,))
    
            _td[layer]['score_entropy'] = MMT.empty(shape=(n_samples,))
             
            self._phs[loader_name] = _td.memmap_like(file_path, num_threads=n_threads)   

            #-----------------------------------------
            # Compute scores
            #-----------------------------------------
            if verbose: print('\n ---- Computing scores \n')
            _dl = DataLoader(self._phs[loader_name], batch_size=bs, collate_fn=lambda x: x)
            for bn, batch in enumerate(tqdm(_dl)):
                n_in = len(batch)
                peepholes = batch[layer]['peepholes']
                self._phs[loader_name][layer]['score_max'][bn*bs:bn*bs+n_in] = torch.max(peepholes, dim=1).values
                self._phs[loader_name][layer]['score_entropy'][bn*bs:bn*bs+n_in] = torch.sum(peepholes * torch.log(peepholes + 1e-12), dim=1)
    
        return

    def evaluate(self, **kwargs): 
        layer = self.layer 
        cvs = kwargs['coreVectors']
        score_type = kwargs['score_type']
        self._classifier.nl_class
        self._classifier.nl_model

        quantiles = torch.arange(0, 1, 0.001) # setting quantiles list
        prob_train = self._phs['train'][layer]['peepholes']
        prob_val = self._phs['val'][layer]['peepholes']
        
        # TODO: vectorize
        conf_t = self._phs['train'][layer]['score_'+score_type] 
        conf_v = self._phs['val'][layer]['score_'+score_type] 
        th = [] 
        lt = []
        lf = []

        c = cvs['val'].dataset['result'].detach().numpy()
        cntt = Counter(c) 
        
        for q in quantiles:
            perc = torch.quantile(conf_t, q)
            th.append(perc)
            idx = torch.argwhere(conf_v > perc)[:,0]

            # TODO: vectorize
            cnt = Counter(c[idx]) 
            lt.append(cnt[True]/cntt[True]) 
            lf.append(cnt[False]/cntt[False])

        plt.figure()
        x = quantiles.numpy()
        y1 = np.array(lt)
        y2 = np.array(lf)
        plt.plot(x, y1, label='OK', c='b')
        plt.plot(x, y2, label='KO', c='r')
        plt.plot(np.array([0., 1.]), np.array([1., 0.]), c='k')
        plt.legend()
        plt.savefig((self.path/self.name).as_posix()+'.png')
        plt.close()

        return np.linalg.norm(y1-y2), np.linalg.norm(y1-y2)

    def get_dataloaders(self, **kwargs):
        batch_dict = kwargs['batch_dict'] if 'batch_dict' in kwargs else {key: 64 for key in self._phs}
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 

        if self._loaders:
            if verbose: print('Loaders exist. Returning existing ones.')
            return self._loaders

        _loaders = {}
        for key in self._phs:
            if verbose: print('creating dataloader for: ', key)
            _loaders[key] = DataLoader(
                    dataset = self._phs[key],
                    batch_size = batch_dict[key], 
                    collate_fn = lambda x: x
                    )

        self._loaders = _loaders 
        return self._loaders
