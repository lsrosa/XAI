# python stuff
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader 

class Peepholes:
    def __init__(self, **kwargs):
        self.layer = kwargs['layer']
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._classifier = kwargs['classifier'] 

        # computed in get_peepholes
        self._phs = {} 
        
        # computed in get_dataloaders()
        self._loaders = None

        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        return

    def get_peepholes(self, **kwargs):
        '''
        Compute model probabilities from classifier probabilities and empirical posteriors.
        
        Args:
        - dataloader (DataLoader): Dataloader containing data to be parsed with the paser function set on __init__() 
        '''
        self.check_uncontexted()

        if self._classifier._empp == None:
            raise RuntimeError('No prediction probabilities. Please run classifier.compute_empirical_posteriors() first.')
        _empp = self._classifier._empp.to(self.device)
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        _dls = kwargs['loaders']

        layer = self.layer 

        for ds_key in _dls:
            bs = _dls[ds_key].batch_size
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name.name+'.'+ds_key)
           
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+').to(self.device)
                n_samples = len(self._phs[ds_key])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(_dls[ds_key].dataset)
                if verbose: print('loader n_samples: ', n_samples) 
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], device=self.device, mode='w')
            
            #-----------------------------------------
            # Pre-allocate peepholes
            #-----------------------------------------
            if not layer in self._phs[ds_key]:
                if verbose: print('allocating peepholes for layer: ', layer)
                self._phs[ds_key][layer] = TensorDict(batch_size=n_samples)
                self._phs[ds_key][layer]['peepholes'] = MMT.empty(shape=(n_samples, self._classifier.nl_model))
             
                #----------------------------------------- 
                # computing peepholes
                #-----------------------------------------
                if verbose: print('\n ---- computing peepholes \n')
                _dl_t = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x:x)
                for batch in tqdm(zip(_dls[ds_key], _dl_t), disable=not verbose, total=len(_dl_t)):
                    data_in, data_t = batch
                    cp = self._classifier.classifier_probabilities(batch=data_in, verbose=verbose).to(self.device)
                    _lp = cp@_empp
                    _lp /= _lp.sum(dim=1, keepdim=True)
                    data_t[layer]['peepholes'] = _lp
            else:
                if verbose: print('Peepholes for {layer} already present. Skipping.')
        return 

    def get_scores(self, **kwargs):
        '''
        Compute scores (score_max and score_entropy) from precomputed peepholes.
        
        Args:
        - dataloader (DataLoader): Dataloader containing data to be parsed with the parser function set on __init__() 
        '''
        self.check_uncontexted()
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64 
        layer = self.layer 

        if self._phs == None:
            raise RuntimeError('No core vectors present. Please run get_peepholes() first.')

        for ds_key in self._phs:
            if verbose: print(f'\n ---- Getting scores for {ds_key}\n')
            file_path = self.path / (self.name.name + '.' + ds_key)
    
            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            if layer not in self._phs[ds_key]:
                raise ValueError(f"Peepholes for layer {layer} do not exist. Please run get_peepholes() first.")
            
            if 'peepholes' not in self._phs[ds_key][layer]:
                raise ValueError(f"Peepholes do not exist in layer {layer}. Please run get_peepholes() first.")
            
            #-----------------------------------------
            # Check if scores already exist
            #-----------------------------------------
            if 'score_max' in self._phs[ds_key][layer] and 'score_entropy' in self._phs[ds_key][layer]:
                if verbose: print(f"Scores already computed for layer {layer}. Skipping computation.")
                continue 

            #-----------------------------------------
            # Pre-allocate scores
            #-----------------------------------------
            if verbose: print('Allocating scores for layer:', layer)
            self._phs[ds_key][layer].batch_size = torch.Size((n_samples,))
            self._phs[ds_key][layer]['score_max'] = MMT.empty(shape=(n_samples,))
    
            self._phs[ds_key][layer]['score_entropy'] = MMT.empty(shape=(n_samples,))
             
            #-----------------------------------------
            # Compute scores
            #-----------------------------------------
            if verbose: print('\n ---- Computing scores \n')
            _dl = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x: x)
            for batch in tqdm(_dl, disable=not verbose, total=len(_dl)):
                peepholes = batch[layer]['peepholes']
                batch[layer]['score_max'] = torch.max(peepholes, dim=1).values
                batch[layer]['score_entropy'] = torch.sum(peepholes * torch.log(peepholes + 1e-12), dim=1)
    
        return
    
    def load_only(self, **kwargs):
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name.name+'.'+ds_key)
           
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r').to(self.device)
            
        return

    def evaluate(self, **kwargs): 
        self.check_uncontexted()

        layer = self.layer 
        cvs = kwargs['coreVectors']
        score_type = kwargs['score_type']

        quantiles = torch.arange(0, 1, 0.001) # setting quantiles list
        prob_train = self._phs['train'][layer]['peepholes']
        prob_val = self._phs['val'][layer]['peepholes']
        
        # TODO: vectorize
        # conf_t = self._phs['train'][layer]['score_'+score_type].detach().cpu() 
        conf_t = self._phs['val'][layer]['score_'+score_type].detach().cpu() 
        conf_v = self._phs['val'][layer]['score_'+score_type].detach().cpu() 
 
        th = [] 
        lt = []
        lf = []

        c = cvs['val'].dataset['result'].detach().cpu().numpy()
        pred = cvs['val'].dataset['pred'].detach().cpu().numpy()
        true = cvs['val'].dataset['label'].detach().cpu().numpy()
        cntt = Counter(c) 

        # initial acc
        ic = np.sum(pred==true)
        iacc = ic/len(true)
        
        for q in quantiles:
            perc = torch.quantile(conf_t, q)
            th.append(perc)
            idx = torch.argwhere(conf_v > perc)[:,0]

            # TODO: vectorize
            cnt = Counter(c[idx]) 
            lt.append(cnt[True]/cntt[True]) 
            lf.append(cnt[False]/cntt[False])

            if q==0.1:
                pred_ = pred[idx]
                true_ = true[idx]
                fc = np.sum(pred_==true_)
                facc = fc/len(idx) if len(idx)>0 else 0

        acc_gain = facc - iacc

        plt.figure()
        x = quantiles.numpy()
        d = 1 - x
        y1 = np.array(lt)
        y2 = np.array(lf)
        plt.plot(x, y1, label='OK', c='b')
        plt.plot(x, y2, label='KO', c='r')
        plt.plot(x, d, c='k')
        plt.legend()
        plt.grid(ls=':')
        plt.savefig((self.path/f'{layer}.{self.name}.{score_type}').as_posix()+'.png')
        plt.close()
        
        # TODO: make dist to diagonal
        cmp = [20, 50, 100]
        d1 = []
        d2 = []
        for i in cmp:
            # d1.append(np.abs(y1[i]-d[i]))
            # d2.append(np.abs(y2[i]-d[i]))
            d1.append(y1[i] - d[i])
            d2.append(d[i] - y2[i])
        return np.mean(d1), np.mean(d2), acc_gain

    def get_dataloaders(self, **kwargs):
        self.check_uncontexted()

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

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._phs == None:
            if verbose: print('no peepholes to close. doing nothing.')
            return

        for ds_key in self._phs:
            if verbose: print(f'closing {ds_key}')
            self._phs[ds_key].close()
            
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return