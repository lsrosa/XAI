# General python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torch.utils.data import Dataset, DataLoader 

class ActivationsDataset(Dataset):
    def __init__(self, activation_keys):
        self.in_activations = {} 
        self.out_activations = {}

        for key in activation_keys:
            self.in_activations[key] = None 
            self.out_activations[key] = None 
                                             
        self.images = None 
        self.labels = None 
        self.preds = None 
        self.results = None 
        self.n = 0
        return
        
    def __getitem__(self, idx):
        if self.n == 0:
            return {} 

        _i = self.images[idx]
        _l = self.labels[idx]
        _p = self.preds[idx]
        _r = self.resuts[idx]
        
        _ia = {}
        for k in self.in_activations:
            _ia[key] = self.in_activations[key][idx]
        
        _oa = {}
        for k in self.out_activations:
            _oa[key] = self.out_activations[key][idx]

        return {
                'image': _i,
                'label': _l,
                'pred': _p,
                'results': _r,
                'in_activations': _ia,
                'out_activations': _oa
                }

    def add(self, image, label, pred, result, in_activations, out_activations):
        bs = image.shape[0]
        
        if self.n == 0:
            self.images = image
            self.labels = label
            self.preds = pred
            self.results = result
            
            for k in self.in_activations:
                if in_activations[k] != None:
                    self.in_activations[k] = in_activations[k]
            for k in self.out_activations:
                if out_activations[k] != None:
                    self.out_activations[k] = out_activations[k]
        else:
            self.images = torch.vstack((self.images, image))
            self.labels = torch.hstack((self.labels, label))
            self.preds = torch.hstack((self.preds, pred))
            self.results = torch.hstack((self.results, result))
        
            for k in self.in_activations:
                if in_activations[k] != None:
                    self.in_activations[k] = torch.vstack((self.in_activations[k], in_activations[k]))
            
            for k in self.out_activations:
                if out_activations[k] != None: 
                    self.out_activations[k] = torch.vstack((self.out_activations[k], out_activations[k]))

        self.n += bs
        return

    def __len__(self):
        return self.n


class Activations():
    def __init__(self, **kwargs):
        # computed in load_activations()
        self._act_loaders = None
    
    def compute_activations(self, **kwargs):
        """
        Computer the activations of a model.BaseModel. 
        We supppose the model hooks have been added using model.add_hooks().
    
        Parameters:
        -------
        model (ModelBase): the reference model, which generates the activation we are interested in
        loaders (Dict): dictionary containing torch.utils.data.DataLoaders. Activations will be computed for each dataloader.
        
        Returns:
        -------
        activations_loader (Dict): a dictonary containing a dataloader for an activations.ActivationDataset for each input loader.
        """
        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        file = path/name
        
        if file.exists():
            print('File exists. Loading from disk.')
            _al = torch.load(file)
            self._act_loaders = _al
            return
            
        model = kwargs['model'] 
        loaders = kwargs['loaders']
        device = model.device 
        hooks, _ = model.get_hooks()
        
        # for saving the activations_loaders
        _al = {} 
        
        for _loader_name in loaders:
            loader = loaders[_loader_name]

            batch_size = loader.batch_size
            n_samples = len(loader.dataset)
            print('bs, nsamples: ', batch_size, n_samples)

            _activations = ActivationsDataset(hooks.keys())
             
            # TODO: we should probably stack everything in a big tensor instead of doing this one data at a time
            # TODO: questo cacchio di output sembra di dare di un solo sample
            i = 0
            for data in loader: 
                image, label = data
                print('\nimg, label:', image.shape, label.shape) 

                image = image.to(device)
                with torch.no_grad():
                    y_predicted = model(image)
                print('predicted: ', y_predicted.shape) 
                
                _in_act = {}
                _out_act = {}
                for key in hooks.keys():
                    _in_act[key] = hooks[key].in_activations
                    _out_act[key] = hooks[key].out_activations
                    hooks[key].clear()
                    print('activations: ', key, ' - ', _in_act[key].shape, _out_act)
                predicted_labels = y_predicted.argmax(axis = 1).detach().cpu()
                result = predicted_labels == label
                
                _activations.add(
                        image = image,
                        label = label,
                        pred = predicted_labels,
                        result = result,
                        in_activations = _in_act, 
                        out_activations = _out_act 
                        )
                print('total data: ', _activations.n)
                print('total images: ', _activations.images.shape)
                print('total labels: ', _activations.labels.shape)
                print('total preds: ', _activations.preds.shape)
                print('total results: ', _activations.results.shape)
                for k in hooks.keys():
                    print(k, 'total in act: ', _activations.in_activations[k].shape)
                    print(k, 'total out act: ', _activations.out_activations[k])
                _al[_loader_name] = DataLoader(_activations) 
                i += 1
                if i == 3: break
        self._act_loaders = _al
        
        path.mkdir(parents=True, exist_ok=True)
        torch.save(_al, file)

        return self._act_loaders

    def get_activations_loader(self):
        if not self._act_loaders:
            raise RuntimeError('No activations data. Please run activations.load_data() or activations.compute_activations() first.')
        return self._act_loaders
