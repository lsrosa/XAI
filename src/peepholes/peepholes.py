from peepholes.conv2d_to_sparse import conv2d_to_sparse as c2s

class Peepholes():
    def __init__():
        self._svds = None

    def get_svds(self, **kwargs):
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        model = kwargs['model']

        # create folder
        path.mkdir(parents=True, exist_ok=True)
        
        _svds = TensorDict()

        file_path = path/(name.name)
        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            _svds = TensorDict.load_memmap(file_path)
        
        _layers_to_compute = []
        for lk in model._target_layers:
            if lk in _svds.keys():
                continue
            _layers_to_compute.append(lk)
        if verbose: print('Layers to compute SVDs: ', _layers_to_compute)
        
        for lk in _layers_to_compute:
            if verbose: print(f'\n ---- Getting SVDs for {lk}\n')

            weight = model._state_dict[lk+'.weight']
            bias = model._state_dict[lk+'.bias']
            print(weight.shape, bias.shape, model._hooks[lk].layer)
            # get layer
            parts = lk.split('.')
            layer = model._model._modules[parts[0]][int(parts[1])]
            if isinstance(layer, torch.nn.Conv2d):
                print('conv layer')
                in_shape = model._hooks[lk].in_shape
                
                # Apply padding
                stride = layer.stride 
                dilation = layer.dilation
                padding = layer.padding
                                                                                                          
                _W_full = c2s(in_shape, weight, bias, stride=stride, padding=padding, dilation=dilation) 
                U, s, Vh = torch.svd_lowrank(_W_full, q=300)

            elif isinstance(layer, torch.nn.Linear):
                print('linear layer')
                W_ = torch.hstack((weight, bias.reshape(-1,1)))
                U, s, Vh = torch.linalg.svd(W_, full_matrices=False)
            _svds[lk] = TensorDict({
                    'U': MMT(U.detach().cpu()),
                    's': MMT(s.detach().cpu()),
                    'Vh': MMT(Vh.detach().cpu())
                    })
        
        if verbose: print(f'saving {file_path}')
        if len(_layers_to_compute) != 0:
            _svds.memmap(file_path)
        
        return _svds
