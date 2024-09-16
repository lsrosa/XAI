# General python stuff
from pathlib import Path as Path

# torch stuff
#import torch
#from torch.utils.data import random_split, DataLoader

class SaveInput:
    def __init__(self):
        self.activations = []
        
    def __call__(self, module, module_in, module_out):
        self.activations.append(module_in)
        
    def clear(self):
        self.activations = []

class SaveOutput:
    def __init__(self):
        self.activations = []
        
    def __call__(self, module, module_in, module_out):
        self.activations.append(module_out)
        
    def clear(self):
        self.activations = []

class Activations():
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.dataset = kwargs['dataset']

        # computed in load_activations()
        self._train_act = None
        self._val_act = None
        self._test_act = None
    
    def compute_activations(self, **kwargs):
        """
        The following function aims to simplify the procedure to extract the activations of a torch.nn.Module. 
        To do so, the function get_activation deploys the built-in hooks to extract the required information. The result of this function 
        is a dictionary which contains firstly the inputs, the corresponding labels, the predictions of the model and then the activations divided in the specific layers.
    
        Parameters:
        -------
        model (ModelBase): the reference model, which generates the activation we are interested in
    
        layers_dict (dict({str:[int]})): dictionary identifying the layers we are interested in. The key is a string for the submodule and the values are a list of integers for the layer numbers within that submodule.
    
        direction: a string that indicates if the activation we get is either the input or the output of the chosen layer.
            the admissible strings are either 'in' or 'out'.
    
        Returns:
        -------
        dict_activations: this dictonary contains the images given as input to the model, the corresponding labels, the output pprovided by the network and then all the activations we selected thorugh the input strings string_section and index. The keys of the provided dictionary are the following:
        1. 'input'
        2. 'labels'
        3. 'pred'
        4. 'activations'- the name of the key is defined by the inputs string_section and index
        """

        dict_activations = {}
        dict_activations['input'] = []
        dict_activations['label'] = []
        dict_activations['pred'] = []
        dict_activations['results'] = []


        loaders = self.dataset._loaders
        device = self.model.device 
        
        for data in loader: 
            image, label = data
            dict_activations['input'].append(image)
            dict_activations['label'].append(label)
            
            image = image.to(device)
            label = label.to(device)
            y_predicted = model(image)
            idx = 0
            
            for key in keys_act:
                # print(key)
                # print(key[0:4])
                if key[0:4] == 'clas':
                    if direction == 'in':
                        dict_activations[key].append(save_activation.activations[idx][0].detach().cpu())
                    else:
                        dict_activations[key].append(save_activation.activations[idx].detach().cpu())
                else:
                    if direction == 'in':
                        dict_activations[key].append(flatten(save_activation.activations[idx][0].detach().cpu()))
                    else:
                        dict_activations[key].append(flatten(save_activation.activations[idx].detach().cpu()))
                
                idx += 1
            
            save_activation.clear()
            labels_predicted = y_predicted.argmax(axis = 1)
            dict_activations['pred'].append(labels_predicted.detach().cpu())
            results = labels_predicted == label
            dict_activations['results'].append(results.detach().cpu())
        return dict_activations
