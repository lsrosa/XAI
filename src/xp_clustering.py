from pathlib import Path
import os
import json
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from peepholes.peepholes import Peepholes
from clustering.clustering import Clustering
from clustering.clustering import prepare_data, compute_scores

if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    #---------------------------------------
    # Parameters and data
    #---------------------------------------
    abs_path = '/srv/newpenny/XAI/generated_data'
    dnn_model = 'vgg16'
   
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64
    ds = Cifar(dataset=dataset)
    
    ds.load_data(batch_size=bs, data_kwargs={'num_workers': 4, 'pin_memory': True}, seed=seed) 
    print('Loading the "ex"-peepholes')
    
    phs_name = 'peepholes'
    phs_dir = os.path.join(abs_path, 'peepholes')
    peepholes = Peepholes(path=phs_dir, name=phs_name)
    loaders = ds.get_dataset_loaders()
    
    # Copy dataset to peepholes dataset
    peepholes.get_peep_dataset(loaders=loaders, verbose=False) 
    ph_dl = peepholes.get_dataloaders(batch_size=128, verbose=False)
    
    available_layers = list(next(iter(ph_dl['train']))['peepholes'].keys())
    
    #---------------------------------------
    # "New" peepholes and confidence scores
    #---------------------------------------
    print('Computing peepholes and confidence scores')

    k_list = [20, 50, 70, 100]
    n_clusters_list = [50, 100, 150, 200]
    # TODO: use model._target_layers
    layers_list = available_layers
    algorithm = 'gmm'
    
    # check for existing results or init results container
    res_dir = 'clustering/confidence_scores'
    res_path = os.path.join(abs_path, res_dir)
    tensor_dict_path = os.path.join(res_path, f'algorithm={algorithm}_dataset={dataset}_dnn={dnn_model}.memmap')
    
    clustering = Clustering()
    clustering.prepare_data()
    clustering.fit_cluster()
    clustering.compute_scores()

    # TODO: remove all os, use Path instead
    if os.path.exists(tensor_dict_path):
        print('Results already present')
        all_scores = TensorDict.load_memmap(tensor_dict_path)
    else:
        all_scores = TensorDict({}, batch_size=[])

    # init the peephole container if not existing
    new_peep_dir = 'clustering/peepholes' 
    new_peep_path = os.path.join(abs_path, new_peep_dir) 
    new_peep_tensor_dict_path = os.path.join(new_peep_path, f'algorithm={algorithm}_dataset={dataset}_dnn={dnn_model}.memmap')
    
    if os.path.exists(new_peep_tensor_dict_path):
        print('New peepholes results already present')
        peephole_scores = TensorDict.load_memmap(new_peep_tensor_dict_path)
    else:
        # print('Initializing peephole container')
        peephole_scores = TensorDict({}, batch_size=[])

    ###################

    # main logic
    data = None
        
    for k in k_list:  # loop over core-vector dimension
        print(f'Core vector dim={k}')
        str_k = str(k)
            
        if str_k not in peephole_scores.keys():
            print(f'new key {k}')
            peephole_scores.set(str_k, TensorDict({}, batch_size=[]))
    
        for n_clusters in n_clusters_list:  # loop over n_clusters
            str_n_clusters = str(n_clusters)
    
            # initialize peephole_scores for n_clusters
            if str_n_clusters not in peephole_scores[str_k].keys():
                peephole_scores[str_k].set(str_n_clusters, TensorDict({
                    'train': TensorDict({}, batch_size=[]),
                    'val': TensorDict({}, batch_size=[]),
                }, batch_size=[]))
    
            # For both train and val splits, ensure layers exist
            for split in ['train', 'val']:
                if split not in peephole_scores[str_k][str_n_clusters].keys():
                    peephole_scores[str_k][str_n_clusters].set(split, TensorDict({}, batch_size=[]))
    
                existing_layers = peephole_scores[str_k][str_n_clusters][split].keys()
                layers_needed = [layer for layer in layers_list if layer not in existing_layers]
    
                # check if the existing layers are empty
                existing_layers_data = {
                    layer: peephole_scores[str_k][str_n_clusters][split][layer]
                    for layer in existing_layers if len(peephole_scores[str_k][str_n_clusters][split][layer]) > 0
                }
    
                # compute scores for missing or empty layers
                if layers_needed:
                     # check on data preparation that needs improvement
                    needs_preparation = False
                    if data is None:
                        needs_preparation = True
                    else:
                        for layer in layers_needed:
                            if data['core_vectors']['train'][layer].shape[1] != k:
                                needs_preparation = True
                                break  # No need to check further if we know preparation is needed
                    
                    if needs_preparation:
                        print(f'Preparing data for core-vector dimension {k}')
                        data = prepare_data(ph_dl, layers_needed, k)
                        compute_scores(k, 
                                   n_clusters, 
                                   algorithm, 
                                   layers_needed, 
                                   data, 
                                   peephole_scores, 
                                   all_scores, 
                                   compute_scores=True, 
                                   seed=42)
    
                # check if we need to compute scores for empty existing layers
                for layer in existing_layers_data.keys():
                    if existing_layers_data[layer].numel() == 0:  
                        data = prepare_data(ph_dl, [layer], k)
                        print(f'Recomputing scores for empty layer={layer} with n_clusters={n_clusters}')
                        compute_scores(k, 
                                       n_clusters, 
                                       algorithm, 
                                       [layer], 
                                       data, 
                                       peephole_scores, 
                                       all_scores, 
                                       compute_scores=True, 
                                       seed=42)
    
            # after processing layers, check if we need to initialize all_scores
            if str_n_clusters not in all_scores[str_k].keys() or not all(layer in peephole_scores[str_k][str_n_clusters]['train'].keys() and peephole_scores[str_k][str_n_clusters]['train'][layer].numel() > 0 for layer in layers_list):
                print(f'Initializing all_scores for k={k}, n_clusters={n_clusters}')
                print('CHECK THIS PART!')
                data = prepare_data(ph_dl, layers_list, k)
    
                n_samples_train = len(data['core_vectors']['train'][layers_list[0]])
                n_samples_val = len(data['core_vectors']['val'][layers_list[0]])
    
                all_scores[str_k].set(str_n_clusters, TensorDict({
                    'train': TensorDict({
                        layer: MMT.empty(shape=(n_samples_train,)) for layer in layers_list}, batch_size=[]),
                    'val': TensorDict({
                        layer: MMT.empty(shape=(n_samples_val,)) for layer in layers_list}, batch_size=[]),
                }, batch_size=[]))
    
                compute_scores(k, 
                               n_clusters, 
                               algorithm, 
                               layers_list, 
                               data, 
                               peephole_scores, 
                               all_scores, 
                               compute_scores=True, 
                               seed=42)
    
            # Save results
            peephole_scores.memmap(new_peep_tensor_dict_path, num_threads=4)
            all_scores.memmap(tensor_dict_path, num_threads=4)
            print('Results saved to memory-mapped tensor.')
    # TODO
    # split data into correct/wrog predictions from DNN
    # set thresholds (on training scores) and compute retention fraction
    # get confusion matrix
    # COMBINED LAYERS VERSION
