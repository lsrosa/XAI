# python stuff
import os
import json
from pathlib import Path as Path
from numpy.random import randint
from tqdm import tqdm

# "the our" stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from peepholes.peepholes import Peepholes
from clustering.clustering import Clustering
from clustering.clustering import prepare_data, compute_scores

# torch stuff
import torch
from tensordict import TensorDict
#from torchvision.models import vgg16, VGG16_Weights

# sklearn stuff
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    #--------------------------------
    # Parameters
    #--------------------------------
    # dnn info
    dnn_model = 'vgg16'
    # absolute paths
    abs_path = '/srv/newpenny/XAI/generated_data'

    #--------------------------------
    # Dataset
    #--------------------------------
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64
    ds = Cifar(dataset=dataset) # TODO: cambiare opzione path

    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            ) 
    #--------------------------------
    # "Ex" Peepholes
    #--------------------------------
    # data from "ex" peepholes
    print('Loading the ex-peepholes')
    phs_name = 'peepholes'
    phs_dir = os.path.join(abs_path, 'peepholes')
    peepholes = Peepholes(
            path = phs_dir,
            name = phs_name,
            )
    loaders = ds.get_dataset_loaders()
    
    # copy dataset to peepholes dataset
    peepholes.get_peep_dataset(
            loaders = loaders,
            verbose = True
            ) 
    ph_dl = peepholes.get_dataloaders(batch_size=128, verbose=True)
    # get list of available layers
    #available_layers = keys da qualche cosa tipo next(iter(ph_dl))
    available_layers = list(next(iter(ph_dl['train']))['peepholes'].keys())
    
    # SINGLE LAYER VERSION
    print('Computing confidence scores')
    # need to handle seed (for now: default)
    # init params
    k_list = [20, 50]
    n_clusters_list = [100, 150]
    layers_list = available_layers # ['classifier.0'] # la roba che vuoi
    algorithm = 'gmm'
    
    # check for existing results or init results container
    res_dir = 'clustering/confidence_scores'
    res_path = os.path.join(abs_path, res_dir)
    res_suffix = '.pth'
    res_filename = f'algorithm={algorithm}_dataset={dataset}_dnn={dnn_model}' # TODO: add seed
    tensor_dict_path = os.path.join(res_path, res_filename + res_suffix)
    
    meta_filename = '_'.join(['metadata', res_filename]) + '.json'
    meta_path = os.path.join(res_path, meta_filename)
    
    # check if results and metadata exist
    results_exist = os.path.exists(tensor_dict_path)
    metadata_exist = os.path.exists(meta_path)
    
    if results_exist:
        print('Results already present')
        all_scores = torch.load(tensor_dict_path)
    else:
        all_scores = TensorDict({}, batch_size=[])
    
    if metadata_exist:
        print('Loading related metadata')
        with open(meta_path, 'r') as json_file:
            metadata = json.load(json_file)
    else:
        metadata = {'k_values': [], 'n_clusters': [], 'layers': []}
    
    # loop over k and n_clusters
    for k in k_list:
        str_k = str(k) if not isinstance(k, str) else k
    
        if str_k not in all_scores.keys():
            all_scores.set(str_k, TensorDict({}, batch_size=[]))
    
        for n_clusters in n_clusters_list:
            str_n_clusters = str(n_clusters) if not isinstance(n_clusters, str) else n_clusters
    
            # check if the combination of k and n_clusters already exists in the scores
            if str_n_clusters in all_scores[str_k].keys():
                existing_layers = all_scores[str_k][str_n_clusters]['train'].keys()
    
                # check if only specific layers need to be computed
                if existing_layers and any(layer not in existing_layers for layer in layers_list):
                    # compute only the missing layers
                    for layer in layers_list:
                        if layer not in existing_layers:
                            data = prepare_data(ph_dl, [layer], k)
                            print(f'Clustering for layer={layer}')
                            compute_scores(k, n_clusters, algorithm, [layer], data, all_scores, metadata)
    
                else:
                    print(f'Skipping {algorithm} k={k}, n_clusters={n_clusters} for all layers')
                    continue  # skip to the next n_clusters if all layers have data
    
            else:
                print('Clustering')
                print(f'algorithm={algorithm}, k={k}, n_clusters={n_clusters}')
                # if not existing, create the subdict for n_clusters and compute all layers
                all_scores[str_k].set(str_n_clusters, TensorDict({
                    'train': TensorDict({}, batch_size=[]),
                    'val': TensorDict({}, batch_size=[])
                }, batch_size=[]))
                existing_layers = []  # no existing layers yet
    
                # prepare data for all splits
                data = prepare_data(ph_dl, layers_list, k)
                
                #print(len(data['core_vectors']['train'][layers_list[0]]))
                #print(len(data['true_labels']['train']))
                
                # compute scores for all layers and splits
                #for split in data['core_vectors'].keys():
                #for layer in layers_list:
                compute_scores(k, n_clusters, algorithm, layers_list, data, all_scores, metadata)
    
    # Save all scores and metadata after processing
    torch.save(all_scores, tensor_dict_path)
    with open(meta_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    
    print('Results and metadata saved.')
    
    # TODO
    # split data into correct/wrog predictions from DNN
    # set thresholds (on training scores) and compute retention fraction
    # get confusion matrix
    # COMBINED LAYERS VERSION