# python stuff
from pathlib import Path as Path
from numpy.random import randint
import pickle
from time import time
from itertools import product
import random
import matplotlib.pyplot as plt

# our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from coreVectors.coreVectors import CoreVectors 
from coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from classifier.classifier_base import trim_corevectors
from classifier.kmeans import KMeans 
from classifier.gmm import GMM 
from classifier.tkmeans import KMeans as tKMeans 
from classifier.tgmm import GMM as tGMM 
from classifier.tkmeans import KMeans as tKMeans 
from classifier.tgmm import GMM as tGMM 
from peepholes.peepholes import Peepholes
from utils.testing import trim_dataloaders

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
#from torch.utils.data import random_split, DataLoader

# tuner
# import tempfile
# from functools import partial
# from ray import tune
# from ray import train
# from ray.train import Checkpoint, get_checkpoint
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.search.optuna import OptunaSearch

if __name__ == "__main__":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        #cuda_index = 5 #torch.cuda.device_count() - 2
        #device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        # model parameters
        dataset = 'CIFAR10' 
        ds_path = f'/srv/newpenny/dataset/{dataset}'

        # pretrained = True
        # seed = 29
        # bs = 64
        model_id = 'vgg16'
        model_dir = '/srv/newpenny/XAI/models'
        with open(Path(model_dir)/'model_config.pkl', 'rb') as f:
                model_config = pickle.load(f)

        model_name = model_config[model_id][dataset]

        svds_name = 'svds' 
        svds_path = Path.cwd()/f'../data/svds/{dataset}/{model_id}'

        cvs_name = 'corevectors'
        cvs_path = Path.cwd()/f'../data/corevectors/{dataset}/{model_id}'

        cls_type = 'tKMeans' # 'tGMM'
        phs_name = 'peepholes'
        phs_path = Path.cwd()/f'../data/peepholes/{dataset}/{model_id}/{cls_type}'

        verbose = True 


        #--------------------------------
        # Dataset 
        #--------------------------------

        # pretrained = True
        seed = 29
        bs = 256

        ds = Cifar(
                data_path = ds_path,
                dataset=dataset
                )
        ds.load_data(
                batch_size = bs, # bs # prendere da qualche config 
                data_kwargs = {'num_workers': 4, 'pin_memory': True},
                seed = seed, # prendere da qualche config
                )

        #--------------------------------
        # Model 
        #--------------------------------

        # if model_id=='vgg16':
        nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        in_features = 4096
        n_classes = len(ds.get_classes()) 
        nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
        model = ModelWrap(device=device)
        model.set_model(model=nn, path=model_dir, name=model_name, verbose=False)

        target_layers = [
                #'classifier.0',
                #'classifier.3',
                'features.28', 
                'features.26',
                'features.24',
                #'features.7'
                ]

        model.set_target_layers(target_layers=target_layers, verbose=verbose)
        '''
        direction = {'save_input':True, 'save_output':False}
        model.add_hooks(**direction, verbose=False) 

        dry_img, _ = ds._train_ds.dataset[0]
        dry_img = dry_img.reshape((1,)+dry_img.shape)
        model.dry_run(x=dry_img)
        
        #--------------------------------
        # SVDs 
        #--------------------------------
        print('target layers: ', model.get_target_layers()) 
        model.get_svds(path=svds_path, name=svds_name, verbose=verbose)
        for k in model._svds.keys():
                for kk in model._svds[k].keys():
                        print('svd shapes: ', k, kk, model._svds[k][kk].shape)


        # --------------------------------
        # CoreVectors 
        # --------------------------------
        ds_loaders = ds.get_dataset_loaders()
        
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                device = device
                )

        with corevecs as cv: 
                # copy dataset to coreVect dataset
                cv.get_coreVec_dataset(
                        loaders = ds_loaders, 
                        verbose = verbose
                        ) 

                cv.get_activations(
                        batch_size = bs,
                        loaders = ds_loaders,
                        verbose = verbose
                        )

                cv.get_coreVectors(
                        batch_size = bs,
                        reduct_matrices = model._svds,
                        parser = parser_fn,
                        verbose = verbose
                        )

                cv_dl = cv.get_dataloaders(verbose=verbose)

                i = 0
                print('\nPrinting some corevecs')
                for data in cv_dl['train']:
                        print(data['coreVectors'][target_layers[0]])
                        i += 1
                        if i == 3: break

                cv.normalize_corevectors(
                        wrt='train',
                        verbose=verbose
                        )
                i = 0
                for data in cv_dl['train']:
                        print(data['coreVectors'][target_layers[0]][34:56,:])
                        i += 1
                        if i == 3: break
        # quit()
        '''
        #--------------------------------
        # Peepholes
        #--------------------------------
        ps = [2**i for i in range(4, 10)] + [1000]
        ncls = [2**i for i in range(4, 9)]
        all_combinations = list(product(ps, ncls))

        for peep_size, n_cls in all_combinations:
                ph_config_name = phs_name+f'.{peep_size}.{n_cls}'

                for layer in target_layers:
                        n_classes = 10
                        parser_cv = trim_corevectors
                        parser_kwargs = {'layer': layer, 'peep_size':peep_size}
                        cls_kwargs = {} # {'batch_size':256} 
                        if cls_type=='tGMM':
                                cls = tGMM(
                                        nl_classifier = n_cls,
                                        nl_model = n_classes,
                                        parser = parser_cv,
                                        parser_kwargs = parser_kwargs,
                                        cls_kwargs = cls_kwargs,
                                        device = device
                                        )
                        elif cls_type=='tKMeans':
                                cls = tKMeans(
                                        nl_classifier = n_cls,
                                        nl_model = n_classes,
                                        parser = parser_cv,
                                        parser_kwargs = parser_kwargs,
                                        cls_kwargs = cls_kwargs,
                                        device = device
                                        )

                        corevecs = CoreVectors(
                                path = cvs_path,
                                name = cvs_name,
                                device = device 
                                )
                        
                        peepholes = Peepholes(
                                path = phs_path,
                                # name = phs_name,
                                name = ph_config_name,
                                classifier = cls,
                                # classifier = None,
                                layer = layer,
                                device = device
                                )

                        with corevecs as cv, peepholes as ph:
                                cv.load_only(
                                        loaders = ['train', 'test', 'val'],
                                        verbose = True
                                        ) 
                                
                                cv_dl = cv.get_dataloaders(
                                        batch_size = bs,
                                        verbose = True,
                                        )
                                
                                i = 0
                                print('\nPrinting some corevecs')
                                for data in cv_dl['val'].dataset['coreVectors']:
                                        print('cvs\n', data[layer])
                                        i += 1
                                        if i == 3: break
                                
                                t0 = time()
                                cls.fit(dataloader = cv_dl['train'], verbose=True)
                                print('Fitting time = ', time()-t0)
                                
                                cls.compute_empirical_posteriors(verbose=verbose)
                                plt.figure()
                                plt.imshow(cls._empp)
                                plt.savefig(Path('results_dfs')/f'empp_{cls_type}_{layer}_{ph_config_name}.png')
                        
                                ph.get_peepholes(
                                        loaders = cv_dl,
                                        verbose = verbose
                                        )
                        
                                ph.get_scores(
                                        # batch_size = 256,
                                        verbose=verbose
                                        )
                                # input('Wait sc')
                        
                        
                                i = 0
                                print('\nPrinting some peeps')
                                ph_dl = ph.get_dataloaders(verbose=verbose)
                                for data in ph_dl['val']:
                                        print('phs\n', data[layer]['peepholes'])
                                        print('max\n', data[layer]['score_max'])
                                        print('ent\n', data[layer]['score_entropy'])
                                        i += 1
                                        if i == 3: break
                        
                                #ph.evaluate(
                                #        layer = layer,
                                #        score_type = 'max',
                                #        coreVectors = cv_dl
                                #        )

                                #ph.evaluate(
                                #        layer = layer,
                                #        score_type = 'entropy',
                                #        coreVectors = cv_dl
                                #        )
                        
            

