# python stuff
from pathlib import Path as Path
from numpy.random import randint
import pickle
from time import time
from itertools import product
import random

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
from peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import random_split, DataLoader

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
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    # model parameters
    dataset = 'CIFAR100' 
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

    cls_type = 'tKMeans'
    phs_name = 'peepholes'
    phs_path = Path.cwd()/f'../data/peepholes/{dataset}/{model_id}/{cls_type}'
    
    verbose = True 

    
    #--------------------------------
    # Dataset 
    #--------------------------------
    
    # pretrained = True
    seed = 29
    bs = 64
    
    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )
    ds.load_data(
            batch_size = 256, # bs # prendere da qualche config 
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
            'classifier.0',
            'classifier.3',
            'features.28', 
            'features.26',
            #'features.7'
            ]
    
    model.set_target_layers(target_layers=target_layers, verbose=verbose)
    
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
                batch_size = 256,
                loaders = ds_loaders,
                verbose = verbose
                )
        cv.get_coreVectors(
                batch_size = 256,
                reduct_matrices = model._svds,
                parser = parser_fn,
                verbose = verbose
                )
        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['test']:
            print(data['coreVectors']['features.28'])
            i += 1
            if i == 3: break

        cv.normalize_corevectors(
                wrt='test',
                verbose=verbose
                )
        i = 0
        for data in cv_dl['test']:
            print(data['coreVectors']['features.28'][34:56,:])
            i += 1
            if i == 3: break
    

    #--------------------------------
    # Peepholes
    #--------------------------------
    ps = [2**i for i in range(3, 9)]
    ncls = [2**i for i in range(6, 9)]
    all_combinations = list(product(ps, ncls))

    for peep_size, n_cls in all_combinations[:1]:
        ph_config_name = phs_name+f'.{peep_size}.{n_cls}'
        
        for layer in target_layers:
                n_classes = 100
                parser_cv = trim_corevectors
                parser_kwargs = {'layer': layer, 'peep_size':32}
                cls_kwargs = {}#{'batch_size':256} 
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
                        
                        t0 = time()
                        cls.fit(dataloader = cv_dl['train'], verbose=verbose)
                        print('Fitting time = ', time()-t0)
                        
                        cls.compute_empirical_posteriors(verbose=verbose)
                
                        ph.get_peepholes(
                                loaders = cv_dl,
                                verbose = verbose
                                )
                
                        ph.get_scores(
                                batch_size = 256,
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
                
                        ph.evaluate(
                                layer = layer,
                                # score_type = 'max',
                                score_type = 'max',
                                coreVectors = cv_dl
                                )

                        ph.evaluate(
                                layer = layer,
                                score_type = 'entropy',
                                coreVectors = cv_dl
                                )
                

'''
def peephole_wrap(config, **kwargs):
    peep_size = config['peep_size'] 
    n_cls = config['n_classifier']
    score_type = config['score_type']
    target_layer = config['target_layer']
    cls_type = config['cls_type']
    
    cv_dl = kwargs['corevec_dataloader']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = path(ph_path) / "tune_checkpoint.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]

    #--------------------------------
    # Peepholes
    #--------------------------------
    parser_cv = trim_corevectors
    n_classes = 100
    # target_layer = 'features.28' #'classifier.3' 
    parser_kwargs = {'layer': target_layer, 'peep_size':peep_size}
    cls_kwargs = {}#{'n_init':n_init, 'max_iter':max_iter} 
    ph_config_name = ph_name+f'.{peep_size}.{n_cls}.{score_type}'

    g = list(ph_path.glob(f'{ph_config_name}.*')) 
    if len(g) > 0:
        print('Already run this configuration, skipping peepholes computation')
        ph = Peepholes(
                path = ph_path,
                name = ph_config_name,
                classifier = None,
                layer = target_layer,
                device = device
                )
        
        ph.load_only(
                #loaders = ['train', 'test', 'val'],
                loaders = ['train', 'val'],
                verbose = True
                )
    else:
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
                                                                            
        cls.fit(dataloader = cv_dl['train'], verbose=True)
        cls.compute_empirical_posteriors(verbose=True)
                                                                     
        ph = Peepholes(
                path = ph_path,
                name = ph_config_name,
                classifier = cls,
                layer = target_layer,
                device = device
                )

        ph.get_peepholes(
                loaders = cv_dl,
                verbose = True
                )
    
        ph.get_scores(verbose=True)

    cok, cko, acc_gain = ph.evaluate(
        score_type = score_type,
        coreVectors = cv_dl
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "tune_checkpoint.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(ph_path, fp)
                                                               
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report({
            "cok": cok,
            "cko": cko,
            "acc_gain": acc_gain
            },
            checkpoint=checkpoint
        )

    return 

if __name__ == "__main__":
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    cuda_index = 4 #torch.cuda.device_count() - 3
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    dataset = 'CIFAR10' 
    ds_path = f'/srv/newpenny/dataset/{dataset}'

    # model parameters
    pretrained = True
    seed = 29
    bs = 64
    model_dir = '/srv/newpenny/XAI/models'
    model_id = 'vgg16'

    with open(Path(model_dir)/'model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)
    
    model_name = model_config[model_id][dataset]
    
    svds_name = 'svds' 
    # svds_path = Path.cwd()/'../data'
    svds_path = '/srv/newpenny/XAI/generated_data/svds' #
    
    cvs_name = 'corevectors'
    #cvs_path = Path.cwd()/f'../data/corevectors/{dataset}/{model_id}'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}'
    
    cls_type = 'tKMeans'
    phs_name = 'peepholes'
    phs_path = Path.cwd()/f'../data/peepholes/{dataset}/{model_id}/{cls_type}'
    phs_path.mkdir(parents=True, exist_ok=True)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = None 
            )
    
    # load corevds 
    cv.load_only(
            loaders = ['train', 'test', 'val'],
            verbose = True
            ) 

    cv_dl = cv.get_dataloaders(verbose=True)
    i = 0
    target_layer = 'classifier.0'
    print('\nPrinting some corevecs')
    for data in cv_dl['val']:
        print(data['coreVectors'][target_layer])
        i += 1
        if i == 3: break
        
    #--------------------------------
    # Tunning 
    #--------------------------------

    config = {
            'cls_type': 'tKMeans',
            'target_layer': target_layer,
            'peep_size': tune.choice([2**i for i in range(4, 10)]),
            'n_classifier': tune.choice([2**i for i in range(4, 9)]),
            'score_type': tune.choice(['max', 'entropy']), 
            }

    if device == 'cpu':
        resources = {"cpu": 1}
    else:
        resources = {"cpu": 16, "gpu": 5}

    hyper_params_file = phs_path/f'hyperparams.pickle'
    if hyper_params_file.exists():
        print("Already tuned parameters found in %s. Skipping"%(hyper_params_file.as_posix()))
        exit()
   
    searcher = OptunaSearch(
            metric = ['cok', 'cko', 'acc_gain'],
            mode = ['max', 'max', 'max']
            )
    
    algo = ConcurrencyLimiter(searcher, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric="cok", mode="max") 
    tune_storage_path = Path.cwd()/'../data/tuning'
    trainable = tune.with_resources(
            partial(
                peephole_wrap,
                device = device,
                corevec_dataloader = cv_dl,
                ph_path = phs_path,
                ph_name = phs_name
                ),
            resources 
            )

    tuner = tune.Tuner(
            trainable,
            tune_config = tune.TuneConfig(
                search_alg = algo,
                num_samples = 15, 
                scheduler = scheduler,
                ),
            run_config = train.RunConfig(
                storage_path = tune_storage_path
                ),
            param_space = config,
            )
    result = tuner.fit()

    results_df = result.get_dataframe()
    print('results: ', results_df)
    results_df.to_pickle(hyper_params_file)
    
    # TODO: use GPU for peephole computations

'''
