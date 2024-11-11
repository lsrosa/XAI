# python stuff
from pathlib import Path as Path
from numpy.random import randint

# Our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from coreVectors.coreVectors import CoreVectors 
from coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from classifier.classifier_base import trim_corevectors
from classifier.kmeans import KMeans 
from classifier.gmm import GMM 
from peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights

# Tuner
import tempfile
from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.cloudpickle as pickle

def peephole_wrap(config, **kwargs):
    peep_size = config['peep_size'] 
    n_cls = config['n_classifier']
    n_init = config['n_init']
    max_iter = config['max_iter']
    score_type = config['score_type']
    
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
    target_layer = 'classifier.0' 
    parser_kwargs = {'layer': target_layer, 'peep_size':peep_size}
    cls_kwargs = {'n_init':n_init, 'max_iter':max_iter} 
    ph_config_name = ph_name+f'.{peep_size}.{n_cls}.{n_init}.{max_iter}.{score_type}'

    g = list(ph_path.glob(f'{ph_config_name}.*')) 
    if len(g) > 0:
        print('Already run this configuration, skipping KMeans')
        ph = Peepholes(
                path = ph_path,
                name = ph_config_name,
                classifier = None,
                layer = target_layer,
                )
        
        ph.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                )
    else:
        cls = KMeans(
                nl_classifier = n_cls,
                nl_model = n_classes,
                parser = parser_cv,
                parser_kwargs = parser_kwargs,
                cls_kwargs = cls_kwargs
                )
                                                                         
        cls.fit(dataloader = cv_dl['train'], verbose=True)
        cls.compute_empirical_posteriors(verbose=True)
                                                                     
        ph = Peepholes(
                path = ph_path,
                name = ph_config_name,
                classifier = cls,
                layer = target_layer,
                )

        ph.get_peepholes(
                loaders = cv_dl,
                verbose = True
                )
    
        ph.get_scores(verbose=True)

    cok, cko = ph.evaluate(
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
            'cko': cok,
            },
            checkpoint=checkpoint
        )

    return 

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 3
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64
    model_dir = '/srv/newpenny/XAI/models'
    model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
    f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'
    
    svds_name = 'svds' 
    svds_path = Path.cwd()/'../data'
    
    cvs_name = 'corevectors'
    cvs_path = Path.cwd()/'../data/corevectors'
    
    phs_name = 'peepholes'
    phs_path = Path.cwd()/'../data/peepholes'

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
    print('\nPrinting some corevecs')
    for data in cv_dl['val']:
        print(data['coreVectors']['classifier.0'])
        i += 1
        if i == 3: break

    #--------------------------------
    # Tunning 
    #--------------------------------

    config = {
            'peep_size': tune.choice([2**i for i in range(2, 9)]),
            'n_classifier': tune.choice([2**i for i in range(2, 9)]),
            'n_init': tune.choice([50*i for i in range(1, 3)]),
            'max_iter': tune.choice([100*i for i in range(3, 4)]),
            'score_type': tune.choice(['max', 'entropy']), 
            }

    if device == 'cpu':
        resources = {"cpu": 1}
    else:
        resources = {"cpu": 16, "gpu": 2}

    hyper_params_file = phs_path/f'hyperparams.pickle'
    if hyper_params_file.exists():
        print("Already tunned parameters fount in %s. Skipping"%(hyper_params_file.as_posix()))
        exit()
   
    searcher = OptunaSearch(
            metric = ['cok', 'cko'],
            mode = ['max', 'max']
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
                num_samples = 1, 
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
    # TODO: check sharing tensor dicts

