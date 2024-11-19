# python stuff
from pathlib import Path as Path
from numpy.random import randint
from time import time

# Our stuff
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

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = 1#torch.cuda.device_count() - 3
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
    
    verbose = True 
    
    '''
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=verbose)

    layers_dict = {'classifier': [0, 3],
                   'features': [28]}
    model.set_target_layers(target_layers=layers_dict, verbose=verbose)

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

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model
            )
    
    loaders = ds.get_dataset_loaders()

    # copy dataset to coreVect dataset
    cv.get_coreVec_dataset(
            loaders = loaders,
            verbose = verbose
            ) 

    cv.get_activations(
            loaders = loaders,
            verbose = verbose
            )
    
    cv.get_coreVectors(
            reduct_matrices = model._svds,
            parser = parser_fn,
            verbose = verbose
            )
    
    cv_dl = cv.get_dataloaders(verbose=verbose)
    i = 0
    print('\nPrinting some corevecs')
    for data in cv_dl['val']:
        print(data['coreVectors']['classifier.0'])
        i += 1
        if i == 3: break

    cv.normalize_corevectors(wrt='train', verbose=verbose)
    i = 0
    for data in cv_dl['val']:
        print(data['coreVectors']['classifier.0'])
        i += 1
        if i == 3: break
    '''
    #--------------------------------
    # Peepholes
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = None 
            )
    cv.load_only(
            loaders = ['train', 'test', 'val'],
            verbose = True
            ) 
    _dlp = 9 
    bd = {'train':2**_dlp, 'test':2**_dlp, 'val':2**_dlp}
    cv_dl = cv.get_dataloaders(verbose=True, batch_dict=bd)
    n_classes = 300

    parser_cv = trim_corevectors
    parser_kwargs = {'layer': 'classifier.0', 'peep_size':300}
    cls_kwargs = {}#{'batch_size':256} 
    cls = tGMM(
            nl_classifier = 300,
            nl_model = n_classes,
            parser = parser_cv,
            parser_kwargs = parser_kwargs,
            cls_kwargs = cls_kwargs,
            device = device
            )
    
    t0 = time()
    cls.fit(dataloader = cv_dl['train'], verbose=verbose)
    print('Fitting time = ', time()-t0)
    
    cls.compute_empirical_posteriors(verbose=verbose)

    ph = Peepholes(
            path = phs_path,
            name = phs_name,
            classifier = cls,
            layer = 'classifier.0',
            device = device
            )
    input('Wait ph')
    ph.get_peepholes(
            loaders = cv_dl,
            verbose = verbose
            )
    input('Wait sc')

    ph.get_scores(verbose=verbose)

    i = 0
    print('\nPrinting some peeps')
    ph_dl = ph.get_dataloaders(verbose=verbose)
    for data in ph_dl['val']:
        print(data['classifier.0']['peepholes'])
        print(data['classifier.0']['score_max'])
        print(data['classifier.0']['score_entropy'])
        i += 1
        if i == 3: break

    ph.evaluate(
            layer = 'classifier.0',
            score_type = 'max',
            coreVectors = cv_dl
            )
