# torch stuff
import torch
from cuda_selector import auto_cuda

# python stuff
from pathlib import Path as Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

# Our stuff
from coreVectors.coreVectors import CoreVectors 
from peepholes.peepholes import Peepholes

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    phs_name = 'peepholes'
    phs_path = Path.cwd()/'../data/peepholes'

    cvs_name = 'corevectors'
    cvs_path = Path.cwd()/'../data/corevectors'
    
    names = list(phs_path.glob(phs_name+'*.train'))
    names = [p.name.replace('.train','') for p in names]
    
    bs = 512 
    target_layer = 'classifier.0' 
    score_type = 'entropy'

    for name in names:
        break
        plot_file = (phs_path/(name+'.png'))
        if plot_file.exists():
            print(f'{plot_file} exists. skipping.')
            continue
        
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                device = device
                )

        peepholes = Peepholes(
                path = phs_path,
                name = name,
                classifier = None,
                layer = target_layer,
                device = device
                )

        with corevecs as cv, peepholes as ph: 
            cv.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True
                    ) 
                                                                    
            ph.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True
                    )
            
            cv_dl = cv.get_dataloaders(batch_size=bs, verbose=True)
            mok, sok, mko, sko = ph.evaluate_dists(
                score_type = score_type,
                coreVectors = cv_dl,
                bins = 20
                )

    hyperp_file = phs_path/'hyperparams.pickle'
    rdf = pd.read_pickle(hyperp_file)
    metrics = np.hstack((rdf['mok'].values , rdf['sok'].values, rdf['mko'].values, rdf['sko'].values))
    configs = np.hstack((rdf['config/peep_size'].values, rdf['config/n_classifier'].values))
    
    fig, axs = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(4*4, 2*4))
    for m in range(metrics.shape[1]):
        for c in range(configs.shape[1]):
            ax = axs[c][m]
            
