import numpy as np
import torch
import json
import pandas as pd

from stable_baselines3.common.env_util import make_vec_env

# Import our model
from fkine.learn import learn
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from utils import *

# Utility stuff
from utils import * 
from pathlib import Path as path
import os

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

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

home_dir = path.cwd()

def validate(models_dir, model_kwargs, n_samples, device=device):
    model_kwargs = model_kwargs.copy()
    model_kwargs.pop('env_models_home')

    fkine_model = model_kwargs.pop('model')
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_file = "fkine_%s"%(fkine_model)+_suffix

    n_dims = model_kwargs['n_dims'] 
    n_joints = model_kwargs['n_joints'] 
    
    env_kwargs={'model_file':home_dir/('rgym/envs/assets/reacher%dd%dj.xml'%(n_dims, n_joints))}
    env = make_env(**env_kwargs)
    obs = env.reset()[0]

    fkine_model_files = sorted(list(models_dir.glob("%s*.pt"%(fkine_file))))
    # For some really weird reason eval does not work before printing the classes
    # It probably has something to do with the class not being on the namespace before being called
    # Weird
    # TODO: maybe installing it as a package with pip solves the problem
    if fkine_model == 'FKineLinked':
        print('Using: ', FKineLinked)
    elif fkine_model == 'FKineMono':
        print('Using: ', FKineMono)

    losses = [] 
    for fkine_model_file in fkine_model_files:
        fkine_net = eval(fkine_model)(**model_kwargs, device=device)
        fkine_net.load_state_dict(torch.load(fkine_model_file, map_location=torch.device(device)))
        fkine_net = fkine_net.to(device)
        
        q, _, y, _ = env.get_wrapper_attr('sample_states')(n_samples, strategy='random')
        q, y = torch.Tensor(q).to(device), torch.Tensor(y).to(device)

        with torch.no_grad():
            if fkine_model == 'FKineLinked':
                y_pred, _ = fkine_net(q)
            elif fkine_model == 'FKineMono':
                y_pred = fkine_net(q)
        error = (y-y_pred).abs().norm(dim=1).mean()
        losses.append(error.detach().cpu().numpy())
    
    mean_loss = np.array(losses).mean()
    
    num_params = 0
    for p in fkine_net.parameters():
        num_params += p.numel()
    print("Validation Loss: ", mean_loss)
    return mean_loss, num_params

def learn_wrap(config, max_epochs, out_dir, model_kwargs, learn_kwargs, target_loss, n_runs, n_val_samples, device=device):
    tune_dir = home_dir/out_dir/('reacher%dd%dj'%(model_kwargs['n_dims'], model_kwargs['n_joints']))

    ls = learn_kwargs['learn_steps']
    _learn_kwargs = learn_kwargs.copy()
    _learn_kwargs['batch_size'] = config['batch_size'] 
     
    model_kwargs['lr'] = config['lr']
    model_kwargs['n_hidden'] = config['nh']
    model_kwargs['size_hidden'] = config['sh']
   
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = path(out_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
    else:
        start_epoch = 0
    
    _ls = 0
    for i in range(start_epoch, max_epochs):
        _ls += ls
        _learn_kwargs['learn_steps'] = _ls
        # create n_runs in the first pass, after that, refine automatically applys for all runs 
        if i == start_epoch:
            _learn_kwargs['append'] = True 
            _learn_kwargs['refine'] = False 
            for run in range(n_runs):
                learn(tune_dir/'models', tune_dir/'results', tune_dir/'plots', model_kwargs, _learn_kwargs, device=device) 
            _learn_kwargs['append'] = False 
            _learn_kwargs['refine'] = True
        else: 
            learn(tune_dir/'models', tune_dir/'results', tune_dir/'plots', model_kwargs, _learn_kwargs, device=device) 
        
        val_loss, num_params = validate(tune_dir/'models', model_kwargs, n_val_samples, device=device)
        
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(out_dir, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report({
                "loss": val_loss,
                'learn_steps': _ls,
                'num_params': num_params,
                },
                checkpoint=checkpoint
            )
        if val_loss < target_loss:
            print("Target loss achieved. Breaking")
            break
    return

if __name__ == '__main__':
    out_dir = home_dir/'tunning_fkine'
    tune_dir = home_dir/'results'/'tunning_fkine'
    out_dir.mkdir(exist_ok=True, parents=True)
    tune_dir.mkdir(exist_ok=True, parents=True)

    config = {
            'lr': tune.loguniform(1e-5, 1e-3),
            'nh': tune.choice([i for i in range(2, 8)]),
            'sh': tune.choice([2**i for i in range(3, 7)]),
            'batch_size': tune.choice([2**i for i in range(3, 7)]),
            }

    learn_kwargs = dict()
    learn_kwargs['seed'] = 1
    learn_kwargs['n_rollouts'] = 100
    learn_kwargs['learn_steps'] = 50
    learn_kwargs['n_envs'] = 32 
    learn_kwargs['n_iter'] = 25
    learn_kwargs['append'] = False
    learn_kwargs['refine'] = True 
    
    model_kwargs = dict()
    model_kwargs['env_models_home'] = path.cwd()
    tune_dir.mkdir(exist_ok=True, parents=True)
    if device == 'cpu':
        resources = {"cpu": 8}
    else:
        resources = {"cpu": 8, "gpu": 1}

    for model in ['FKineMono', 'FKineLinked']:
        for n_dims in [3]:#, 2]:
            for n_joints in [7, 6, 5, 4, 3, 2]:
                hyper_params_file = tune_dir/('reacher%dd%dj_%s_hyperparams.pickle'%(n_dims, n_joints, model))

                model_kwargs['model'] = model 
                model_kwargs['n_joints'] = n_joints 
                model_kwargs['n_dims'] = n_dims 
                if hyper_params_file.exists():
                    print("Already tunned parameters fount in %s. Skipping"%(hyper_params_file.as_posix()))
                    continue

                searcher = OptunaSearch(
                        metric = ['loss', 'learn_steps', 'num_params'],
                        mode = ['min', 'min', 'min']
                        )
                algo = ConcurrencyLimiter(searcher, max_concurrent=4)
                scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric="loss", mode="min")
                trainable = tune.with_resources(
                        partial(
                            learn_wrap,
                            max_epochs=40,
                            out_dir=out_dir,
                            model_kwargs=model_kwargs,
                            learn_kwargs=learn_kwargs,
                            target_loss=0.005,
                            n_runs=10,
                            n_val_samples=100,
                            device=device
                            ),
                        resources 
                        )
                tuner = tune.Tuner(
                        trainable,
                        tune_config=tune.TuneConfig(
                            search_alg=algo,
                            num_samples=50, 
                            scheduler=scheduler,
                            ),
                        run_config=train.RunConfig(
                            storage_path=out_dir/('reacher%dd%dj_%s_tunning_results'%(n_dims, n_joints, model)),
                            ),
                        param_space=config,
                        )
                result = tuner.fit()

                results_df = result.get_dataframe()
                print('results: ', results_df)
                results_df.to_pickle(hyper_params_file)

                os.system('git add %s'%hyper_params_file.as_posix())
                os.system('git commit -m "add hyper param"')
                os.system('git push lsrosa tunning')
