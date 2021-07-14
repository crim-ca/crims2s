"""Utilities to ensure a smooth interaction between the project and Dask."""

import os
import yaml
import getpass
import dask.distributed
import dask_jobqueue
import dask.config


def load_project_defaults():
    """Make sure to update the dask defaults with the project configuration.
    Implementation suggested by https://docs.dask.org/en/latest/configuration.html."""

    fn = os.path.join(os.path.dirname(__file__), "crims2s_dask.yaml")

    with open(fn) as f:
        project_config = yaml.safe_load(f)
        dask.config.update(dask.config.config, project_config, priority="new")

        
def start_dask(conda_env: str = "s2s", jobs: int = 2):
    
    username = getpass.getuser()
    homedir = os.path.expanduser("~")
    
    print("Start Dask for", username, "with", conda_env, "conda environment")
    
    bash_profile: str = os.path.join(homedir, '.bash_profile')
    print('bash_profile:', bash_profile)
    
    env_extrat = ['source ' + bash_profile, 'conda activate ' + conda_env]    
    cluster = dask_jobqueue.SLURMCluster(env_extra=env_extrat)
    cluster.scale(jobs=jobs)

    return dask.distributed.Client(cluster)