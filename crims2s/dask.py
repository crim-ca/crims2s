"""Utilities to ensure a smooth interaction between the project and Dask."""

import os
import dask.distributed
import dask_jobqueue
import dask.config
import logging
import getpass
import yaml


_logger = logging.getLogger(__name__)


def load_project_defaults():
    """Make sure to update the dask defaults with the project configuration.
    Implementation suggested by https://docs.dask.org/en/latest/configuration.html."""

    fn = os.path.join(os.path.dirname(__file__), "crims2s_dask.yaml")

    with open(fn) as f:
        project_config = yaml.safe_load(f)
        dask.config.update(dask.config.config, project_config, priority="new")


def create_dask_cluster(conda_env: str = "s2s"):
    username = getpass.getuser()
    homedir = os.path.expanduser("~")

    _logger.debug(f"Start Dask for {username} with {conda_env} conda environment")

    bash_profile: str = os.path.join(homedir, '.bash_profile')
    _logger.debug(f'bash_profile: {bash_profile}')

    env_extrat = ['source ' + bash_profile, 'conda activate ' + conda_env]
    return dask_jobqueue.SLURMCluster(env_extra=env_extrat)