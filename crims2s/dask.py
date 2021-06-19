"""Utilities to ensure a smooth interaction between the project and Dask."""

import os
import yaml

import dask.config


def load_project_defaults():
    """Make sure to update the dask defaults with the project configuration.
    Implementation suggested by https://docs.dask.org/en/latest/configuration.html."""

    fn = os.path.join(os.path.dirname(__file__), "crims2s_dask.yaml")

    with open(fn) as f:
        project_config = yaml.safe_load(f)
        dask.config.update(dask.config.config, project_config, priority="new")
