import collections.abc
from typing import Iterable
import torch
import torch.nn as nn
import numpy as np

from ...util import ECMWF_FORECASTS


class PytorchMultiplexer(nn.Module):
    """Model multiplexer that only works on pytorch tensor inputs."""

    def __init__(self, key, models):
        super().__init__()

        if isinstance(key, str):
            self.key_fn = lambda x: x[key]
        else:
            self.key_fn = key

        self.models = nn.ModuleDict(models)

    def forward(self, key, *args):
        if isinstance(key, str):
            model = self.models[key]
            return model(*args)
        if isinstance(key, collections.abc.Iterable):
            model_outputs = []
            for i, k in enumerate(key):
                unbatched_args = [a[i] for a in args]
                model = self.models[k]

                model_output = model(*unbatched_args)
                model_outputs.append(model_output)

            return torch.stack(model_outputs, dim=0)
        else:
            model = self.models[key]
            return model(*args)


class ModelMultiplexer(nn.Module):
    """Dispatch the training examples to multiple models depending on the example.
    For instance, we could use this to use a different model for every monthday forecast.

    Because it uses an arbitraty model for every sample, this module does not support batching.
    To use it, it is recommended to disable automatic batching on the dataloader."""

    def __init__(self, key, models):
        """Args:
            key: If a str, used as a key to fetch the model name from the example dict.
                 If a callable, called on the example and should return to model name to use.
            models: A mapping from model names to model instances. They keys should correspond to what is returned when applying key on the example."""
        super().__init__()

        if isinstance(key, str):
            self.key_fn = lambda x: x[key]
        else:
            self.key_fn = key

        self.models = nn.ModuleDict(models)

    def forward(self, example):
        model_name = self.key_fn(example)
        model = self.models[model_name]

        return model(example)


class WeeklyModel(ModelMultiplexer):
    def __init__(self, cls, **kwargs):
        monthdays = [f"{m:02}{d:02}" for m, d in ECMWF_FORECASTS]
        weekly_models = {monthday: cls(**kwargs) for monthday in monthdays}

        super().__init__("monthday", weekly_models)


class MonthlyModel(ModelMultiplexer):
    def __init__(self, cls, **kwargs):
        monthly_models = {f"{month:02}": cls(**kwargs) for month in range(1, 13)}

        super().__init__(lambda x: x["monthday"][:2], monthly_models)


def compute_edges_cdf_from_distribution(distribution, edges, regularization=0.0):
    edges_nan_mask = edges.isnan()
    edges[edges_nan_mask] = 0.0

    cdf = distribution.cdf(edges + regularization)
    edges[edges_nan_mask] = np.nan
    cdf[edges_nan_mask] = np.nan

    return cdf


def edges_cdf_to_terciles(edges_cdf):
    if len(edges_cdf.shape) == 5:
        stack_dim = 1
    else:
        stack_dim = 0

    return torch.stack(
        [edges_cdf[0], edges_cdf[1] - edges_cdf[0], 1.0 - edges_cdf[1],], dim=stack_dim
    )


class DistributionToTerciles(nn.Module):
    def __init__(self, regularization=0.0):
        super().__init__()
        self.regularization = regularization

    def forward(self, distribution, edges):
        edges_cdf = compute_edges_cdf_from_distribution(
            distribution, edges, self.regularization
        )
        return edges_cdf_to_terciles(edges_cdf)


class DistributionModelAdapter(nn.Module):
    """Convert a model that outputs distributions into a model that outputs terciles."""

    def __init__(self, model, tp_regularization=0.0):
        super().__init__()
        self.model = model
        self.t2m_to_terciles = DistributionToTerciles()
        self.tp_to_terciles = DistributionToTerciles(regularization=tp_regularization)

    def forward(self, example):
        t2m_dist, tp_dist = self.model(example)

        edges_t2m = example["edges_t2m"]
        edges_tp = example["edges_tp"]

        if len(edges_t2m.shape) == 5:
            """There is a batch dim but we need the egdges dim on the first dim."""
            edges_t2m = torch.transpose(edges_t2m, 0, 1)
            edges_tp = torch.transpose(edges_tp, 0, 1)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, edges_t2m)
        tp_terciles = self.tp_to_terciles(tp_dist, edges_tp)

        return t2m_terciles, tp_terciles
