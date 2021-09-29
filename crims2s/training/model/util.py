import collections.abc
from crims2s.training.util import find_checkpoint_file

import torch
import torch.nn as nn
import numpy as np

from typing import Mapping, TypeVar, Callable, Iterable, Hashable

from ...util import ECMWF_FORECASTS


class PytorchMultiplexer(nn.Module):
    """Model multiplexer that only works on pytorch tensor inputs."""

    def __init__(self, models):
        super().__init__()
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


class PytorchRolllingWindowMultiplexer(nn.Module):
    """Model multiplexer that, for a given key, runs a rolling window of models.
    
    The models_of_key callable gives the list of models that apply to a given key.
    For instance, if we want to call model 3 with a rolling window of two, then
    model_of_key(3) could return [1,2,3,4,5]."""

    def __init__(
        self,
        models_of_key: Callable[[str], Iterable[str]],
        models: Mapping[str, nn.Module],
    ):
        super().__init__()

        self.models_of_key = models_of_key
        self.models = nn.ModuleDict(models)

    def forward(self, key, *args):
        if isinstance(key, str):
            return self._compute_one_example(key, *args)
        elif isinstance(key, collections.abc.Iterable):
            outputs = []
            for i, k in enumerate(key):
                unbatched_args = [a[i] for a in args]
                outputs.append(self._compute_one_example(k, *unbatched_args))

            return torch.stack(outputs, dim=0)
        else:
            raise RuntimeError("Unregognized key type.")

    def _compute_one_example(self, key: str, *args):
        model_keys = self.models_of_key(key)
        models = [self.models[k] for k in model_keys]
        outputs = torch.stack([m(*args) for m in models], dim=0)

        return outputs.mean(dim=0)


class MonthlyMultiplexer(PytorchMultiplexer):
    def __init__(self, cls, *args, **kwargs):
        monthly_models = {f"{month:02}": cls(*args, **kwargs) for month in range(1, 13)}

        super().__init__(monthly_models)


class WeeklyMultiplexer(PytorchMultiplexer):
    def __init__(self, cls, *args, **kwargs):
        monthdays = [f"{m:02}{d:02}" for m, d in ECMWF_FORECASTS]
        weekly_models = {monthday: cls(*args, **kwargs) for monthday in monthdays}

        super().__init__(weekly_models)


class WeeklyRollingWindowMultiplexer(PytorchRolllingWindowMultiplexer):
    def __init__(self, window_size, cls, *args, **kwargs):
        self.window_size = window_size
        self.monthdays = [f"{m:02}{d:02}" for m, d in ECMWF_FORECASTS]

        weekly_models = {monthday: cls(*args, **kwargs) for monthday in self.monthdays}

        super().__init__(self.models_of_key, weekly_models)

    def models_of_key(self, key):
        left_lookup = self.window_size // 2
        right_lookup = self.window_size // 2 + 1

        padded_monthdays = [
            *self.monthdays[-left_lookup:],
            *self.monthdays,
            *self.monthdays[:right_lookup],
        ]

        i = self.monthdays.index(key)
        model_keys = padded_monthdays[i : i + self.window_size]

        return model_keys


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


class ModelWithCheckpoint(nn.Module):
    def __init__(self, model: nn.Module, checkpoint_path, remove_prefix="model."):
        super().__init__()
        self.model = model

        checkpoint_file = find_checkpoint_file(checkpoint_path)
        state_dict = torch.load(checkpoint_file)["state_dict"]

        state_dict = {k[len(remove_prefix) :]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
