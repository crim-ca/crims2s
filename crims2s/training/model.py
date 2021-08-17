import torch
import torch.nn as nn

from ..util import ECMWF_FORECASTS


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


class LinearModel(nn.Module):
    """Slight variation on the nn.Linear module. The main difference is that it uses
    an element-wise mapping instead of a matrix multiplication."""

    def __init__(self, *shape, fill_weights=1.0, fill_intercept=0.0):
        super().__init__()

        self.weights = nn.Parameter(torch.full(shape, fill_weights))
        self.intercept = nn.Parameter(torch.full(shape, fill_intercept))

    def forward(self, x):
        return self.intercept + self.weights * x


class TempPrecipEMOS(nn.Module):
    """EMOS model for temperature and precipitation.

    Args:
        biweekly: If True, train a separate model for every 2 weeks period."""

    def __init__(self, biweekly=False):
        super().__init__()

        shape = (3, 121, 240) if biweekly else (121, 240)

        self.tp_mu_model = LinearModel(*shape)
        self.tp_sigma_model = LinearModel(*shape, fill_intercept=1.0)

        self.t2m_mu_model = LinearModel(*shape)
        self.t2m_sigma_model = LinearModel(*shape, fill_intercept=1.0)

    def forward(self, example):
        forecast_tp_mu, forecast_tp_sigma = (
            example["model_tp_mu"],
            example["model_tp_sigma"],
        )
        forecast_t2m_mu, forecast_t2m_sigma = (
            example["model_t2m_mu"],
            example["model_t2m_sigma"],
        )

        tp_mu = self.tp_mu_model(forecast_tp_mu)
        tp_sigma = self.tp_sigma_model(forecast_tp_sigma)
        tp_sigma = torch.clip(tp_sigma, min=1e-6)

        t2m_mu = self.t2m_mu_model(forecast_t2m_mu)
        t2m_sigma = self.t2m_sigma_model(forecast_t2m_sigma)
        t2m_sigma = torch.clip(t2m_sigma, min=1e-6)

        tp_dist = torch.distributions.Normal(loc=tp_mu, scale=tp_sigma)
        t2m_dist = torch.distributions.Normal(loc=t2m_mu, scale=t2m_sigma)

        return t2m_dist, tp_dist


class MultiplexedEMOSModel(ModelMultiplexer):
    """A collection of EMOS models: one for each forecast that ECMWF does each year."""

    def __init__(self, biweekly=True):
        monthdays = [f"{m:02}{d:02}" for m, d in ECMWF_FORECASTS]
        weekly_models = {
            monthday: TempPrecipEMOS(biweekly=biweekly) for monthday in monthdays
        }

        super().__init__("monthday", weekly_models)

