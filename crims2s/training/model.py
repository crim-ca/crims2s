import torch
import torch.nn as nn

from ..distribution import Gamma
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


class NormalEMOSModel(nn.Module):
    def __init__(self, mu_key, sigma_key, biweekly=False):
        super().__init__()

        shape = (3, 121, 240) if biweekly else (121, 240)

        self.mu_model = LinearModel(*shape)
        self.sigma_model = LinearModel(*shape, fill_intercept=1.0)

        self.mu_key = mu_key
        self.sigma_key = sigma_key

    def forward(self, example):
        forecast_mu, forecast_sigma = (
            example[self.mu_key],
            example[self.sigma_key],
        )

        mu = self.mu_model(forecast_mu)
        sigma = self.sigma_model(forecast_sigma)
        sigma = torch.clip(sigma, min=1e-6)

        return torch.distributions.Normal(loc=mu, scale=sigma)


class GammaEMOSModel(nn.Module):
    def __init__(self, alpha_key, beta_key, biweekly=False, regularization=1e-9):
        super().__init__()

        shape = (3, 121, 240) if biweekly else (121, 240)

        self.regularization = regularization

        self.alpha_model = LinearModel(*shape)
        self.beta_model = LinearModel(*shape, fill_intercept=1.0)

        self.alpha_key = alpha_key
        self.beta_key = beta_key

    def forward(self, example):
        forecast_alpha, forecast_beta = (
            example[self.alpha_key],
            example[self.beta_key],
        )

        alpha = self.alpha_model(forecast_alpha)
        alpha = torch.clip(alpha, min=self.regularization)

        beta = self.beta_model(forecast_beta)
        beta = torch.clip(beta, min=self.regularization)

        # Here we use a home grown version of Gamma until they implement CDF in PyTorch.
        return Gamma(alpha, beta)


class TempPrecipEMOS(nn.Module):
    """EMOS model for temperature and precipitation.

    Args:
        biweekly: If True, train a separate model for every 2 weeks period."""

    def __init__(self, t2m_model, tp_model):
        super().__init__()

        self.t2m_model = t2m_model
        self.tp_model = tp_model

    def forward(self, example):
        t2m_dist = self.t2m_model(example)
        tp_dist = self.tp_model(example)

        return t2m_dist, tp_dist


class NormalNormalEMOS(TempPrecipEMOS):
    def __init__(self, biweekly=False):
        t2m_model = NormalEMOSModel(
            "model_t2m_mu", "model_t2m_sigma", biweekly=biweekly
        )
        tp_model = NormalEMOSModel("model_tp_mu", "model_tp_sigma", biweekly=biweekly)

        super().__init__(t2m_model, tp_model)


class NormalGammaEMOS(TempPrecipEMOS):
    def __init__(self, biweekly=False):
        t2m_model = NormalEMOSModel(
            "model_parameters_t2m_mu", "model_parameters_t2m_sigma", biweekly=biweekly
        )
        tp_model = GammaEMOSModel(
            "model_parameters_tp_alpha", "model_parameters_tp_beta", biweekly=biweekly
        )
        super().__init__(t2m_model, tp_model)


class TempPrecipEMOSGamma(nn.Module):
    """EMOS model for temperature and precipitation. Swap out the placeholder normal
    distribution that we used for the precipitation. Use a Gamma instead.

    Args:
        biweekly: If True, train a separate model for every 2 weeks period."""

    def __init__(self, biweekly=False):
        super().__init__()

        shape = (3, 121, 240) if biweekly else (121, 240)

        self.tp_alpha_model = LinearModel(*shape)
        self.tp_beta_model = LinearModel(*shape, fill_intercept=1.0)

        self.t2m_mu_model = LinearModel(*shape)
        self.t2m_sigma_model = LinearModel(*shape, fill_intercept=1.0)

    def forward(self, example):
        forecast_tp_alpha, forecast_tp_beta = (
            example["model_tp_alpha"],
            example["model_tp_beta"],
        )
        forecast_t2m_mu, forecast_t2m_sigma = (
            example["model_t2m_mu"],
            example["model_t2m_sigma"],
        )

        tp_alpha = self.tp_mu_model(forecast_tp_alpha)
        tp_alpha = torch.clip(tp_alpha, min=1e-6)

        tp_beta = self.tp_sigma_model(forecast_tp_beta)
        tp_beta = torch.clip(tp_beta, min=1e-6)

        t2m_mu = self.t2m_mu_model(forecast_t2m_mu)
        t2m_sigma = self.t2m_sigma_model(forecast_t2m_sigma)
        t2m_sigma = torch.clip(t2m_sigma, min=1e-6)

        tp_dist = torch.distributions.Gamma(tp_alpha, tp_beta)
        t2m_dist = torch.distributions.Normal(loc=t2m_mu, scale=t2m_sigma)

        return t2m_dist, tp_dist


class MultiplexedEMOSModel(ModelMultiplexer):
    """A collection of EMOS models: one for each forecast that ECMWF does each year."""

    def __init__(self, forecast_model, biweekly=False):
        monthdays = [f"{m:02}{d:02}" for m, d in ECMWF_FORECASTS]
        weekly_models = {
            monthday: forecast_model(biweekly=biweekly) for monthday in monthdays
        }

        super().__init__("monthday", weekly_models)


class NormalGammaMultiplexedEMOS(MultiplexedEMOSModel):
    def __init__(self, biweekly=False):
        super().__init__(NormalGammaEMOS, biweekly=biweekly)


class NormalNormalMultiplexedEMOS(MultiplexedEMOSModel):
    def __init__(self, biweekly=False):
        super().__init__(NormalNormalEMOS, biweekly=biweekly)


class NormalGammaMonthlyEMOS(ModelMultiplexer):
    def __init__(self, biweekly=False):
        monthly_models = {
            f"{month:02}": NormalGammaEMOS(biweekly=biweekly) for month in range(1, 13)
        }

        super().__init__(lambda x: x["monthday"][:2], monthly_models)
