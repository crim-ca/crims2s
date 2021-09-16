import torch
import torch.nn as nn

from .util import ModelMultiplexer
from ...distribution import Gamma
from ...util import ECMWF_FORECASTS


__all__ = [
    "NormalGammaEMOS",
    "NormalNormalEMOS",
    "NormalCubeNormalEMOS",
    "NormalNormalMonthlyEMOS",
    "NormalGammaMonthlyEMOS",
    "NormalCubeNormalMonthlyEMOS",
    "NormalNormalMultiplexedEMOS",
    "NormalGammaMultiplexedEMOS",
    "NormalCubeNormalMultiplexedEMOS",
]


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

        shape = (2, 121, 240) if biweekly else (121, 240)

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

        shape = (2, 121, 240) if biweekly else (121, 240)

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
            "model_parameters_t2m_mu", "model_parameters_t2m_sigma", biweekly=biweekly
        )
        tp_model = NormalEMOSModel(
            "model_parameters_tp_mu", "model_parameters_tp_sigma", biweekly=biweekly
        )

        super().__init__(t2m_model, tp_model)


class NormalCubeNormalEMOS(TempPrecipEMOS):
    def __init__(self, biweekly=False):
        t2m_model = NormalEMOSModel(
            "model_parameters_t2m_mu", "model_parameters_t2m_sigma", biweekly=biweekly
        )
        tp_model = NormalEMOSModel(
            "model_parameters_tp_cube_root_mu",
            "model_parameters_tp_cube_root_sigma",
            biweekly=biweekly,
        )

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


class NormalCubeNormalMultiplexedEMOS(MultiplexedEMOSModel):
    def __init__(self, biweekly=False):
        super().__init__(NormalCubeNormalEMOS, biweekly=biweekly)


class NormalGammaMonthlyEMOS(ModelMultiplexer):
    MODEL = NormalGammaEMOS

    def __init__(self, biweekly=False):
        monthly_models = {
            f"{month:02}": self.MODEL(biweekly=biweekly) for month in range(1, 13)
        }

        super().__init__(lambda x: x["monthday"][:2], monthly_models)


class NormalNormalMonthlyEMOS(NormalGammaMonthlyEMOS):
    MODEL = NormalNormalEMOS


class NormalCubeNormalMonthlyEMOS(NormalGammaMonthlyEMOS):
    MODEL = NormalCubeNormalEMOS
