import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_backward_hook

from .util import (
    DistributionModelAdapter,
    ModelMultiplexer,
    PytorchMultiplexer,
    WeeklyModel,
)
from ...distribution import Gamma
from ...util import ECMWF_FORECASTS


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
    def __init__(self, biweekly=False, regularization=1e-9):
        super().__init__()

        shape = (2, 121, 240) if biweekly else (121, 240)

        self.mu_model = LinearModel(*shape)
        self.sigma_model = LinearModel(*shape, fill_weights=1.2)

        self.regularization = regularization

    def forward(self, forecast_mu, forecast_sigma):
        mu = self.mu_model(forecast_mu)

        sigma = self.sigma_model(forecast_sigma)
        sigma = torch.clip(sigma, min=1e-6)

        return torch.distributions.Normal(loc=mu, scale=sigma)


class GammaEMOSModel(nn.Module):
    def __init__(self, biweekly=False, regularization=1e-9):
        super().__init__()

        shape = (2, 121, 240) if biweekly else (121, 240)

        self.regularization = regularization

        self.alpha_model = LinearModel(*shape)
        self.beta_model = LinearModel(*shape, fill_intercept=1.0)

    def forward(self, forecast_alpha, forecast_beta):
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

    def __init__(
        self, t2m_model, tp_model,
    ):
        super().__init__()

        self.t2m_model = t2m_model
        self.tp_model = tp_model

    def forward(self, example):
        t2m_dist = self.t2m_model(example)
        tp_dist = self.tp_model(example)

        return t2m_dist, tp_dist


class NormalNormalEMOS(TempPrecipEMOS):
    def __init__(self, biweekly=False, regularization=1e-9):
        t2m_model = MonthlyNormalEMOSModel(
            "model_parameters_t2m_mu",
            "model_parameters_t2m_sigma",
            biweekly=biweekly,
            regularization=regularization,
        )
        tp_model = MonthlyNormalEMOSModel(
            "model_parameters_tp_mu",
            "model_parameters_tp_sigma",
            biweekly=biweekly,
            regularization=regularization,
        )

        super().__init__(
            t2m_model, tp_model,
        )


class NormalCubeNormalEMOS(TempPrecipEMOS):
    def __init__(self, biweekly=False, regularization=1e-9):
        t2m_model = MonthlyNormalEMOSModel(
            "model_parameters_t2m_mu",
            "model_parameters_t2m_sigma",
            biweekly=biweekly,
            regularization=regularization,
        )
        tp_model = MonthlyNormalEMOSModel(
            "model_parameters_tp_cube_root_mu",
            "model_parameters_tp_cube_root_sigma",
            biweekly=biweekly,
            regularization=regularization,
        )

        super().__init__(
            t2m_model, tp_model,
        )


class MonthlyMultiplexer(PytorchMultiplexer):
    def __init__(self, cls, *args, **kwargs):
        monthly_models = {f"{month:02}": cls(*args, **kwargs) for month in range(1, 13)}

        super().__init__("month", monthly_models)


class WeeklyMultiplexer(PytorchMultiplexer):
    def __init__(self, cls, *args, **kwargs):
        monthdays = [f"{m:02}{d:02}" for m, d in ECMWF_FORECASTS]
        weekly_models = {monthday: cls(*args, **kwargs) for monthday in monthdays}

        super().__init__("monthday", weekly_models)


class MonthlyLinearModel(MonthlyMultiplexer):
    def __init__(self, *args, **kwargs):
        super().__init__(LinearModel, *args, **kwargs)


class WeeklyLinearModel(WeeklyMultiplexer):
    def __init__(self, *args, **kwargs):
        super().__init__(LinearModel, *args, **kwargs)


class MultiplexedNormalEMOSModel(nn.Module):
    """EMOS model that supports multiplexed linear models."""

    def __init__(
        self,
        loc_key,
        scale_key,
        linear_model_cls,
        key,
        biweekly=False,
        regularization=1e-9,
    ):
        super().__init__()

        self.loc_key = loc_key
        self.scale_key = scale_key

        shape = (2, 121, 240) if biweekly else (121, 240)

        self.loc_model = linear_model_cls(*shape)
        self.scale_model = linear_model_cls(*shape, fill_weights=1.2)

        self.regularization = regularization

        self.key = key

    def forward(self, batch):
        forecast_loc, forecast_scale = batch[self.loc_key], batch[self.scale_key]
        key = batch[self.key]

        loc = self.loc_model(key, forecast_loc)
        scale = self.scale_model(key, forecast_scale)
        scale = torch.clip(scale, min=self.regularization)

        return torch.distributions.Normal(loc=loc, scale=scale)


class MonthlyNormalEMOSModel(MultiplexedNormalEMOSModel):
    def __init__(self, loc_key, scale_key, biweekly=False, regularization=1e-9):
        super().__init__(
            loc_key,
            scale_key,
            MonthlyLinearModel,
            "month",
            biweekly=biweekly,
            regularization=regularization,
        )


class WeeklyNormalEMOSModel(MultiplexedNormalEMOSModel):
    def __init__(self, loc_key, scale_key, biweekly=False, regularization=1e-9):
        super().__init__(
            loc_key,
            scale_key,
            WeeklyLinearModel,
            "monthday",
            biweekly=biweekly,
            regularization=regularization,
        )
