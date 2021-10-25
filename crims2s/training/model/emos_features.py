"""EMOS Models that use the feature matrix instead of the predictand directly."""

import torch
import torch.distributions
import torch.nn as nn

from ...transform import FIELD_STD
from .util import DistributionModelAdapter, WeeklyRollingWindowMultiplexer


class LinearModelFromFeatures(nn.Module):
    def __init__(self, in_features, param_scale=1.0, init_std=0.1):
        super().__init__()

        self.param_scale = param_scale

        self.weights = nn.Parameter(torch.ones((2, 121, 240)))
        self.intercept = nn.Parameter(torch.zeros((2, 121, 240)))

        self.adjustment_weights = nn.Parameter(torch.zeros(2, 121, 240, in_features))
        self.adjustment_bias = nn.Parameter(torch.zeros(2, 121, 240, in_features))

        # self.multiplier_weights = nn.parameter.Parameter(
        #     torch.normal(mean=torch.zeros(2, 121, 240, in_features), std=init_std)
        # )
        # self.multiplier_bias = nn.parameter.Parameter(torch.ones(2, 121, 240))

        # self.intercept_weights = nn.parameter.Parameter(
        #     torch.normal(mean=torch.zeros(2, 121, 240, in_features), std=init_std)
        # )
        # self.intercept_bias = nn.parameter.Parameter(torch.zeros(2, 121, 240))

    def forward(self, predictor, features):

        # Keys: Batch, Lead time, lAtitude, lOngitude, Channel
        adjustment = torch.einsum(
            "blaoc,laoc->blao",
            features + self.adjustment_bias,
            self.adjustment_weights,
        )

        return self.weights * predictor + self.intercept + adjustment


class RollingWindowFeaturesLinearModel(WeeklyRollingWindowMultiplexer):
    def __init__(self, window_size, *args, **kwargs):
        super().__init__(window_size, LinearModelFromFeatures, *args, **kwargs)

    def forward(self, key, predictor, features):
        return super().forward(key, predictor, features)


class EMOSFeatures(nn.Module):
    def __init__(self, in_features, window_size=1, regularization=1e-9, **kwargs):
        super().__init__()
        self.regularization = regularization

        self.t2m_mu_model = RollingWindowFeaturesLinearModel(
            window_size, in_features, param_scale=FIELD_STD["t2m"], **kwargs
        )
        self.t2m_sigma_model = RollingWindowFeaturesLinearModel(
            window_size, in_features, param_scale=FIELD_STD["t2m"], **kwargs
        )
        self.tp_mu_model = RollingWindowFeaturesLinearModel(
            window_size,
            in_features,
            param_scale=FIELD_STD["tp"] ** (1.0 / 3.0),
            **kwargs
        )
        self.tp_sigma_model = RollingWindowFeaturesLinearModel(
            window_size,
            in_features,
            param_scale=FIELD_STD["tp"] ** (1.0 / 3.0),
            **kwargs
        )

    def __call__(self, batch):
        x = batch["features_features"]
        x = x.mean(dim=[2, -2])  # Average over step whithin forecast, and member.

        key = batch["monthday"]

        t2m_mu = self.t2m_mu_model(key, batch["model_parameters_t2m_mu"], x)
        t2m_sigma = self.t2m_sigma_model(key, batch["model_parameters_t2m_sigma"], x)
        t2m_sigma = torch.clip(t2m_sigma, min=self.regularization)
        t2m_normal = torch.distributions.Normal(loc=t2m_mu, scale=t2m_sigma,)

        tp_mu = self.tp_mu_model(key, batch["model_parameters_tp_cube_root_mu"], x)
        tp_sigma = self.tp_sigma_model(
            key, batch["model_parameters_tp_cube_root_sigma"], x
        )
        tp_sigma = torch.clip(tp_sigma, min=self.regularization)
        tp_normal = torch.distributions.Normal(loc=tp_mu, scale=tp_sigma,)

        return t2m_normal, tp_normal


class TercilesEMOSFeatures(DistributionModelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(EMOSFeatures(*args, **kwargs), tp_regularization=1e-9)
