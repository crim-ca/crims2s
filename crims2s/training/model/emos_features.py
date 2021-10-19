"""EMOS Models that use the feature matrix instead of the predictand directly."""

import torch
import torch.distributions
import torch.nn as nn

from .util import DistributionModelAdapter


class LinearModelFromFeatures(nn.Module):
    def __init__(self, in_features, param_scale=1e-6):
        super().__init__()

        self.param_scale = param_scale

        self.multiplier_weights = nn.parameter.Parameter(
            torch.normal(mean=torch.zeros(2, 121, 240, in_features), std=param_scale)
        )
        self.multiplier_bias = nn.parameter.Parameter(torch.ones(2, 121, 240))

        self.intercept_weights = nn.parameter.Parameter(
            torch.normal(mean=torch.zeros(2, 121, 240, in_features), std=param_scale)
        )
        self.intercept_bias = nn.parameter.Parameter(torch.zeros(2, 121, 240))

    def forward(self, predictor, features):
        # Keys: Batch, Lead time, lAtitude, lOngitude, Channel
        multiplier = (
            torch.einsum(
                "blaoc,laoc->blao", features, self.param_scale * self.multiplier_weights
            )
            + self.multiplier_bias
        )
        intercept = (
            torch.einsum(
                "blaoc,laoc->blao", features, self.param_scale * self.intercept_weights
            )
            + self.intercept_bias
        )

        return multiplier * predictor + intercept


class EMOSFeatures(nn.Module):
    def __init__(self, in_features, regularization=1e-9):
        super().__init__()
        self.regularization = regularization

        self.t2m_mu_model = LinearModelFromFeatures(in_features)
        self.t2m_sigma_model = LinearModelFromFeatures(in_features)
        self.tp_mu_model = LinearModelFromFeatures(in_features)
        self.tp_sigma_model = LinearModelFromFeatures(in_features)

    def __call__(self, batch):
        x = batch["features_features"]
        x = x.mean(dim=[2, -2])  # Average over step whithin forecast, and member.

        t2m_mu = self.t2m_mu_model(batch["model_parameters_t2m_mu"], x)
        t2m_sigma = self.t2m_sigma_model(batch["model_parameters_t2m_sigma"], x)
        t2m_sigma = torch.clip(t2m_sigma, min=self.regularization)
        t2m_normal = torch.distributions.Normal(loc=t2m_mu, scale=t2m_sigma,)

        tp_mu = self.tp_mu_model(batch["model_parameters_tp_cube_root_mu"], x)
        tp_sigma = self.tp_sigma_model(batch["model_parameters_tp_cube_root_sigma"], x)
        tp_sigma = torch.clip(tp_sigma, min=self.regularization)
        tp_normal = torch.distributions.Normal(loc=tp_mu, scale=tp_sigma,)

        return t2m_normal, tp_normal


class TercilesEMOSFeatures(DistributionModelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(EMOSFeatures(*args, **kwargs), tp_regularization=1e-9)
