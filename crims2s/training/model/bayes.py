"""Models that use deep learning to perform a bayesian update on the prior and thus make
a better forecast."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bias import ModelParameterBiasCorrection
from .util import DistributionToTerciles


class WeightModel(nn.Module):
    """Output a weight that tells how informative the input model is w/r to a uniform prior."""

    def __init__(self):
        super().__init__()

    def forward(self, example):
        return torch.full((2, 121, 240), 100.0)


class TrivialWeightModel(nn.Module):
    def __init__(self, initial_w=1.0, device="cuda"):
        super().__init__()
        self.w = nn.parameter.Parameter(torch.tensor(initial_w))
        self.device = device

    def forward(self, example):
        return self.w * torch.ones(2, 121, 240, device=self.device)


class TileWeightModel(nn.Module):
    def __init__(self, initial_w=1.0):
        super().__init__()
        self.w = nn.parameter.Parameter(torch.full((2, 121, 240), initial_w))

    def forward(self, example):
        return self.w


class LinearWeightModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.lin = nn.parameter.Parameter(torch.rand(2, 121, 240, 14))
        self.bias = nn.parameter.Parameter(torch.rand(2, 121, 240))

    def forward(self, example):
        example = example.mean(dim=-2).mean(dim=1)
        remapped = (self.lin * example).sum(dim=-1) + self.bias

        return remapped


class ConvolutionalWeightModel(nn.Module):
    def __init__(
        self, in_features, kernel_size=(5, 5, 2), padding=(2, 2, 0), embedding_size=128
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_features,
            embedding_size,
            kernel_size=kernel_size,
            padding_mode="circular",
            padding=padding,
        )

        self.conv2 = nn.Conv3d(
            embedding_size,
            embedding_size,
            kernel_size=kernel_size,
            padding_mode="circular",
            padding=padding,
        )

        self.conv3 = nn.Conv3d(
            embedding_size,
            embedding_size,
            kernel_size=kernel_size,
            padding_mode="circular",
            padding=padding,
        )

        self.lin = nn.Linear(embedding_size, 1)

    def forward(self, example):
        # Dims of example: week, lead time, lat, lon, realization, dim.
        print("example", example.shape)

        x = example[..., 0, :]  # Grab the first member.

        if len(x.shape) == 4:
            # If there are no batches, simulate it by adding a batch dim.
            x = x.unsqueeze(dim=0)

        x = torch.transpose(x, 1, -1)  # Swap dims and depth.

        print("x", x.shape)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=-1)

        x = torch.transpose(
            torch.transpose(x, 1, -1), 1, 2
        )  # Bring the channels back at the end.
        x = self.lin(x)

        return x.squeeze()


class BayesianUpdateModel(nn.Module):
    def __init__(
        self,
        forecast_model,
        t2m_weight_model,
        tp_weight_model,
        weight_model_factor=1.0,
        bias_correction=False,
    ):
        super().__init__()
        self.forecast_model = forecast_model
        self.t2m_weight_model = t2m_weight_model
        self.tp_weight_model = tp_weight_model
        self.t2m_to_terciles = DistributionToTerciles()
        self.tp_to_terciles = DistributionToTerciles()
        self.weight_model_factor = weight_model_factor

        if bias_correction:
            self.bias_correction_model = ModelParameterBiasCorrection()
        else:
            self.bias_correction_model = None

    def make_weights(self, weight_model, features, nan_mask):
        weights_from_model = self.weight_model_factor * weight_model(features)

        raw_update_weights = torch.where(
            nan_mask, torch.zeros_like(weights_from_model), weights_from_model,
        )

        raw_prior_weights = torch.full_like(raw_update_weights, 0.0)

        raw_weights = torch.stack([raw_prior_weights, raw_update_weights])
        weights = torch.softmax(raw_weights, dim=0)

        return weights

    def forward(self, example):
        if self.bias_correction_model is not None:
            example = self.bias_correction_model(example)

        t2m_dist, tp_dist = self.forecast_model(example)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, example["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, example["edges_tp"])

        tp_nan_mask = tp_terciles.isnan().any(dim=0)
        tp_terciles[:, tp_nan_mask] = 0.0

        t2m_nan_mask = t2m_terciles.isnan().any(dim=0)
        t2m_terciles[:, t2m_nan_mask] = 0.0

        # We use the key features features because of our conversion convention from
        # xarray to pytorch. The first features designates the features dataset. The
        # second features designates the features array inside the dataset.
        t2m_weights = self.make_weights(
            self.t2m_weight_model, example["features_features"], t2m_nan_mask
        )
        tp_weights = self.make_weights(
            self.tp_weight_model, example["features_features"], tp_nan_mask
        )

        prior = torch.full_like(t2m_terciles, 1.0 / 3.0)

        t2m_estimates = torch.stack([prior, t2m_terciles])
        tp_estimates = torch.stack([prior, tp_terciles])

        # Meaning of the keys: Ditribution (prior or forecast), Category, Lead time, lAtitude, lOngitude
        t2m = torch.einsum("dclao,dlao->clao", t2m_estimates, t2m_weights)
        tp = torch.einsum("dclao,dlao->clao", tp_estimates, tp_weights)

        t2m[:, t2m_nan_mask] = np.nan
        tp[:, tp_nan_mask] = np.nan

        t2m_prior_weights = torch.where(
            ~t2m_nan_mask, t2m_weights[0], torch.zeros_like(t2m_weights[0])
        )
        tp_prior_weights = torch.where(
            ~tp_nan_mask, tp_weights[0], torch.zeros_like(tp_weights[0])
        )

        return t2m, tp, t2m_prior_weights[~t2m_nan_mask], tp_prior_weights[~tp_nan_mask]
