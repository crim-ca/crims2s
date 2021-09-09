"""Models that use deep learning to perform a bayesian update on the prior and thus make
a better forecast."""

import numpy as np
import torch
import torch.nn as nn

from .emos import NormalCubeNormalMultiplexedEMOS
from .util import DistributionToTerciles


class WeightModel(nn.Module):
    """Output a weight that tells how informative the input model is w/r to a uniform prior."""

    def __init__(self):
        super().__init__()

    def forward(self, example):
        return


class BayesianUpdateModel(nn.Module):
    def __init__(self, forecast_model):
        super().__init__()
        self.forecast_model = forecast_model
        self.weight_model = WeightModel()
        self.t2m_to_terciles = DistributionToTerciles()
        self.tp_to_terciles = DistributionToTerciles()

    def forward(self, example):
        t2m_dist, tp_dist = self.forecast_model(example)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, example["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, example["edges_tp"])

        raw_update_weights = self.weight_model(example["features"])
        raw_prior_weights = torch.full_like(raw_update_weights, 0.5)

        raw_weights = torch.stack([raw_prior_weights, raw_update_weights])
        weights = torch.softmax(raw_weights, dim=0)

        t2m_prior = torch.full_like(t2m_terciles, 1.0 / 3.0)
        tp_prior = torch.full_like(tp_terciles, 1.0 / 3.0)

        t2m_estimates = torch.stack([t2m_prior, t2m_terciles])
        tp_estimates = torch.stack([tp_prior, tp_terciles])

        t2m = (t2m_estimates * weights).mean(dim=0)
        tp = (tp_estimates * weights).mean(dim=0)

        return t2m, tp