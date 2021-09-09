import torch
import torch.nn as nn
import numpy as np


def compute_edges_cdf_from_distribution(distribution, edges, regularization=0.0):
    edges_nan_mask = edges.isnan()
    edges[edges_nan_mask] = 0.0

    cdf = distribution.cdf(edges + regularization)
    edges[edges_nan_mask] = np.nan
    cdf[edges_nan_mask] = np.nan

    return cdf


def edges_cdf_to_terciles(edges_cdf):
    return torch.stack(
        [edges_cdf[0], edges_cdf[1] - edges_cdf[0], 1.0 - edges_cdf[1],], dim=0
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

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.t2m_to_terciles = DistributionToTerciles()
        self.tp_to_terciles = DistributionToTerciles()

    def forward(self, example):
        t2m_dist, tp_dist = self.model(example)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, example["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, example["edges_tp"])

        return t2m_terciles, tp_terciles
