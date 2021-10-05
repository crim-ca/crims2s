"""Models that use deep learning to perform a bayesian update on the prior and thus make
a better forecast."""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bias import ModelParameterBiasCorrection
from .emos import NormalCubeNormalEMOS, RollingWindowEMOS
from .util import DistributionModelAdapter, DistributionToTerciles


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
    def __init__(self, n_models=4):
        super().__init__()
        self.t2m_weights = nn.parameter.Parameter(torch.ones((n_models, 2, 121, 240)))
        self.tp_weights = nn.parameter.Parameter(torch.ones((n_models, 2, 121, 240)))

    def forward(self, batch_size):
        # The same weights are shared across the batch, but we fake a batch dimension
        # nontheless so that the output of this model matches the output of other models.
        t2m_weights = torch.unsqueeze(self.t2m_weights, dim=0).repeat(
            (batch_size, 1, 1, 1, 1)
        )
        tp_weights = torch.unsqueeze(self.tp_weights, dim=0).repeat(
            (batch_size, 1, 1, 1, 1)
        )

        return t2m_weights, tp_weights


class LinearWeightModel(nn.Module):
    def __init__(self, in_features, n_models, moments=True):
        super().__init__()

        self.moments = moments

        in_features = 2 * in_features if moments else in_features

        self.t2m_weights = nn.parameter.Parameter(
            torch.rand(n_models, 2, 121, 240, in_features)
        )
        self.t2m_bias = nn.parameter.Parameter(torch.rand(n_models, 2, 121, 240))
        self.tp_weights = nn.parameter.Parameter(
            torch.rand(n_models, 2, 121, 240, in_features)
        )
        self.tp_bias = nn.parameter.Parameter(torch.rand(n_models, 2, 121, 240))

    def forward(self, features):
        x = features.mean(dim=1)  # Collapse the time dimension.

        # Compute mean and std of features across members.
        if self.moments:
            # Use average and STD of members as features.
            x = torch.cat([x.mean(dim=-2), x.std(dim=-2)], dim=-1)
        else:
            # Select first member.
            x = features[..., 0, :]

        # Dimension keys: Batch, Time, Leadtime, lAtitude, lOngitude, Channel, Model
        t2m = torch.einsum("baoc,mtaoc->bmtao", x, self.t2m_weights) + self.t2m_bias
        tp = torch.einsum("baoc,mtaoc->bmtao", x, self.tp_weights) + self.tp_bias

        return t2m, tp


class OptionnalModelWrapper(DistributionModelAdapter):
    """Wrap a model for which we don't have a full dataset. The examples contain a key
    named `available_key` which tells us if the example is available or not. For 
    instance, if inside the example `example['eccc_available'] == False` then we 
    know that ECCC is not available for that example and we have to react accordingly."""

    def __init__(self, model, parameters_prefix, available_key):
        self.parameters_prefix = parameters_prefix
        self.available_key = available_key
        super().__init__(model)

    def forward(self, batch):
        available = batch[self.available_key]

        if available.all():
            return super().forward(batch)

        # Assume fully undefined forcast. If forecast is available, replace nans with
        # forecast.
        t2m = torch.full_like(batch["terciles_t2m"], np.nan)
        tp = torch.full_like(batch["terciles_tp"], np.nan)

        if not available.any():
            return t2m, tp

        # The mixed case is the trickyest.

        # Make a new batch where there only is eccc available examples.
        available_batch = {}
        for k in batch.keys():
            if k.startswith(self.parameters_prefix) or k.startswith("edges_"):
                available_batch[k] = batch[k][available]

            if k in ["month", "monthday", "year"]:
                available_batch[k] = np.array(batch[k])[available.cpu()]

        t2m_eccc, tp_eccc = super().forward(available_batch)

        t2m[available] = t2m_eccc
        tp[available] = tp_eccc

        return t2m, tp


class ECMWFModelWrapper(DistributionModelAdapter):
    def __init__(self):
        model = NormalCubeNormalEMOS(biweekly=True)
        super().__init__(model)


class RollingECMWFWrapper(DistributionModelAdapter):
    def __init__(self, window_size=20):
        model = RollingWindowEMOS(window_size=window_size)
        super().__init__(model)


class ECCCModelWrapper(OptionnalModelWrapper):
    def __init__(self):
        parameters_prefix = "eccc_parameters"
        model = NormalCubeNormalEMOS(biweekly=True, prefix=parameters_prefix)
        super().__init__(model, parameters_prefix, "eccc_available")


class RollingECCCWrapper(OptionnalModelWrapper):
    """Wrap a rolling window ECCC EMOS model such that it is compatible with our 
    dataset dictinnary."""

    def __init__(self, window_size=20):
        parameters_prefix = "eccc_parameters"
        model = RollingWindowEMOS(window_size, prefix=parameters_prefix)
        super().__init__(model, parameters_prefix, "eccc_available")


class NCEPModelWrapper(OptionnalModelWrapper):
    def __init__(self):
        parameters_prefix = "ncep_parameters"
        model = NormalCubeNormalEMOS(biweekly=True, prefix=parameters_prefix)
        super().__init__(model, parameters_prefix, "ncep_available")


class RollingNCEPWrapper(OptionnalModelWrapper):
    def __init__(self, window_size=20):
        parameters_prefix = "ncep_parameters"
        model = RollingWindowEMOS(window_size, prefix=parameters_prefix)
        super().__init__(model, parameters_prefix, "ncep_available")


class ClimatologyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        t2m = torch.full_like(batch["terciles_t2m"], 1.0 / 3.0)
        tp = torch.full_like(batch["terciles_tp"], 1.0 / 3.0)
        return t2m, tp


class MultiCenterBayesianUpdateModel(nn.Module):
    def __init__(
        self, forecast_models, weight_model: nn.Module, bias_correction=False,
    ):
        super().__init__()
        self.forecast_models = nn.ModuleList(forecast_models)
        self.weight_model = weight_model

        if bias_correction:
            self.bias_correction_model = ModelParameterBiasCorrection()
        else:
            self.bias_correction_model = None

    def make_weights(self, weight_model_output, nan_mask):
        weights = weight_model_output

        batch_size = weight_model_output.shape[0]
        n_forecasts = len(self.forecast_models)
        n_lead_time = 2
        weights = weight_model_output.reshape(
            batch_size, n_forecasts, n_lead_time, *weight_model_output.shape[-2:]
        )

        # Don't give any weight to a forecast that had a nan.
        weights = torch.where(
            ~nan_mask,
            weights,
            torch.tensor(-1e12, dtype=weights.dtype, device=weights.device),
        )
        weights = torch.softmax(weights, dim=1)

        return weights

    def make_forecasts(self, example):
        t2m_forecasts, tp_forecasts = [], []
        for m in self.forecast_models:
            t2m_forecast, tp_forecast = m(example)

            t2m_forecasts.append(t2m_forecast)
            tp_forecasts.append(tp_forecast)

        t2m_forecasts = torch.stack(t2m_forecasts, dim=1)
        tp_forecasts = torch.stack(tp_forecasts, dim=1)

        return t2m_forecasts, tp_forecasts

    def forward(self, example):
        if self.bias_correction_model is not None:
            example = self.bias_correction_model(example)

        t2m_forecasts, tp_forecasts = self.make_forecasts(example)

        # At this point the dims are: batch, model, category, lead_time, lat, lon.

        t2m_nan_mask = t2m_forecasts.isnan().any(dim=2)
        t2m_forecasts = torch.nan_to_num(t2m_forecasts, 0.0)

        tp_nan_mask = tp_forecasts.isnan().any(dim=2)
        tp_forecasts = torch.nan_to_num(tp_forecasts, 0.0)

        batch_size = len(example["terciles_t2m"])
        features_or_batch_size = example.get("features_features", batch_size)
        t2m_forecast_weight, tp_forecast_weight = self.weight_model(
            features_or_batch_size
        )

        t2m_weights = self.make_weights(t2m_forecast_weight, t2m_nan_mask)
        tp_weights = self.make_weights(tp_forecast_weight, tp_nan_mask)

        # Meaning of the keys: Batch, Model (prior or ecmwf or eccc), Category, Lead time, lAtitude, lOngitude
        t2m = torch.einsum("bmclao,bmlao->bclao", t2m_forecasts, t2m_weights)
        tp = torch.einsum("bmclao,bmlao->bclao", tp_forecasts, tp_weights)

        t2m_nan_mask = example["terciles_t2m"].isnan().any(dim=1)
        tp_nan_mask = example["terciles_tp"].isnan().any(dim=1)

        # Model index 0 is the climatology model. It's the one we regularize again.
        return (
            t2m,
            tp,
            torch.transpose(t2m_weights, 0, 1)[:, ~t2m_nan_mask],
            torch.transpose(tp_weights, 0, 1)[:, ~tp_nan_mask],
        )

    def forecast_models_parameters(self):
        return self.forecast_models.parameters()

    def weight_model_parameters(self):
        return self.weight_model.parameters()


class Projection(nn.Module):
    def __init__(self, in_features, out_features, moments=False, width=7, depth=2):
        super().__init__()

        self.conv = nn.Conv3d(
            in_features,
            out_features,
            kernel_size=(width, width, depth),
            padding=(width // 2, width // 2, 0),
            padding_mode="circular",
            bias=False,
        )

        self.bn = nn.BatchNorm3d(out_features)
        self.act = nn.LeakyReLU()

        self.moments = moments

    def forward(self, features):
        # Dims of example: batch, week, lead time, lat, lon, realization, dim.
        x = features

        # Compute mean and std of features across members.
        if self.moments:
            # Use average and STD of members as features.
            x = torch.cat([features.mean(dim=-2), features.std(dim=-2)], dim=-1)
        else:
            # Select first member.
            x = features[..., 0, :]

        return self.act(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, n_features, width, depth, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(
            n_features,
            n_features,
            kernel_size=(width, width, depth),
            padding=(width // 2, width // 2, depth // 2),
            padding_mode="circular",
            bias=False,
        )
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm3d(n_features)

        self.conv2 = nn.Conv3d(
            n_features,
            n_features,
            kernel_size=(width, width, depth),
            padding=(width // 2, width // 2, depth // 2),
            padding_mode="circular",
            bias=False,
        )
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm3d(n_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_in = x
        x = self.act1(self.bn1(self.conv1(x)))

        if self.drop.p > 0.0:
            x = self.act2(self.drop(x_in + self.conv2(x)))
        else:
            x = self.act2(self.bn2(x_in + self.conv2(x)))

        return x


class CommonTrunk(nn.Module):
    def __init__(self, embedding_size, n_blocks=3, width=3, depth=3, dropout=0.0):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvBlock(embedding_size, width, depth, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x


class VariableBranch(nn.Module):
    def __init__(
        self, embedding_size, n_blocks=1, width=3, depth=1, dropout=0.0, out_features=2
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvBlock(embedding_size, width, depth, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )

        self.end_layer = nn.Conv2d(embedding_size, out_features, kernel_size=(1, 1))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        x = x.mean(dim=-1)  # Flatten the depth (time) dimension.

        return self.end_layer(x)


class GlobalBranchBlock(nn.Module):
    def __init__(self, in_features, out_features, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_features,
            out_features,
            kernel_size=3,
            stride=2,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_features)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(
            out_features, out_features, kernel_size=3, bias=False, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_features)
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))

        return self.act2(self.bn2(self.conv2(x)))


class GlobalBranch(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        size1 = max(out_features // 4, in_features)
        size2 = max(out_features // 2, in_features)

        self.block1 = GlobalBranchBlock(in_features, size1)
        self.block2 = GlobalBranchBlock(size1, size2)
        self.block3 = GlobalBranchBlock(size2, out_features)

    def forward(self, x):
        x = x.mean(dim=-1)  # Flatten time dimension.

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.mean(dim=[-1, -2])  # Global average pooling to get a single vector.

        return x


class BiheadedWeightModel(nn.Module):
    def __init__(
        self,
        in_features,
        embedding_size,
        global_branch=True,
        dropout=0.0,
        out_features=2,
        moments=False,
        variable_branch_blocks=1,
    ):
        super().__init__()

        self.global_branch = global_branch

        self.projection = Projection(in_features, embedding_size, moments=moments)
        self.common_trunk = CommonTrunk(embedding_size, dropout=dropout)
        self.t2m_branch = VariableBranch(
            embedding_size,
            dropout=dropout,
            out_features=out_features,
            n_blocks=variable_branch_blocks,
        )
        self.tp_branch = VariableBranch(
            embedding_size,
            dropout=dropout,
            out_features=out_features,
            n_blocks=variable_branch_blocks,
        )

        if self.global_branch:
            self.global_branch = GlobalBranch(embedding_size, embedding_size)

    def forward(self, x):
        x = torch.transpose(x, 1, -1)  # Swap dims and depth.

        x = self.projection(x)
        x = self.common_trunk(x)

        if self.global_branch:
            global_features = self.global_branch(x)
            global_features = (
                global_features.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            )

            x_t2m = self.t2m_branch(x + global_features)
            x_tp = self.tp_branch(x + global_features)
        else:
            x_t2m = self.t2m_branch(x)
            x_tp = self.tp_branch(x)

        return x_t2m, x_tp
