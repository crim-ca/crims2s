"""Models that use deep learning to perform a bayesian update on the prior and thus make
a better forecast."""

from crims2s.training.model.emos import NormalCubeNormalEMOS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bias import ModelParameterBiasCorrection
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
        self,
        in_features,
        kernel_size=(5, 5, 2),
        padding=(2, 2, 0),
        embedding_size=128,
        moments=False,
    ):
        super().__init__()

        self.moments = moments

        in_features = 2 * in_features if moments else in_features

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

        self.lin = nn.Linear(embedding_size, 2)

    def forward(self, example):
        # Dims of example: batch, week, lead time, lat, lon, realization, dim.

        # Compute mean and std of features across members.
        if self.moments:
            # Use average and STD of members as features.
            x = torch.cat([example.mean(dim=-2), example.std(dim=-2)], dim=-1)
        else:
            # Select first member.
            x = example[..., 0, :]

        if len(x.shape) == 4:
            # If there are no batches, simulate it by adding a batch dim.
            x = x.unsqueeze(dim=0)

        x = torch.transpose(x, 1, -1)  # Swap dims and depth.

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Mean over the depth (time) dimension.
        x = x.mean(dim=-1)

        x = torch.transpose(
            torch.transpose(x, 1, -1), 1, 2
        )  # Bring the channels back at the end.
        x = self.lin(x)

        x = torch.transpose(torch.transpose(x, 2, 3), 1, 2)

        return x.squeeze()


class SimpleConvWeightModel(nn.Module):
    def __init__(self, in_features, kernel_size, padding, embedding_size, moments):
        super().__init__()

        self.t2m_model = ConvolutionalWeightModel(
            in_features, kernel_size, padding, embedding_size, moments
        )

        self.tp_model = ConvolutionalWeightModel(
            in_features, kernel_size, padding, embedding_size, moments
        )

    def forward(self, x):
        x_t2m = self.t2m_model(x)
        x_tp = self.tp_model(x)

        return x_t2m, x_tp


class BayesianUpdateModel(nn.Module):
    def __init__(
        self,
        forecast_model,
        weight_model,
        weight_model_factor=1.0,
        bias_correction=False,
    ):
        super().__init__()
        self.forecast_model = forecast_model
        self.weight_model = weight_model
        self.t2m_to_terciles = DistributionToTerciles()
        self.tp_to_terciles = DistributionToTerciles()
        self.weight_model_factor = weight_model_factor

        if bias_correction:
            self.bias_correction_model = ModelParameterBiasCorrection()
        else:
            self.bias_correction_model = None

    def make_weights(self, weight_model_output, nan_mask):
        weights_from_model = self.weight_model_factor * weight_model_output

        raw_update_weights = torch.where(
            nan_mask, torch.zeros_like(weights_from_model), weights_from_model,
        )

        raw_prior_weights = torch.full_like(raw_update_weights, 0.0)

        raw_weights = torch.stack([raw_prior_weights, raw_update_weights], dim=1)
        weights = torch.softmax(raw_weights, dim=1)

        return weights

    def forward(self, example):
        if self.bias_correction_model is not None:
            example = self.bias_correction_model(example)

        t2m_dist, tp_dist = self.forecast_model(example)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, example["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, example["edges_tp"])

        t2m_nan_mask = t2m_terciles.isnan().any(dim=1)
        torch.transpose(t2m_terciles, 0, 1)[:, t2m_nan_mask] = 0.0

        tp_nan_mask = tp_terciles.isnan().any(dim=1)
        torch.transpose(tp_terciles, 0, 1)[:, tp_nan_mask] = 0.0

        # We use the key features features because of our conversion convention from
        # xarray to pytorch. The first features designates the features dataset. The
        # second features designates the features array inside the dataset.
        t2m_forecast_weight, tp_forecast_weight = self.weight_model(
            example["features_features"]
        )

        t2m_weights = self.make_weights(t2m_forecast_weight, t2m_nan_mask)
        tp_weights = self.make_weights(tp_forecast_weight, tp_nan_mask)

        prior = torch.full_like(t2m_terciles, 1.0 / 3.0)

        t2m_estimates = torch.stack([prior, t2m_terciles], dim=1)
        tp_estimates = torch.stack([prior, tp_terciles], dim=1)

        # Meaning of the keys: Batch, Ditribution (prior or forecast), Category, Lead time, lAtitude, lOngitude
        t2m = torch.einsum("bdclao,bdlao->bclao", t2m_estimates, t2m_weights)
        tp = torch.einsum("bdclao,bdlao->bclao", tp_estimates, tp_weights)

        torch.transpose(t2m, 0, 1)[:, t2m_nan_mask] = np.nan
        torch.transpose(tp, 0, 1)[:, tp_nan_mask] = np.nan

        # t2m_prior_weights = torch.where(
        #     ~t2m_nan_mask, t2m_weights[1], torch.zeros_like(t2m_weights[1])
        # )
        # tp_prior_weights = torch.where(
        #     ~tp_nan_mask, tp_weights[1], torch.zeros_like(tp_weights[1])
        # )

        return (
            t2m,
            tp,
            t2m_weights[:, 0][~t2m_nan_mask],
            tp_weights[:, 0][~tp_nan_mask],
            # t2m_terciles,
            # tp_terciles,
        )


class ECMWFModelWrapper(DistributionModelAdapter):
    def __init__(self):
        model = NormalCubeNormalEMOS(biweekly=True)
        super().__init__(model)


class ECCCModelWrapper(DistributionModelAdapter):
    def __init__(self):
        model = NormalCubeNormalEMOS(biweekly=True, prefix="eccc_parameters")
        super().__init__(model)

    def forward(self, batch):
        eccc_available = batch["eccc_available"]
        print(eccc_available)

        if eccc_available.all():
            return super().forward(batch)

        # Assume uniform distribution. If eccc forecast is available, replace
        # uniform prior with the eccc forecast.
        t2m = torch.full_like(batch["terciles_t2m"], 1.0 / 3.0)
        tp = torch.full_like(batch["terciles_tp"], 1.0 / 3.0)

        if not eccc_available.any():
            return t2m, tp


        # The mixed case is the trickyest.

        # Make a new batch where there only is eccc available examples.
        eccc_available_batch = {}
        for k in batch.keys():
            if k.startswith("eccc_parameters") or k.startswith("edges_"):
                eccc_available_batch[k] = batch[k][eccc_available]

            if k in ["month", "monthday", "year"]:
                eccc_available_batch[k] = np.array(batch[k])[eccc_available.cpu()]

        print(eccc_available_batch["month"])

        t2m_eccc, tp_eccc = super().forward(eccc_available_batch)

        t2m[eccc_available] = t2m_eccc
        tp[eccc_available] = tp_eccc

        return t2m, tp


class ClimatologyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        t2m = torch.full_like(batch["terciles_t2m"], 1.0 / 3.0)
        tp = torch.full_like(batch["terciles_tp"], 1.0 / 3.0)
        return t2m, tp


class MultiCenterBayesianUpdateModel(nn.Module):
    def __init__(
        self, forecast_models, weight_model, bias_correction=False,
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
        print("make forecasts")
        t2m_forecasts, tp_forecasts = [], []
        for m in self.forecast_models:
            t2m_forecast, tp_forecast = m(example)

            t2m_forecasts.append(t2m_forecast)
            tp_forecasts.append(tp_forecast)

            print("t2m nans", t2m_forecast.isnan().float().mean())
            print("tp nans", tp_forecast.isnan().float().mean())

        t2m_forecasts = torch.stack(t2m_forecasts, dim=1)
        tp_forecasts = torch.stack(tp_forecasts, dim=1)

        return t2m_forecasts, tp_forecasts

    def forward(self, example):
        if self.bias_correction_model is not None:
            example = self.bias_correction_model(example)

        t2m_forecasts, tp_forecasts = self.make_forecasts(example)

        # At this point the dims are: batch, model, category, lead_time, lat, lon.

        print("t2m estimates", t2m_forecasts.shape)

        t2m_nan_mask = t2m_forecasts.isnan().any(dim=2)
        t2m_forecasts = torch.nan_to_num(t2m_forecasts, 0.0)

        tp_nan_mask = tp_forecasts.isnan().any(dim=2)
        tp_forecasts = torch.nan_to_num(tp_forecasts, 0.0)

        t2m_forecast_weight, tp_forecast_weight = self.weight_model(
            example["features_features"]
        )

        t2m_weights = self.make_weights(t2m_forecast_weight, t2m_nan_mask)
        tp_weights = self.make_weights(tp_forecast_weight, tp_nan_mask)

        # Meaning of the keys: Batch, Model (prior or ecmwf or eccc), Category, Lead time, lAtitude, lOngitude
        t2m = torch.einsum("bmclao,bmlao->bclao", t2m_forecasts, t2m_weights)
        tp = torch.einsum("bmclao,bmlao->bclao", tp_forecasts, tp_weights)

        # torch.transpose(t2m, 0, 1)[:, t2m_nan_mask] = np.nan
        # torch.transpose(tp, 0, 1)[:, tp_nan_mask] = np.nan

        # t2m_prior_weights = torch.where(
        #     ~t2m_nan_mask, t2m_weights[1], torch.zeros_like(t2m_weights[1])
        # )
        # tp_prior_weights = torch.where(
        #     ~tp_nan_mask, tp_weights[1], torch.zeros_like(tp_weights[1])
        # )

        t2m_nan_mask = example["terciles_t2m"].isnan().any(dim=1)
        tp_nan_mask = example["terciles_tp"].isnan().any(dim=1)

        print("t2m weights", t2m_weights.shape)
        print("t2m_nan_mask", t2m_nan_mask.shape)

        print("t2m_nan rate where prior", t2m_nan_mask[:, 0].float().mean())

        # Model index 0 is the climatology model. It's the one we regularize again.
        return (
            t2m,
            tp,
            t2m_weights[:, 0][~t2m_nan_mask],
            tp_weights[:, 0][~tp_nan_mask],
        )


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
        self, embedding_size, n_blocks=1, width=1, depth=1, dropout=0.0, out_features=2
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
    ):
        super().__init__()

        self.global_branch = global_branch

        self.projection = Projection(in_features, embedding_size)
        self.common_trunk = CommonTrunk(embedding_size, dropout=dropout)
        self.t2m_branch = VariableBranch(
            embedding_size, dropout=dropout, out_features=out_features
        )
        self.tp_branch = VariableBranch(
            embedding_size, dropout=dropout, out_features=out_features
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
