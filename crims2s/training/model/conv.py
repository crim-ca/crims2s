import numpy as np 
import torch
import torch.distributions
import torch.nn as nn
from typing import Tuple, Mapping

from .bias import ModelParameterBiasCorrection
from .bayes import GlobalBranch, VariableBranch, CommonTrunk, Projection
from .util import DistributionModelAdapter


class ConvPostProcessingModel(nn.Module):
    def __init__(
        self,
        in_features,
        embedding_size,
        global_branch=True,
        dropout=0.0,
        out_features=4,
        moments=False,
        variable_branch_blocks=3,
        common_trunk_blocks=3,
        global_stride=2,
        global_dilation=1,
        global_padding=1,
        global_n_constant_blocks=0,
        flatten_time=True,
    ):
        super().__init__()

        self.global_branch = global_branch

        self.projection = Projection(
            in_features, embedding_size, moments=moments, flatten_time=flatten_time
        )

        if flatten_time:
            depth = 1
        else:
            depth = 3

        self.common_trunk = CommonTrunk(
            embedding_size, n_blocks=common_trunk_blocks, dropout=dropout, depth=depth,
        )
        self.t2m_branch = VariableBranch(
            embedding_size,
            dropout=dropout,
            out_features=out_features,
            n_blocks=variable_branch_blocks,
            depth=depth,
        )
        self.tp_branch = VariableBranch(
            embedding_size,
            dropout=dropout,
            out_features=out_features,
            n_blocks=variable_branch_blocks,
            depth=depth,
        )

        if self.global_branch:
            self.global_branch = GlobalBranch(
                embedding_size,
                embedding_size,
                stride=global_stride,
                dilation=global_dilation,
                padding=global_padding,
                n_constant_blocks=global_n_constant_blocks,
            )

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

        # Every branch outputs 4 channels: mu w34, mu w56, std w34, std w56.
        # Reshape accordingly.
        batch_size = x_t2m.shape[0]
        x_t2m = x_t2m.reshape(batch_size, 2, 121, 240, 2)
        x_tp = x_tp.reshape(batch_size, 2, 121, 240, 2)

        return x_t2m, x_tp


class DistributionConvPostProcessing(nn.Module):
    def __init__(self, conv_model, regularization=1e-9, debias=False):
        super().__init__()
        self.conv_model = conv_model
        self.regularization = regularization

        if debias:
            self.debias_model = ModelParameterBiasCorrection()
        else:
            self.debias_model = None

    def forward(
        self, batch: Mapping
    ) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        if self.debias_model is not None:
            batch = self.debias_model(batch)

        x = batch["features_features"]

        t2m_post, tp_post = self.conv_model(x)

        t2m_post, tp_post = (
            torch.transpose(t2m_post, 1, -1),
            torch.transpose(tp_post, 1, -1),
        )

        t2m_mu = batch["model_parameters_t2m_mu"] + t2m_post[..., 0]
        t2m_sigma = batch["model_parameters_t2m_sigma"] + t2m_post[..., 1]

        tp_mu = batch["model_parameters_tp_cube_root_mu"] + tp_post[..., 0]
        tp_sigma = batch["model_parameters_tp_cube_root_sigma"] + tp_post[..., 1]

        t2m_dist = torch.distributions.Normal(
            t2m_mu, torch.clip(t2m_sigma, min=self.regularization)
        )
        tp_dist = torch.distributions.Normal(
            tp_mu, torch.clip(tp_sigma, min=self.regularization)
        )

        return t2m_dist, tp_dist


class TercilesConvPostProcessing(DistributionModelAdapter):
    def __init__(
        self, in_features, embedding_size, debias=False, **kwargs,
    ):
        conv_model = ConvPostProcessingModel(in_features, embedding_size, **kwargs)
        distribution_model = DistributionConvPostProcessing(conv_model, debias=debias)

        super().__init__(distribution_model)


class ConvBlock(nn.Module):
    def __init__(self, embedding_size, kernel_size=(3, 3, 3)):
        super().__init__()

        if kernel_size[0] < 3:
            padding = int(np.round(kernel_size[0] / 3))
        else:
            padding = int(np.ceil(kernel_size[0] / 3))

        padding = (padding, padding, padding)

        self.conv1 = nn.Conv3d(
            embedding_size,
            embedding_size,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv3d(
            embedding_size,
            embedding_size,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        self.act2 = nn.LeakyReLU()

        self.bn1 = nn.BatchNorm3d(embedding_size)
        self.bn2 = nn.BatchNorm3d(embedding_size)

    def forward(self, input):
        x = input
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x) + x))
        return x


class ConvModelJG(nn.Module):
    def __init__(self, in_features, out_features, embedding_size, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv3d(in_features, embedding_size, kernel_size=(1, 1, 1))
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm3d(embedding_size)

        self.blocks = nn.ModuleList(
            [
            ConvBlock(embedding_size, kernel_size=(3, 3, 3)),
            ConvBlock(embedding_size, kernel_size=(3, 3, 3))]
            )

        self.conv_weeks_34 = nn.Conv2d(embedding_size, out_features, kernel_size=(1, 1))
        self.conv_weeks_56 = nn.Conv2d(embedding_size, out_features, kernel_size=(1, 1))


    def forward(self, input):
        x= input
        x = torch.transpose(x, -1, 1)
        x = torch.squeeze(x, -2)  # Swap channels and time dim.
        x = self.act1(self.bn1(self.conv1(x)))

        for b in self.blocks:
            x = b(x)

        x = x.mean(dim=-1)

        post_weeks_34 = self.conv_weeks_34(x)
        post_weeks_56 = self.conv_weeks_56(x)

        stack = torch.stack([post_weeks_34, post_weeks_56], dim=-1)

        t2m_post = stack[:, 0:2]
        tp_post = stack[:, 2:4]

        return t2m_post, tp_post


class ConvPostProcessingJG(DistributionModelAdapter):
    def __init__(self, in_features, embedding_size, kernel_size=(1, 1, 1), debias=False):
        conv_model = ConvModelJG(in_features, 4,  embedding_size, kernel_size)
        distribution_model = DistributionConvPostProcessing(conv_model, debias=debias)

        super().__init__(distribution_model)    