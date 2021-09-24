import torch.distributions
import torch.nn as nn

from .bias import ModelParameterBiasCorrection
from .util import DistributionModelAdapter


class ConvBlock(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.conv1 = nn.Conv3d(
            embedding_size,
            embedding_size,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            padding_mode="circular",
        )
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(
            embedding_size,
            embedding_size,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
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


class ConvModel(nn.Module):
    def __init__(self, in_features, out_features, n_blocks, embedding_size):
        super().__init__()

        self.conv1 = nn.Conv3d(in_features, embedding_size, kernel_size=(1, 1, 1),)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm3d(embedding_size)

        self.blocks = nn.ModuleList(
            [ConvBlock(embedding_size) for _ in range(n_blocks)]
        )

        self.conv_weeks_34 = nn.Conv2d(embedding_size, out_features, kernel_size=(1, 1))
        self.conv_weeks_56 = nn.Conv2d(embedding_size, out_features, kernel_size=(1, 1))

    def forward(self, input):
        x = self.act1(self.bn1(self.conv1(input)))

        for b in self.blocks:
            x = b(x)

        x = x.mean(dim=-1)

        post_weeks_34 = self.conv_weeks_34(x)
        post_weeks_56 = self.conv_weeks_56(x)

        return torch.stack([post_weeks_34, post_weeks_56], dim=-1)


class DistributionConvPostProcessing(nn.Module):
    def __init__(self, conv_model, regularization=1e-9, debias=False):
        super().__init__()
        self.conv_model = conv_model
        self.regularization = regularization

        if debias:
            self.debias_model = ModelParameterBiasCorrection()
        else:
            self.debias_model = None

    def forward(self, batch):
        if self.debias_model is not None:
            batch = self.debias_model(batch)

        x = batch["features_features"]

        x = x[..., 0, :]  # Grab the first member.
        x = torch.transpose(x, 1, -1)  # Swap dims and depth.

        post_processing = torch.transpose(self.conv_model(x), 1, -1)

        t2m_mu = batch["model_parameters_t2m_mu"] + post_processing[..., 0]
        t2m_sigma = batch["model_parameters_t2m_sigma"] + post_processing[..., 1]

        tp_mu = batch["model_parameters_tp_cube_root_mu"] + post_processing[..., 2]
        tp_sigma = (
            batch["model_parameters_tp_cube_root_sigma"] + post_processing[..., 3]
        )

        t2m_dist = torch.distributions.Normal(
            t2m_mu, torch.clip(t2m_sigma, min=self.regularization)
        )
        tp_dist = torch.distributions.Normal(
            tp_mu, torch.clip(tp_sigma, min=self.regularization)
        )

        return t2m_dist, tp_dist


class ConvPostProcessing(DistributionModelAdapter):
    def __init__(self, in_features, n_blocks, embedding_size, debias=False):
        conv_model = ConvModel(in_features, 4, n_blocks, embedding_size)
        distribution_model = DistributionConvPostProcessing(conv_model, debias=debias)

        super().__init__(distribution_model)
