import torch
import torch.distributions
import torch.nn as nn


class PerceptronForecastModel(nn.Module):
    def __init__(self, in_features, embedding_size, moments=True, regularization=1e-6):
        super().__init__()

        self.moments = moments
        in_features = in_features * 2 if moments else in_features

        self.perceptron = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, 8),
        )

        self.regularization = regularization

    def forward(self, batch):
        x = batch["features_features"]

        # Take temporal average.
        x = torch.mean(x, dim=1)

        # Compute mean and std of features across members.
        if self.moments:
            # Use average and STD of members as features.
            x = torch.cat([x.mean(dim=-2), x.std(dim=-2)], dim=-1)
        else:
            # Select first member.
            x = x[..., 0, :]

        x = self.perceptron(x)

        batch_size = x.shape[0]
        lead_times = 2
        x = x.reshape(batch_size, lead_times, 121, 240, 4)

        t2m_loc, t2m_scale = (
            batch["model_parameters_t2m_mu"],
            batch["model_parameters_t2m_sigma"],
        )

        tp_loc, tp_scale = (
            batch["model_parameters_tp_mu"],
            batch["model_parameters_tp_sigma"],
        )

        t2m_loc = t2m_loc + x[..., 0]
        t2m_scale = t2m_scale + x[..., 1]
        t2m_scale = torch.clip(t2m_scale, min=1e-6)

        tp_loc = tp_loc + x[..., 2]
        tp_scale = tp_scale + x[..., 3]
        tp_scale = torch.clip(tp_scale, min=1e-6)

        t2m_dist = torch.distributions.Normal(loc=t2m_loc, scale=t2m_scale)
        tp_dist = torch.distributions.Normal(loc=tp_loc, scale=tp_scale)

        return t2m_dist, tp_dist
