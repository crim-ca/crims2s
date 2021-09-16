import torch
import torch.nn as nn

from .util import MonthlyModel, WeeklyModel


class BiasCorrectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.parameter.Parameter(torch.normal(torch.zeros(2, 121, 240)))

    def forward(self, model_parameter):
        return model_parameter + self.bias


class ModelParameterBiasCorrection(nn.Module):
    def __init__(self):
        super().__init__()

        keys_to_correct = [
            "model_parameters_t2m_mu",
            "model_parameters_t2m_sigma",
            "model_parameters_tp_mu",
            "model_parameters_tp_sigma",
            "model_parameters_tp_cube_root_mu",
            "model_parameters_tp_cube_root_sigma",
        ]
        models = {k: BiasCorrectionModel() for k in keys_to_correct}

        self.models = nn.ModuleDict(models)

    def forward(self, batch):
        for k in self.models:
            batch[k] = self.models[k](batch[k])

        return batch


class DistributionBiasCorrectionModel(nn.Module):
    def __init__(self, regularization=1e-9):
        super().__init__()
        self.inner = ModelParameterBiasCorrection()
        self.regularization = regularization

    def forward(self, batch):
        batch = self.inner(batch)

        t2m_mu = batch["model_parameters_t2m_mu"]
        t2m_sigma = torch.clip(
            batch["model_parameters_t2m_sigma"], min=self.regularization
        )

        tp_mu = batch["model_parameters_tp_cube_root_mu"]
        tp_sigma = torch.clip(
            batch["model_parameters_tp_cube_root_sigma"], min=self.regularization
        )

        t2m_distribution = torch.distributions.Normal(loc=t2m_mu, scale=t2m_sigma)
        tp_distribution = torch.distributions.Normal(loc=tp_mu, scale=tp_sigma)

        return t2m_distribution, tp_distribution


class MonthlyBiasCorrection(MonthlyModel):
    def __init__(self):
        super().__init__(DistributionBiasCorrectionModel)


class WeeklyBiasCorrection(WeeklyModel):
    def __init__(self):
        super().__init__(DistributionBiasCorrectionModel)

