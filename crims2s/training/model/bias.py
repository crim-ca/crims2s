import torch
import torch.nn as nn

from .util import MonthlyMultiplexer, WeeklyMultiplexer


class BiasCorrectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.parameter.Parameter(torch.normal(torch.zeros(2, 121, 240)))

    def forward(self, model_parameter):
        return model_parameter + self.bias


class WeeklyBiasCorrectionModel(WeeklyMultiplexer):
    def __init__(self):
        super().__init__(BiasCorrectionModel)


class MonthlyBiasCorrectionModel(MonthlyMultiplexer):
    def __init__(self):
        super().__init__(BiasCorrectionModel)


class ModelParameterBiasCorrection(nn.Module):
    def __init__(self, inner_cls=BiasCorrectionModel):
        super().__init__()

        keys_to_correct = [
            "model_parameters_t2m_mu",
            "model_parameters_t2m_sigma",
            "model_parameters_tp_mu",
            "model_parameters_tp_sigma",
            "model_parameters_tp_cube_root_mu",
            "model_parameters_tp_cube_root_sigma",
        ]
        models = {k: inner_cls() for k in keys_to_correct}

        self.models = nn.ModuleDict(models)

    def forward(self, batch):
        for k in self.models:
            batch[k] = self.models[k](batch[k])

        return batch


class ModelParameterWeeklyBiasCorrection(ModelParameterBiasCorrection):
    def __init__(self):
        super().__init__(inner_cls=WeeklyBiasCorrectionModel)

    def forward(self, batch):
        for k in self.models:
            batch[k] = self.models[k](batch["monthday"], batch[k])

        return batch


class ModelParameterMonthlyBiasCorrection(ModelParameterBiasCorrection):
    def __init__(self):
        super().__init__(inner_cls=MonthlyBiasCorrectionModel)

    def forward(self, batch):
        for k in self.models:
            batch[k] = self.models[k](batch["month"], batch[k])

        return batch


class DistributionBiasCorrectionModel(nn.Module):
    def __init__(self, inner_model, regularization=1e-9):
        super().__init__()
        self.inner = inner_model
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
