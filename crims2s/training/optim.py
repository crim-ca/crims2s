import torch

from collections.abc import Mapping, Iterable
from .optim_spec import create_optim


def bayes_optimizer(model, cls, forecast_lr, weights_lr, **kwargs):
    weight_params = set(
        set(model.t2m_weight_model.parameters()).union(
            model.tp_weight_model.parameters()
        )
    )
    other_params = set(model.parameters()) - weight_params

    return cls(
        [
            {"params": list(other_params), "lr": forecast_lr, **kwargs},
            {"params": list(weight_params), "lr": weights_lr, **kwargs},
        ]
    )


def bayes_adam(model, forecast_lr, weights_lr):
    return bayes_optimizer(model, torch.optim.Adam, forecast_lr, weights_lr)


def emos_optimizer(model, cls, t2m_lr, tp_lr, **kwargs):
    t2m_parameters = [p for (n, p) in model.named_parameters() if "t2m_model" in n]
    tp_parameters = [p for (n, p) in model.named_parameters() if "tp_model" in n]

    return cls(
        [
            {"params": t2m_parameters, "lr": t2m_lr, **kwargs},
            {"params": tp_parameters, "lr": tp_lr, **kwargs},
        ]
    )


def emos_adam(model, t2m_lr, tp_lr, **kwargs):
    return emos_optimizer(model, torch.optim.Adam, t2m_lr, tp_lr, **kwargs)


def adam(model, **kwargs):
    return torch.optim.Adam(model.parameters(), **kwargs)


def adagrad(model, **kwargs):
    return torch.optim.Adagrad(model.parameters(), **kwargs)


class OptimizerMaker:
    """Base class that I use to detect if we have an optimizer factory instead
    of an optimizer."""

    def __call__(self, model):
        raise NotImplementedError


class OptimSpecMaker(OptimizerMaker):
    def __init__(self, model, optim_spec):
        self.optim_spec = optim_spec

    def __call__(self, model):
        return create_optim(model, self.optim_spec, check_requires_grad=True)


def three_groups_scheduler(optimizer, freeze_forecast_epoch):
    forecast_lambda = lambda epoch: 1.0 if epoch < freeze_forecast_epoch else 0.0
    weights_lambda = lambda epoch: 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [forecast_lambda, forecast_lambda, weights_lambda]
    )
