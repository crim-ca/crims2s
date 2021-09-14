import torch


def bayes_optimizer(model, cls, forecast_lr, weights_lr, **kwargs):
    return cls(
        [
            {"params": model.forecast_model.parameters(), "lr": forecast_lr, **kwargs},
            {"params": model.t2m_weight_model.parameters(), "lr": weights_lr, **kwargs},
            {"params": model.tp_weight_model.parameters(), "lr": weights_lr, **kwargs},
        ]
    )


def bayes_adam(model, forecast_lr, weights_lr):
    return bayes_optimizer(model, torch.optim.Adam, forecast_lr, weights_lr)


def adam(model, **kwargs):
    return torch.optim.Adam(model.parameters(), **kwargs)
