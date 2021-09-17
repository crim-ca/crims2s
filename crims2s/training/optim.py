import torch


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


def adam(model, **kwargs):
    return torch.optim.Adam(model.parameters(), **kwargs)


def adagrad(model, **kwargs):
    return torch.optim.Adagrad(model.parameters(), **kwargs)
