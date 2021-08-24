import numpy as np
import torch
import torch.distributions
import xarray as xr

from .util import std_estimator


def fit_normal_xarray(array: xr.DataArray, dim=None) -> xr.Dataset:
    model_t2m_mean = array.mean(dim=dim).rename(f"{array.name}_mu")
    model_t2m_std = std_estimator(array, dim=dim).rename(f"{array.name}_sigma")

    return xr.merge([model_t2m_mean, model_t2m_std])


def fit_gamma_xarray(
    array: xr.DataArray, dim=None, regularization=1e-9, **kwargs
) -> xr.Dataset:
    """Given a DataArray, fit a Gamma distribution across dimensions dim. Return a 
    dataset with the distribution parameters."""
    # Use method of moments for initial estimate.
    a_hat_xarray = array.mean(dim=dim) ** 2 / (array.var(dim=dim) + regularization)
    b_hat_xarray = (array.mean(dim=dim) + regularization) / (
        array.var(dim=dim) + regularization
    )

    transposed = array.transpose(dim, ...)

    alpha, beta = fit_gamma_pytorch(
        transposed.data.compute(),
        a_hat_xarray.data.compute(),
        b_hat_xarray.data.compute(),
        regularization=regularization,
        **kwargs,
    )

    alpha_xarray = xr.zeros_like(a_hat_xarray).rename(f"{a_hat_xarray.name}_alpha")
    beta_xarray = xr.zeros_like(b_hat_xarray).rename(f"{a_hat_xarray.name}_beta")

    alpha_xarray.data = alpha.numpy()
    beta_xarray.data = beta.numpy()

    return xr.merge([alpha_xarray, beta_xarray])


def fit_gamma_pytorch(
    data,
    a_hat,
    b_hat,
    regularization=1e-9,
    max_epochs=500,
    lr=1e-2,
    tol=1e-5,
    patience=5,
    return_losses=False,
):
    n_iter_waited = 0

    alpha = torch.tensor(a_hat, requires_grad=True)
    beta = torch.tensor(b_hat, requires_grad=True)
    data = torch.tensor(data)

    optimizer = torch.optim.Adam([alpha, beta], lr=lr)
    log_likelihoods = []
    for epoch in range(max_epochs):
        clamped_alpha = torch.clamp(alpha, min=regularization)
        clamped_beta = torch.clamp(beta, min=regularization)

        estimated_gamma = torch.distributions.Gamma(clamped_alpha, clamped_beta)

        loss = -estimated_gamma.log_prob(data + regularization).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if len(log_likelihoods) > 0:
            best_loss = np.array(log_likelihoods).min()
            if np.abs(best_loss - loss.detach()) < tol:
                n_iter_waited += 1

                if n_iter_waited >= patience:
                    break

        log_likelihoods.append(loss.detach().item())

    alpha, beta = (
        torch.clamp(alpha, min=regularization).detach(),
        torch.clamp(beta, min=regularization).detach(),
    )

    if return_losses:
        return alpha, beta, log_likelihoods
    else:
        return alpha, beta
