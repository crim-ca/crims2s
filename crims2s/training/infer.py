import hydra
import logging
import numpy as np
import os
import torch
import xarray as xr

from ..dataset import S2SDataset
from ..util import ECMWF_FORECASTS
from .lightning import S2SLightningModule

_logger = logging.getLogger(__name__)


def compute_edges_cdf_from_distribution(distribution, edges):
    edges_nan_mask = edges.isnan()
    edges[edges_nan_mask] = 0.0
    cdf = distribution.cdf(edges)
    edges[edges_nan_mask] = np.nan
    cdf[edges_nan_mask] = np.nan

    return cdf


def edges_cdf_to_terciles(edges_cdf):
    return torch.stack(
        [edges_cdf[0], edges_cdf[1] - edges_cdf[0], 1.0 - edges_cdf[1],], dim=0
    )


def terciles_pytorch_to_xarray(
    t2m, tp, example_forecast, dims=["category", "lead_time", "latitude", "longitude"]
):
    t2m_array = xr.DataArray(data=t2m.detach().numpy(), dims=dims, name="t2m")
    tp_array = xr.DataArray(data=tp.detach().numpy(), dims=dims, name="tp")
    dataset = xr.Dataset(data_vars={"t2m": t2m_array, "tp": tp_array,})

    dataset = dataset.assign_coords(
        {
            "forecast_year": example_forecast.forecast_year.data,
            "forecast_monthday": example_forecast.forecast_monthday.data,
            "lead_time": example_forecast.lead_time.data,
            "valid_time": example_forecast.valid_time,
            "forecast_time": example_forecast.forecast_time.data,
            "latitude": example_forecast.latitude.data,
            "longitude": example_forecast.longitude.data,
            "category": ["below normal", "near normal", "above normal"],
        }
    ).expand_dims(["forecast_year", "forecast_monthday"])

    return dataset


@hydra.main(config_path="conf", config_name="infer")
def cli(cfg):
    transform = hydra.utils.instantiate(cfg.transform)
    years = list(range(cfg.begin, cfg.end))

    if cfg.index is not None:
        month, day = ECMWF_FORECASTS[cfg.index]
        label = f"{month:02}{day:02}.nc"

        _logger.info("Targetting monthday %s", label)
        name_filter = lambda x: x.endswith(label)
    else:
        name_filter = None

    dataset = S2SDataset(
        cfg.dataset_dir, years=years, name_filter=name_filter, include_features=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        num_workers=int(cfg.num_workers),
        shuffle=False,
    )

    model = S2SLightningModule.load_from_checkpoint(cfg.checkpoint_dir).eval().freeze()

    datasets_of_examples = []
    for example in dataloader:
        transformed_example = transform(example)
        t2m_dist, tp_dist = model(transformed_example)

        t2m_edges = torch.cat(
            [torch.full((2, 1, 121, 240), np.nan), transformed_example["edges_t2m"]], 1
        )
        t2m_cdf = compute_edges_cdf_from_distribution(t2m_dist, t2m_edges)

        tp_edges = torch.cat(
            [torch.full((2, 1, 121, 240), np.nan), transformed_example["edges_tp"]], 1
        )
        tp_cdf = compute_edges_cdf_from_distribution(tp_dist, tp_edges)

        t2m_terciles = edges_cdf_to_terciles(t2m_cdf)
        tp_terciles = edges_cdf_to_terciles(tp_cdf)

        example_forecast = example["model"]

        dataset = terciles_pytorch_to_xarray(
            t2m_terciles, tp_terciles, example_forecast
        )
        datasets_of_examples.append(dataset)

    ml_prediction = xr.combine_by_coords(datasets_of_examples)

    _logger.info(f"Outputting forecasts to {os.cwd() + cfg.output_filename}.")
    ml_prediction.to_netcdf(cfg.output_filename)


if __name__ == "__main__":
    cli()
