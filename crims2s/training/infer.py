from crims2s.transform import CompositeTransform
import hydra
import logging
import numpy as np
import os
import pathlib
import torch
import tqdm
import xarray as xr

from ..dataset import S2SDataset, TransformedDataset
from ..transform import ExampleToPytorch
from ..util import ECMWF_FORECASTS, collate_with_xarray
from .lightning import S2SBayesModelModule, S2STercilesModule

_logger = logging.getLogger(__name__)


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
            "valid_time": example_forecast.valid_time.data,
            "forecast_time": example_forecast.forecast_time.data,
            "latitude": example_forecast.latitude.data,
            "longitude": example_forecast.longitude.data,
            "category": ["below normal", "near normal", "above normal"],
        }
    ).expand_dims(["forecast_year", "forecast_monthday"])

    return dataset


def concat_predictions(predictions):
    yearly_predictions = {}
    for p in predictions:
        year = int(p.forecast_year.data)
        yearly_list = yearly_predictions.get(year, [])
        yearly_list.append(p)
        yearly_predictions[year] = yearly_list

    nested_datasets = [yearly_predictions[k] for k in sorted(yearly_predictions.keys())]

    yearly_datasets = []
    for l in nested_datasets:
        l = sorted(l, key=lambda x: str(x.forecast_monthday[0]))
        d = xr.concat(l, dim="forecast_monthday")
        yearly_datasets.append(d)

    return xr.concat(yearly_datasets, dim="forecast_year")


def fix_dims_for_output(forecast_dataset):
    """Manipulate the dimensions of the dataset of a single forecast so that we
    can concatenate them easily."""

    return (
        forecast_dataset.stack(
            {"forecast_label": ["forecast_year", "forecast_monthday"]}
        )
        .expand_dims("forecast_time")
        .drop("forecast_label")
        .squeeze("forecast_label")
    )


def example_to_cuda(example):
    new_example = {}

    for k in example:
        if k not in ["monthday", "month"]:
            new_example[k] = example[k].cuda()
        else:
            new_example[k] = example[k]

    return new_example


def find_checkpoint_file(checkpoint_dir):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_files = sorted(list(checkpoint_dir.rglob("*.ckpt")))

    return checkpoint_files[-1]


@hydra.main(config_path="conf", config_name="infer")
def cli(cfg):
    transform = hydra.utils.instantiate(cfg.experiment.transform)

    # Find where we convert to pytorch. For inference we delay the conversion to pytorch
    # because we want to use the xarray data as a template to generate the output file.
    for i, t in enumerate(transform.transforms):
        if isinstance(t, ExampleToPytorch):
            pytorch_transform_idx = i

    last_transform = CompositeTransform(transform.transforms[pytorch_transform_idx:])
    transform.transforms = transform.transforms[:pytorch_transform_idx]

    years = list(range(cfg.begin, cfg.end))

    if cfg.experiment.dataset.index is not None:
        month, day = ECMWF_FORECASTS[cfg.experiment.dataset.index]
        label = f"{month:02}{day:02}.nc"

        _logger.info("Targetting monthday %s", label)
        name_filter = lambda x: x.endswith(label)
    else:
        name_filter = None

    dataset = TransformedDataset(
        S2SDataset(
            cfg.test_dataset_dir,
            years=years,
            name_filter=name_filter,
            include_features=True,
        ),
        transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        batch_sampler=None,
        collate_fn=collate_with_xarray,
        num_workers=int(cfg.num_workers),
        shuffle=False,
    )

    checkpoint_path = find_checkpoint_file(
        hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    )
    _logger.info(f"Will run on checkpoint {checkpoint_path}")

    model = hydra.utils.instantiate(cfg.experiment.model)
    optimizer = hydra.utils.call(cfg.experiment.optimizer, model)

    lightning_module = S2STercilesModule.load_from_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer
    )
    lightning_module.eval()
    lightning_module.freeze()

    lightning_module.cuda()

    datasets_of_examples = []
    for example in tqdm.tqdm(dataloader):
        example_forecast = example["obs"]

        pytorch_example = last_transform(example)
        pytorch_example = example_to_cuda(pytorch_example)

        t2m_terciles, tp_terciles, *_ = lightning_module(pytorch_example)

        dataset = terciles_pytorch_to_xarray(
            t2m_terciles.cpu(),
            tp_terciles.cpu(),
            example_forecast,
            dims=["batch", "category", "lead_time", "latitude", "longitude"],
        )
        datasets_of_examples.append(fix_dims_for_output(dataset))

    sorted_datasets = sorted(
        datasets_of_examples, key=lambda x: str(x.forecast_time.data[0])
    )

    ml_prediction = (
        xr.concat(sorted_datasets, dim="forecast_time")
        .drop("valid_time")
        .squeeze("batch")
    )

    _logger.info(f"Outputting forecasts to {os.getcwd() + '/' + cfg.output_file}.")
    ml_prediction.to_netcdf(cfg.output_file)


if __name__ == "__main__":
    cli()
