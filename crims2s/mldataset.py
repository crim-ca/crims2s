"""Generate a ML-ready dataset from the S2S competition data."""

import dask.distributed
import hydra
import logging
import os
import pathlib
import xarray as xr

from .data import normalize_dataset
from .dask import create_dask_cluster
from .util import fix_dataset_dims

_logger = logging.getLogger(__name__)


def datestrings_from_input_dir(input_dir, center):
    input_path = pathlib.Path(input_dir)
    return [
        x.stem.split("-")[-1]
        for x in input_path.iterdir()
        if "t2m" in x.stem and center in x.stem
    ]


def preprocess_single_level_file(d):
    level = int(d.plev[0])
    new_names = {k: f"{k}{level}" for k in d.data_vars}

    return d.rename(new_names).isel(plev=0).drop("plev")


def read_flat_fields(input_dir, center, fields, datestring):
    filenames = [f"{input_dir}/{center}-hindcast-{f}-{datestring}.nc" for f in fields]
    flat_dataset = (
        xr.open_mfdataset(filenames, preprocess=fix_dataset_dims)
        .isel(depth_below_and_layer=0, meanSea=0)
        .drop(["depth_below_and_layer", "meanSea"])
    )

    return flat_dataset


def read_plev_fields(input_dir, center, fields, datestring):
    plev_files = []
    for field, levels in fields.items():
        for level in levels:
            plev_files.append(
                f"{input_dir}/{center}-hindcast-{field}{level}-{datestring}.nc"
            )

    return xr.open_mfdataset(plev_files, preprocess=preprocess_single_level_file)


def make_yearly_examples(
    dataset, obs_terciled,
):
    examples = []
    for i in range(dataset.dims["forecast_year"]):
        to_export_x = (
            dataset.isel(forecast_year=i, forecast_dayofyear=0)
            .to_array()
            .rename("x")
            .transpose("lead_time", "latitude", "longitude", "realization", "variable")
        )
        to_export_y = (
            obs_terciled.sel(forecast_time=to_export_x.forecast_time)
            .to_array()
            .rename("y")
            .transpose("latitude", "longitude", "variable", "lead_time", "category")
        )
        examples.append((to_export_x, to_export_y))

    return examples


def save_examples(examples, output_path):
    for x, y in examples:
        _logger.info(f"Saving year: {int(y.forecast_year)}")
        save_example(x, y, output_path)


def save_example(x, y, output_path):
    forecast_time = y.forecast_time
    year = int(forecast_time.dt.year)
    month = int(forecast_time.dt.month)
    day = int(forecast_time.dt.day)
    filename = f"train_example_{year:04}{month:02}{day:02}.nc"

    output_file = output_path / filename

    x.to_netcdf(output_file, group="/x", mode="w")
    y.to_netcdf(output_file, group="/y", mode="a")


@hydra.main(config_path="conf", config_name="mldataset")
def cli(cfg):
    print(cfg)

    input_dir = hydra.utils.to_absolute_path(cfg.input_dir)
    input_dir_plev = hydra.utils.to_absolute_path(cfg.input_dir_plev)
    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Will output in {output_path}")

    datestrings = datestrings_from_input_dir(input_dir, cfg.center)

    if cfg.index is not None:
        datestrings = datestrings[cfg.index : cfg.index + 1]

    _logger.info(f"Will only operate on datestrings: {datestrings}")

    obs_terciled = xr.open_dataset(cfg.terciled_obs_file)

    for datestring in datestrings:
        _logger.info(f"Processing datestring {datestring}...")

        _logger.info("Reading flat fields...")
        flat_dataset = read_flat_fields(
            cfg.input_dir, cfg.center, cfg.flat_fields, datestring
        )
        _logger.info("Reading fields with vertical levels...")
        plev_dataset = read_plev_fields(
            input_dir_plev, cfg.center, cfg.plev_fields, datestring
        )

        ds = xr.merge([flat_dataset, plev_dataset])
        ds = normalize_dataset(ds)

        _logger.info("Persisting merged dataset...")
        ds = ds.persist()

        examples = make_yearly_examples(ds, obs_terciled)

        _logger.info("Writing examples to disk...")
        save_examples(examples, output_path)


if __name__ == "__main__":
    cli()
