"""Generate a ML-ready dataset from the S2S competition data."""

import hydra
import logging
import omegaconf
import pathlib
import xarray as xr

from .transform import normalize_dataset
from .dask import create_dask_cluster
from .util import fix_dataset_dims

_logger = logging.getLogger(__name__)


def datestrings_from_input_dir(input_dir, center):
    input_path = pathlib.Path(input_dir)
    return sorted(
        [
            x.stem.split("-")[-1]
            for x in input_path.iterdir()
            if "t2m" in x.stem and center in x.stem
        ]
    )


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
    features, model, obs_terciled,
):
    examples = []
    for i in range(features.dims["forecast_year"]):
        to_export_x = (
            features.isel(forecast_year=i, forecast_dayofyear=0)
            .to_array()
            .rename("x")
            .transpose("lead_time", "latitude", "longitude", "realization", "variable")
        )

        to_export_model = model.isel(forecast_year=i, forecast_dayofyear=0).transpose(
            "lead_time", "latitude", "longitude", "realization",
        )

        to_export_y = (
            obs_terciled.sel(forecast_time=to_export_x.forecast_time)
            .to_array()
            .rename("y")
            .transpose("latitude", "longitude", "variable", "lead_time", "category")
        )
        examples.append((to_export_x, to_export_y, to_export_model))

    return examples


def save_examples(examples, output_path):
    for x, y, model, obs in examples:
        _logger.info(f"Saving year: {int(y.forecast_year)}")
        save_example(x, model, y, obs, output_path)


def save_example(x, model, y, obs, output_path):
    """Save an example to a single netcdf file. x is the input features. obs is the
    observations for every valid time in the forecast. y is the terciled target
    distribution (below, within, above normal)."""
    forecast_time = y.forecast_time
    year = int(forecast_time.dt.year)
    month = int(forecast_time.dt.month)
    day = int(forecast_time.dt.day)
    filename = f"train_example_{year:04}{month:02}{day:02}.nc"

    output_file = output_path / filename

    x.to_netcdf(output_file, group="/x", mode="w")
    model.to_netcdf(output_file, group="/model", mode="a")
    y.to_netcdf(output_file, group="/y", mode="a")
    obs.to_netcdf(output_file, group="/obs", mode="a")


def remove_nans(dataset):
    """Little hack to make sure we can train an not have everything explode.
    The data should be approx. zero centered so replacing the nans with zeros should
    not create extreme values. The dataset contains an LSM so the network should figure
    out that it should not use those zeroes that are outside of the land sea mask."""
    return dataset.fillna(0.0)


def read_raw_obs(t2m_file, pr_file, preprocess=lambda x: x):
    t2m = preprocess(xr.open_dataset(t2m_file))
    pr = preprocess(xr.open_dataset(pr_file))

    return xr.merge([t2m, pr])


def obs_of_forecast(forecast, raw_obs):
    valid_time = forecast.valid_time.compute()
    obs = raw_obs.sel(time=valid_time)
    return obs


@hydra.main(config_path="conf", config_name="mldataset")
def cli(cfg):
    print(cfg)

    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "confg.yaml", "w") as f:
        cfg_string = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)
        f.write(cfg_string)

    input_dir = hydra.utils.to_absolute_path(cfg.input_dir)
    input_dir_plev = hydra.utils.to_absolute_path(cfg.input_dir_plev)

    _logger.info(f"Will output in {output_path}")

    datestrings = datestrings_from_input_dir(input_dir, cfg.center)

    if cfg.index is not None:
        datestrings = datestrings[cfg.index : cfg.index + 1]

    _logger.info(f"Will only operate on datestrings: {datestrings}")

    obs_terciled = xr.open_dataset(cfg.terciled_obs_file)
    raw_obs = read_raw_obs(cfg.observations.t2m_file, cfg.observations.pr_file)

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

        features = xr.merge([flat_dataset, plev_dataset])
        features = normalize_dataset(features)
        features = remove_nans(features)

        model = flat_dataset[["t2m", "tp"]]
        # Super evil temporary hack: the ECMWF data is sprinkled with nans, but it looks
        # like what are nans should be zeros. So we replace them arbitratily with zeros.
        model["tp"] = model["tp"].fillna(0.0)

        # A lot of the fields we use have null lead time 0 for ECMWF.
        # We remove lead time 1 for all the dataset.
        # Shouldn't matter too much since we are interested in leads 14-42 anyways.
        features = features.isel(lead_time=slice(1, None))
        model = model.isel(lead_time=slice(1, None))

        _logger.info("Persisting merged dataset...")
        features = features.persist()

        examples = make_yearly_examples(features, model, obs_terciled)
        examples = [(x, y, m, obs_of_forecast(x, raw_obs)) for x, y, m in examples]

        _logger.info("Writing examples to disk...")
        save_examples(examples, output_path)


if __name__ == "__main__":
    cli()
