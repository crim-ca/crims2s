"""Generate a ML-ready dataset from the S2S competition data."""

import datetime
import hydra
import logging
import omegaconf
import pathlib
import xarray as xr

from .transform import normalize_dataset
from .util import fix_dataset_dims

_logger = logging.getLogger(__name__)


def obs_of_forecast(forecast, raw_obs):
    valid_time = forecast.valid_time.compute()
    obs = raw_obs.sel(time=valid_time)
    return obs


class ExamplePartMaker:
    """Fabricated abstraction that makes part of an example. Abstraction is useful 
    because it allows the user to choose what he wants his examples made of. Plausible
    par of examples include features, observations, terciles, etc."""

    def __call__(self, year, example):
        """Generate the part of an example for a given year."""
        raise NotImplementedError


class FeatureExamplePartMaker(ExamplePartMaker):
    def __init__(self, features):
        self.features = features

    def __call__(self, year, example):
        return (
            self.features.isel(forecast_dayofyear=0)
            .sel(forecast_year=year)
            .to_array()
            .rename("features")
            .transpose("lead_time", "latitude", "longitude", "realization", "variable")
        )


class ModelExamplePartMaker(ExamplePartMaker):
    def __init__(self, model):
        self.model = model

    def __call__(self, year, example):
        return (
            self.model.isel(forecast_dayofyear=0)
            .sel(forecast_year=year)
            .transpose("lead_time", "latitude", "longitude", "realization",)
        )


class TargetExamplePartMaker(ExamplePartMaker):
    def __init__(self, target):
        self.target = target

    def __call__(self, year, example):
        model = example["model"]

        return (
            self.target.sel(forecast_time=model.forecast_time)
            .to_array()
            .rename("target")
            .transpose("latitude", "longitude", "variable", "lead_time", "category")
        )


class ObsExamplePartMaker(ExamplePartMaker):
    def __init__(self, obs):
        self.obs = obs

    def __call__(self, year, example):
        model = example["model"]
        return obs_of_forecast(model, self.obs)


class EdgesExamplePartMaker(ExamplePartMaker):
    def __init__(self, edges):
        self.edges = edges

    def __call__(self, year, example):
        pass


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
    flat_dataset = xr.open_mfdataset(filenames, preprocess=fix_dataset_dims)

    for dim in ["depth_below_and_layer", "meanSea"]:
        if dim in flat_dataset.dims:
            flat_dataset = flat_dataset.isel({dim: 0})
            flat_dataset = flat_dataset.drop(dim)

    return flat_dataset


def read_plev_fields(input_dir, center, fields, datestring):
    plev_files = []
    for field, levels in fields.items():
        for level in levels:
            plev_files.append(
                f"{input_dir}/{center}-hindcast-{field}{level}-{datestring}.nc"
            )

    return xr.open_mfdataset(plev_files, preprocess=preprocess_single_level_file)


def make_yearly_examples(features, model, obs_terciled, raw_obs):
    examples = []
    for i in range(features.dims["forecast_year"]):
        year = features.forecast_year[i]
        example = {}

        feature_maker = FeatureExamplePartMaker(features)
        example["features"] = feature_maker(year, example)

        model_maker = ModelExamplePartMaker(model)
        example["model"] = model_maker(year, example)

        target_maker = TargetExamplePartMaker(obs_terciled)
        example["target"] = target_maker(year, example)

        obs_maker = ObsExamplePartMaker(raw_obs)
        example["obs"] = obs_maker(year, example)

        week = example["features"].forecast_time.dt.weekofyear
        _logger.debug(f"Selecting climatology from week of year {week}")

        examples.append(example)

    return examples


def save_examples(examples, output_path):
    for e in examples:
        _logger.info(f"Saving year: {int(e['target'].forecast_year)}")
        save_example(e, output_path)


def save_example(example, output_path: pathlib.Path):
    """Save an example to a single netcdf file. x is the input features. obs is the
    observations for every valid time in the forecast. y is the terciled target
    distribution (below, within, above normal)."""
    forecast_time = example["target"].forecast_time
    year = int(forecast_time.dt.year)
    month = int(forecast_time.dt.month)
    day = int(forecast_time.dt.day)
    filename = f"train_example_{year:04}{month:02}{day:02}.nc"

    output_file = output_path / filename

    if output_file.exists():
        _logger.info(f"Target file {output_file} already exists. Replacing.")
        output_file.unlink()

    for k in example:
        mode = "a" if output_file.exists() else "w"
        example[k].to_netcdf(output_file, group=f"/{k}", mode=mode, compute=True)


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

    edges = xr.open_dataset(cfg.edges_file)
    obs_terciled = xr.open_dataset(cfg.terciled_obs_file)
    raw_obs = read_raw_obs(cfg.observations.t2m_file, cfg.observations.pr_file)

    for datestring in datestrings:
        _logger.info(f"Processing datestring {datestring}...")

        _logger.info("Reading flat fields...")
        flat_dataset = read_flat_fields(
            cfg.input_dir, cfg.center, cfg.fields.flat, datestring
        )

        _logger.info("Reading fields with vertical levels...")
        plev_fields = cfg.fields.get("plev", dict())
        datasets = [flat_dataset]
        if plev_fields:
            plev_dataset = read_plev_fields(
                input_dir_plev, cfg.center, plev_fields, datestring
            )
            datasets.append(plev_dataset)

        features = xr.merge(datasets)
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

        examples = make_yearly_examples(features, model, obs_terciled, raw_obs)

        _logger.info("Writing examples to disk...")
        save_examples(examples, output_path)


if __name__ == "__main__":
    cli()
