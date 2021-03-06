"""Various constants and distributions that decribe our dataset. Intended use
is normalization of the fields before sending them to a neural net.

See notebook distributions-of-parameters.ipynb"""

import logging
import numpy as np
import torch
import random
import xarray as xr

from .util import add_biweekly_dim, obs_to_biweekly, std_estimator, fix_s2s_dataset_dims

_logger = logging.getLogger(__name__)


FIELD_MEAN = {
    "gh10": 30583.0,
    "gh100": 16070.0,
    "gh1000": 76.19,
    "gh200": 11765.0,
    "gh500": 5524.374,
    "gh850": 1403.0,
    "lsm": 0.0,
    "msl": 100969.28,
    "orog": 387.1,
    "siconc": 0.17,
    "sst": 286.96,
    "st100": 268.75,
    "st20": 268.69,
    "sm20": 250.68,
    "t2m": 278.2237,
    "tp": 34.1,
    "u1000": -0.17,
    "u850": 1.26,
    "u500": 6.43,
    "u200": 14.43,
    "u100": 5.30,
    "v1000": 0.18,
    "v850": 0.11,
    "v500": -0.03,
    "v200": -0.01,
    "v100": 0.10,
}

FIELD_STD = {
    "gh10": 993.0,
    "gh100": 577.0,
    "gh1000": 110.14,
    "gh200": 605.0,
    "gh500": 341.80862,
    "gh850": 149.6,
    "lsm": 1.0,
    "msl": 1343.6,
    "orog": 856.0,
    "siconc": 0.35,
    "sst": 11.73,
    "st100": 26.74,
    "st20": 26.91,
    "sm20": 125.99,
    "tp": 43.7,
    "t2m": 21.2692,
    "u1000": 6.09,
    "u850": 8.07,
    "u500": 11.73,
    "u200": 17.76,
    "u100": 12.02,
    "v1000": 5.22,
    "v850": 6.144,
    "v500": 9.03,
    "v200": 12.18,
    "v100": 6.57,
}


def normalize_dataset(dataset):
    for v in dataset.data_vars:
        dataset[v] = (dataset[v] - FIELD_MEAN[v]) / FIELD_STD[v]

    return dataset


def denormalize_dataset(dataset):
    for v in dataset.data_vars:
        dataset[v] = (dataset[v] * FIELD_STD[v]) + FIELD_MEAN[v]

    return dataset


def apply_to_all(transform, example):
    """Utility function to apply a transform on all the kews of an example."""
    new_example = {}
    for k in example:
        new_example[k] = transform(example[k])

    return new_example


class AddBiweeklyDimTransform:
    """Transform that takes a training example and adds the biweekly dimension to it."""

    def __init__(self, weeks_12=False, features=False):
        self.weeks_12 = weeks_12
        self.features = features

    def __call__(self, example):

        to_transform = ["model", "obs"]
        if self.features:
            to_transform.append("features")

        new_example = {}
        for k in example:
            if k in to_transform:
                new_example[k] = add_biweekly_dim(example[k], weeks_12=self.weeks_12)
            else:
                new_example[k] = example[k]

        return new_example


class AddMetadata:
    """Add various metadata to the example dict."""

    def __call__(self, example):
        model = example["terciles"]
        year = int(model.forecast_time.dt.year)
        month = int(model.forecast_time.dt.month)
        day = int(model.forecast_time.dt.day)
        example["monthday"] = f"{month:02}{day:02}"
        example["month"] = f"{month:02}"
        example["year"] = f"{year:04}"

        example["latitude"] = model.latitude
        example["longitude"] = model.longitude

        return example


class AddDryMask:
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def __call__(self, example):
        edges = example["edges"]
        wet_mask = (edges.isel(category_edge=0) > self.threshold).drop("t2m")
        example["dry_mask"] = ~wet_mask
        return example


class ExampleToPytorch:
    def __call__(self, example):
        pytorch_example = {}

        for dataset_name in [
            "obs",
            "model",
            "features",
            "terciles",
            "edges",
            "model_parameters",
            "dry_mask",
            "eccc_parameters",
            "ncep_parameters",
        ]:
            if dataset_name in example:
                dataset = example[dataset_name]
                for variable in dataset.data_vars:
                    new_key = f"{dataset_name}_{variable}"
                    pytorch_example[new_key] = torch.from_numpy(dataset[variable].data)

        for k in ["year", "monthday", "month", "eccc_available", "ncep_available"]:
            pytorch_example[k] = example[k]

        for k in ["latitude", "longitude"]:
            pytorch_example[k] = torch.from_numpy(example[k].data)

        return pytorch_example


class CompositeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        transformed_example = example
        for t in self.transforms:
            transformed_example = t(transformed_example)

        return transformed_example

    def __repr__(self):
        inner_str = ", ".join([repr(t) for t in self.transforms])

        return f"CompositeTransform([{inner_str}])"


def t2m_to_normal(model):
    model_t2m_mean = model.t2m.mean(dim=["lead_time", "realization"]).rename("t2m_mu")
    model_t2m_std = std_estimator(model.t2m, dim=["lead_time", "realization"]).rename(
        "t2m_sigma"
    )

    return xr.merge([model_t2m_mean, model_t2m_std]).rename(
        biweekly_forecast="lead_time"
    )


def tp_to_normal(model):
    model_tp_mean = model.tp.isel(lead_time=-1).mean(dim="realization").rename("tp_mu")
    model_tp_std = std_estimator(model.tp.isel(lead_time=-1), dim="realization").rename(
        "tp_sigma"
    )

    return (
        xr.merge([model_tp_mean, model_tp_std])
        .drop("lead_time")
        .rename(biweekly_forecast="lead_time")
    )


def model_to_distribution(model):
    model_t2m = t2m_to_normal(model)
    model_tp = tp_to_normal(model)

    return xr.merge([model_t2m, model_tp])


class LinearModelAdapter:
    def __init__(self, make_distributions=True):
        self.make_distributions = make_distributions

    def __call__(self, example):
        if self.make_distributions:
            example["model"] = model_to_distribution(example["model"])

        example["obs"] = obs_to_biweekly(example["obs"])

        return example


class CubeRootTP:
    """Apply a cubic root on precipitation data."""

    def __init__(self):
        pass

    def __call__(self, example):
        for k in ["obs_tp", "edges_tp"]:
            if k in example:
                example[k] = example[k] ** (1.0 / 3.0)

        return example


class AddLatLonFeature:
    def __init__(self):
        pass

    def __call__(self, example):
        obs = example["terciles"]
        lat_array = obs["latitude"].assign_coords(variable="lat")
        lat_array = (lat_array / lat_array.max()).astype("float32")

        lon_array = obs["longitude"].assign_coords(variable="lon")
        lon_array = np.sin(np.deg2rad(lon_array)).astype("float32")

        features_array = example["features"].features

        catted_features = xr.concat(
            [features_array, lat_array, lon_array], dim="variable"
        )

        example["features"] = catted_features.to_dataset()

        return example


class AddGeographyFeatures:
    def __init__(self, geography_file):
        geo_dataset = fix_s2s_dataset_dims(xr.open_dataset(geography_file))
        subset = geo_dataset[["orog"]]
        geo = normalize_dataset(subset)
        self.geo_features = geo.to_array().to_dataset(name="features")

    def __call__(self, batch):
        features = batch["features"]

        geo_at_lead = self.geo_features.sel(lead_time=features.lead_time)
        new_features_dataset = xr.concat([features, geo_at_lead], dim="variable")

        batch["features"] = new_features_dataset

        return batch


class RandomNoise:
    def __init__(self, keys=["features_features"], sigma=0.01):
        self.keys = keys
        self.sigma = sigma

    def __call__(self, example):
        for k in self.keys:
            x = example[k]
            example[k] += self.sigma * torch.randn_like(x)

        return example


class LongitudeRoll:
    def __init__(self):
        pass

    def __call__(self, example):
        obs = example["terciles"]
        longitude_length = obs.sizes["longitude"]

        roll = random.randint(0, longitude_length)

        rolled_example = example
        for k in example:
            if k not in ["eccc_available", "ncep_available"]:
                rolled_dataset = (
                    example[k].roll(longitude=roll, roll_coords=True).drop("longitude")
                )

                rolled_example[k] = rolled_dataset

        return rolled_example


class MembersSubsetTransform:
    def __init__(self, subset_size=1):
        self.subset_size = subset_size

    def __call__(self, example):
        features = example["features"]

        n_members = features.sizes["realization"]
        members = sorted(random.choices(range(n_members), k=self.subset_size))
        features = features.isel(realization=members)

        example["features"] = features

        return example


class AddDateFeatureTransform:
    def __call__(self, example):
        features = example["features"]
        date_features = np.sin(
            features.valid_time.assign_coords(variable="date").dt.dayofyear / 366
        )
        new_features = xr.concat(
            [features.features, date_features], dim="variable"
        ).astype("float32")

        example["features"] = new_features.to_dataset()

        return example


class VariableFilterTransform:
    def __init__(self, to_filter=None):
        self.to_filter = to_filter

        if to_filter is not None:
            _logger.info("Will filter vars: %s", to_filter)

    def __call__(self, batch):
        if self.to_filter is not None:
            batch["features"] = batch["features"].sel(variable=self.to_filter)

        return batch


def full_transform(
    geography_file,
    weeks_12=False,
    make_distributions=False,
    random_noise_sigma=0.0,
    roll=False,
    n_members=1,
    filter_vars=None,
    biweekly_features=False,
    add_date=False,
):
    xarray_transforms = [
        MembersSubsetTransform(n_members),
        AddLatLonFeature(),
        AddGeographyFeatures(geography_file),
        VariableFilterTransform(filter_vars),
        AddBiweeklyDimTransform(weeks_12, features=biweekly_features),
    ]

    if add_date:
        xarray_transforms.insert(2, AddDateFeatureTransform())

    if roll:
        xarray_transforms.append(LongitudeRoll())

    transforms = [
        *xarray_transforms,
        # LinearModelAdapter(make_distributions=make_distributions),
        AddMetadata(),
        ExampleToPytorch(),
        CubeRootTP(),
        RandomNoise(sigma=random_noise_sigma),
    ]
    return CompositeTransform(transforms)
