"""Various constants and distributions that decribe our dataset. Intended use
is normalization of the fields before sending them to a neural net.

See notebook distributions-of-parameters.ipynb"""

import numpy as np
import torch
import xarray as xr

from .util import add_biweekly_dim, obs_to_biweekly, std_estimator

FIELD_MEAN = {
    "gh10": 30583.0,
    "gh100": 16070.0,
    "gh1000": 76.19,
    "gh200": 11765.0,
    "gh500": 5524.374,
    "gh850": 1403.0,
    "lsm": 0.0,
    "msl": 100969.28,
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

    def __init__(self, weeks_12=False):
        self.weeks_12 = weeks_12

    def __call__(self, example):
        new_example = {}
        for k in example:
            if k in ["model", "obs"]:
                new_example[k] = add_biweekly_dim(example[k], weeks_12=self.weeks_12)
            else:
                new_example[k] = example[k]

        return new_example


class AddMetadata:
    """Add various metadata to the example dict."""

    def __call__(self, example):
        model = example["model"]
        month = int(model.forecast_time.dt.month)
        day = int(model.forecast_time.dt.day)
        example["monthday"] = f"{month:02}{day:02}"

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
        ]:
            if dataset_name in example:
                dataset = example[dataset_name]
                for variable in dataset.data_vars:
                    pytorch_example[f"{dataset_name}_{variable}"] = torch.from_numpy(
                        dataset[variable].data
                    )

        for k in ["monthday"]:
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
            example[k] = example[k] ** (1.0 / 3.0)

        return example


class AddLatLonFeature:
    def __init__(self):
        pass

    def __call__(self, example):
        features = example["features_features"]
        lat = example["latitude"]
        lon = example["longitude"]

        lat_feature = torch.zeros(121, 240)
        lat_feature[:, :] = (lat / lat.max()).unsqueeze(dim=-1)

        lon_feature = torch.zeros(121, 240)
        lon_feature[:] = torch.sin((lon / 360.0) * 2.0 * np.pi)

        lat_lon_features = torch.stack([lat_feature, lon_feature], dim=-1).unsqueeze(-2)

        lat_lon_features_weekly = torch.zeros_like(features[..., :2])
        lat_lon_features_weekly[:, :] = lat_lon_features

        new_feature = torch.cat([features, lat_lon_features_weekly], dim=-1)
        example["features_features"] = new_feature

        return example
