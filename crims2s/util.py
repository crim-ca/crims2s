import numpy as np
import pandas as pd
import xarray as xr

# Month day list of the forecasts ECMWF does each year.
ECMWF_FORECASTS = [
    (1, 2),
    (1, 9),
    (1, 16),
    (1, 23),
    (1, 30),
    (2, 6),
    (2, 13),
    (2, 20),
    (2, 27),
    (3, 5),
    (3, 12),
    (3, 19),
    (3, 26),
    (4, 2),
    (4, 9),
    (4, 16),
    (4, 23),
    (4, 30),
    (5, 7),
    (5, 14),
    (5, 21),
    (5, 28),
    (6, 4),
    (6, 11),
    (6, 18),
    (6, 25),
    (7, 2),
    (7, 9),
    (7, 16),
    (7, 23),
    (7, 30),
    (8, 6),
    (8, 13),
    (8, 20),
    (8, 27),
    (9, 3),
    (9, 10),
    (9, 17),
    (9, 24),
    (10, 1),
    (10, 8),
    (10, 15),
    (10, 22),
    (10, 29),
    (11, 5),
    (11, 12),
    (11, 19),
    (11, 26),
    (12, 3),
    (12, 10),
    (12, 17),
    (12, 24),
    (12, 31),
]

# The last date at which we can use the obs for training.
TEST_THRESHOLD = "2020-01-01"


def fix_dataset_dims(d):
    """Given one of the dataset files given by the organizers, fix its
    dimensions so its easier to concatenate and use with xr.open_mfdataset.

    Arguments:
      d. xr.Dataset. The dataset you get when you open one of the provided files.
    """

    month = int(d.forecast_time[0].dt.month)
    day = int(d.forecast_time[0].dt.day)
    label = f"{month:02}{day:02}"

    new_d = d.expand_dims("forecast_monthday").assign_coords(
        forecast_monthday=xr.DataArray([label], dims="forecast_monthday")
    )
    new_d = new_d.assign_coords(forecast_year=new_d.forecast_time.dt.year).swap_dims(
        forecast_time="forecast_year"
    )

    # Reorder the dimensions to something that is more intuitive (according to me).
    dims = set(new_d.dims)
    dims.difference_update(
        ("forecast_monthday", "forecast_year", "latitude", "longitude")
    )

    new_d = new_d.transpose(
        "forecast_year", "forecast_monthday", *dims, "latitude", "longitude"
    )

    # new_d = new_d.chunk(chunks="auto")

    return new_d


def add_biweekly_dim(dataset):
    """From a dataset with a lead time, add a dimension so that there is one
    dimension for which biweekly forecast we're in, and one dimension for the lead time
    whithin that biweekly forecast."""
    weeklys = []
    for s in [slice("0D", "13D"), slice("14D", "27D"), slice("28D", "41D")]:
        weekly_forecast = dataset.sel(lead_time=s)

        first_lead = pd.to_timedelta(s.start)

        weekly_forecast = weekly_forecast.expand_dims(
            dim="biweekly_forecast"
        ).assign_coords(biweekly_forecast=[first_lead])
        weekly_forecast = weekly_forecast.assign_coords(
            lead_time=(weekly_forecast.lead_time - first_lead)
        )
        weeklys.append(weekly_forecast)

    with_weekly = xr.concat(weeklys, dim="biweekly_forecast").transpose(
        "forecast_year", "forecast_monthday", "biweekly_forecast", ...
    )

    # Fix the validity time for the first step (which we don't have any data for).
    with_weekly["valid_time"] = (
        with_weekly.forecast_time
        + with_weekly.biweekly_forecast
        + with_weekly.lead_time
    )

    if "tp" in with_weekly.data_vars:
        # For day zero we can infer the value of TP (which is zero everywhere).
        with_weekly.tp.loc[
            dict(biweekly_forecast=pd.to_timedelta(0), lead_time=pd.to_timedelta(0))
        ] = 0.0

        first_week, second_week, third_week = (
            with_weekly.tp.isel(biweekly_forecast=[0]),
            with_weekly.tp.isel(biweekly_forecast=[1]),
            with_weekly.tp.isel(biweekly_forecast=[2]),
        )

        new_second_week = second_week - dataset.tp.sel(lead_time="13D")
        new_third_week = third_week - dataset.tp.sel(lead_time="27D")

        new_tp = xr.concat(
            [first_week.drop("valid_time"), new_second_week, new_third_week],
            dim="biweekly_forecast",
        )

        with_weekly["tp"] = new_tp.clip(
            min=0.0
        )  # See recommendation https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge/-/issues/38.

    return with_weekly


def std_estimator(dataset, dim=None):
    """Estimator for the sigma parameter of a normal distribution. It is different from 
    calling std directly because it uses n - 1 on the denominator instead of n."""
    dataset_mean = dataset.mean(dim=dim)

    if dim is None:
        dim_sizes = [dataset.sizes[x] for x in dataset_mean.dims]
    elif isinstance(dim, str):
        dim_sizes = dataset.sizes[dim]
    else:
        dim_sizes = [dataset.sizes[x] for x in dim]

    n = np.prod(dim_sizes)

    return xr.ufuncs.sqrt(
        xr.ufuncs.square(dataset - dataset_mean).sum(dim=dim) / (n - 1)
    )


def obs_to_biweekly(obs):
    """Given an xarray Dataset that contains observations, aggregate them into a
    biweekly format."""
    aggregate_obs_tp = obs.pr.sum(dim="lead_time", min_count=2).rename("tp")
    aggregate_obs_t2m = obs.t2m.mean(dim="lead_time")
    return xr.merge([aggregate_obs_tp, aggregate_obs_t2m])
