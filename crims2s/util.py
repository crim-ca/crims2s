import numpy as np
import pandas as pd
import xarray as xr


def fix_dataset_dims(d):
    """Given one of the dataset files given by the organizers, fix its
    dimensions so its easier to concatenate and use with xr.open_mfdataset.

    Arguments:
      d. xr.Dataset. The dataset you get when you open one of the provided files.
    """

    day_of_year = d.forecast_time[0].dt.dayofyear.data.item()

    new_d = d.expand_dims("forecast_dayofyear").assign_coords(
        forecast_dayofyear=[day_of_year]
    )
    new_d = new_d.assign_coords(forecast_year=new_d.forecast_time.dt.year).swap_dims(
        forecast_time="forecast_year"
    )

    # Reorder the dimensions to something that is more intuitive (according to me).
    dims = set(new_d.dims)
    dims.difference_update(
        ("forecast_dayofyear", "forecast_year", "latitude", "longitude")
    )

    new_d = new_d.transpose(
        "forecast_year", "forecast_dayofyear", *dims, "latitude", "longitude"
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
        "forecast_year", "forecast_dayofyear", "biweekly_forecast", ...
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

        # with_weekly["tp"] = new_tp.clip(
        ##    min=0.0
        # )  # Clip is temporary until we figure out why tp isn't monotonous in ECMWF dataset.

        with_weekly["tp"] = new_tp

    return with_weekly

