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
    for s in [slice("14D", "27D"), slice("28D", "41D")]:
        weekly_forecast = dataset.sel(lead_time=s)

        first_lead = pd.to_timedelta(weekly_forecast.lead_time[0].item())

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

    if "tp" in with_weekly.data_vars:
        first_forecast, second_forecast = (
            with_weekly.tp.isel(biweekly_forecast=[0]),
            with_weekly.tp.isel(biweekly_forecast=[1]),
        )

        new_first_forecast = first_forecast - dataset.tp.sel(lead_time="13D")
        new_second_forecast = second_forecast - dataset.tp.sel(lead_time="27D")

        new_tp = xr.concat(
            [new_first_forecast, new_second_forecast], dim="biweekly_forecast"
        )

        with_weekly["tp"] = new_tp

    return with_weekly

