import dask.array as da


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
