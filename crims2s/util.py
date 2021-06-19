def fix_dataset_dims(d):
    """Given one of the dataset files given by the organizers, fix its
    dimensions so its easier to concatenate and use with xr.open_mfdataset.

    Arguments:
      d. xr.Dataset. The dataset you get when you open one of the provided files.
    """

    day_of_year = d.forecast_time[0].dt.dayofyear.data.item()

    new_d = d.expand_dims("dayofyear").assign_coords(dayofyear=[day_of_year])
    new_d = new_d.assign_coords(year=new_d.forecast_time.dt.year).swap_dims(
        forecast_time="year"
    )
    return new_d
