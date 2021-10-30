import hydra
import logging
import multiprocessing
import pathlib
import xarray as xr


_logger = logging.getLogger(__name__)


def split_one_file(input_file, output_dir, levels, overwrite=False):
    f = pathlib.Path(input_file)
    output_path = pathlib.Path(output_dir)

    [center, label, field, datestring] = f.stem.split("-")

    d = xr.open_dataset(f)

    for level in levels:
        level_str = int(level)
        new_filename = f"{center}-{label}-{field}{level_str}-{datestring}.nc"
        new_path = output_path / new_filename

        subset = d.sel(plev=[level])

        if new_path.is_file() and not overwrite:
            _logger.info(f"File {new_path} already exists. Skipping.")
        else:
            _logger.info(f"Writing {new_path}...")
            subset.to_netcdf(new_path, mode="w")


@hydra.main(config_path="conf", config_name="config")
def cli(cfg):
    for field in cfg.fields:
        _logger.info(f"Processing field {field}")
        input_path = pathlib.Path(hydra.utils.to_absolute_path(cfg.set.input_dir))
        input_files = [
            f
            for f in input_path.iterdir()
            if cfg.center in f.stem and f"-{field}-" in f.stem
        ]

        with multiprocessing.Pool(int(cfg.n_workers)) as pool:
            fn_inputs = [
                (
                    f,
                    hydra.utils.to_absolute_path(cfg.set.output_dir),
                    cfg.fields[field],
                    cfg.overwrite,
                )
                for f in input_files
            ]
            pool.starmap(split_one_file, fn_inputs)
