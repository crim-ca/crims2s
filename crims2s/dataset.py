"""PyTorch dataset for S2S data."""

import pathlib
from typing import Iterable, Union
import torch
import xarray as xr


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


class S2SDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        name_filter=None,
        include_features=True,
        years: Union[Iterable[int], None] = None,
    ):
        """Args:
          dataset_dir: Directory where the ml ready dataset files are.
          name_filter: Include only files for which name_filter(filename) is True.
          include_features: Do not load the features to memory, only the rest of the data.
          years: If provided, only include files which have one of the specified years"""
        dataset_path = pathlib.Path(dataset_dir)

        files = []
        for f in dataset_path.iterdir():
            if f.name.endswith(".nc"):
                year_of_filename = int(f.stem.split("_")[2][:4])

                if (name_filter is None or name_filter(f.name)) and (
                    not years or year_of_filename in years
                ):
                    files.append(f)
        self.files = sorted(files)

        self.include_features = include_features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]

        groups_to_read = ["/obs", "/model", "/terciles", "/edges", "/model_parameters"]

        if self.include_features:
            groups_to_read.append("/features")

        example = {k[1:]: xr.open_dataset(f, group=k) for k in groups_to_read}

        return example
