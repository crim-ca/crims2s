"""PyTorch dataset for S2S data."""

import pathlib
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
    def __init__(self, dataset_dir, filter_str=None, include_features=True):
        dataset_path = pathlib.Path(dataset_dir)
        self.files = [
            x
            for x in dataset_path.iterdir()
            if filter_str is None or filter_str in x.name
        ]

        self.include_features = include_features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        obs = xr.open_dataset(f, group="/obs")
        model = xr.open_dataset(f, group="/model")
        target = xr.open_dataset(f, group="/y")

        example = {"obs": obs, "model": model, "target": target}

        if self.include_features:
            example["features"] = xr.open_dataset(f, group="/x")

        return example

