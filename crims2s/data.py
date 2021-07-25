"""Various constants and distributions that decribe our dataset. Intended use
is normalization of the fields before sending them to a neural net.

See notebook distributions-of-parameters.ipynb"""

FIELD_MEAN = {
    "gh10": 30583.0,
    "gh100": 16070.0,
    "gh1000": 76.19,
    "gh200": 11765.0,
    "gh500": 5524.374,
    "gh850": 1403.0,
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

def denormalize_Dataset(dataset):
    for v in dataset.data_vars:
        dataset[v] = (dataset[v] * FIELD_STD[v]) + FIELD_MEAN[v]

    return dataset