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
    "sst": 286.96,
    "t2m": 278.2237,
    "tp": 34.1,
}

FIELD_STD = {
    "gh10": 993.0,
    "gh100": 577.0,
    "gh1000": 110.14,
    "gh200": 605.0,
    "gh500": 341.80862,
    "gh850": 149.6,
    "sst": 11.73,
    "tp": 43.7,
    "t2m": 21.2692,
}

