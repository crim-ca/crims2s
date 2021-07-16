# S2S Challenge Notebooks Index

This is our collection of notebooks wrote for the [S2S Challenge](https://s2s-ai-challenge.github.io/).

- [download-data.ipybn](./download-data.ipybn). Download all files provided for training by the organizers.
- [data-exploration-2.ipynb](./data-exploration-2.ipybn). Data exploration I made with Jordan to introduce Xarray.
- [simple-linear-model.ipynb](./simple-linear-model.ipynb). Try to make a simple linear model for only one forecast-time/lead-time pair.
- [parametric-distribution.ipynb](./parametric-distribution.ipynb). Tests to use different statistical models on the data, just before making the conversion to 3-class probability forecast.
- [rechunk.ipybn](./rechunk.ipybn). Split the dataset files that have multiple vertical levels into smaller files. 
- [learn-gamma.ipybn](./learn-gamma.ipybn). Use pytorch to learn the parameters of a gamma distribution. This will be useful to model precipitation.