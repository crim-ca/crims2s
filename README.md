# CRIM S2S

Source code supporting the participation of the CRIM S2S team to the WMO S2S Forecast
challenge. [Challenge Website](https://s2s-ai-challenge.github.io/).

## Getting Started

### Notebooks

The notebooks import the `crims2s` package in an absolute manner. That is, they grab
the package from the environment instead of relatively. That means you need to have
the `crims2s` package installed in your environment for the notebooks to work. To
install the package in development mode, go in the repository root and use
```
pip install -e .
```
If you do this from a conda environment, make sure that pip is installed in your
conda environment. Otherwise, this command could install the crims2s package in your
system instead of in your conda environment. To make sure that you use the right pip,
you can type
```
type pip
```
and validate that the path to pip leads to your conda environment.