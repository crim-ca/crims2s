# CRIM S2S

Source code supporting the participation of the CRIM S2S team to the [WMO S2S Forecast
challenge](https://s2s-ai-challenge.github.io/).

Our source code lives in a distinct git repository because of the size of the 
submitted code. We copied the safeguards/checklists we could find in this README
to have the same documentation as if we had submitted 

## Safeguards

This section was copied from the [submission template](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/blob/master/notebooks/ML_forecast_template.ipynb).

### Ressources used

* Data preparation done over a SLURM cluster.
* Training performed on GPUs, most often a GTX 1080 Ti. No multi-GPU setup ended up 
being used in our final pipeline.

### Data used

* Multiple fields from the ECMWF hindcasts/forecasts. See the `crims2s/conf/fields/both_easy.yaml` file for a complete list.
* `t2m` and `tp` from NCEP and ECCC.
* `orog` for ECMWF, which was not downloaded through renku, but we provide the file in our repo.


### Safeguards to prevent overfitting
If the organizers suspect overfitting, your contribution can be disqualified.
* [x] We did not use 2020 observations in training (explicit overfitting and cheating)
* [x] We did not repeatedly verify my model on 2020 observations and incrementally improved my RPSS (implicit overfitting)
* [ ] We provide RPSS scores for the training period with script skill_by_year, see in section 6.3 predict.
* [x] We tried our best to prevent data leakage.
* [x] We honor the train-validate-test split principle. This means that the hindcast data is split into train and validate, whereas test is withheld.
    - In the final push to improve our results, we use the validation set for training.
    That is, we ran the training once with a validation set to determine all the hyperparameters. Then we fixed the hyperparameters and ran the training again using
    the training set + the validation set. The test set is in a separate folder and
    was never loaded for training purposes.
* [x] We did not use test explicitly in training or implicitly in incrementally adjusting parameters.
* [x] We considered cross-validation.
    - Our validation strategy was not precisely a cross validation. Since we use 
    data from all three centers (ECMWF, ECCC and NCEP), we used years 2010, 2017 and 2019 
    as validation years. That way, we have at least one validation year where the 
    data is available for each center.

### Safeguards for Reproducibility
Notebook/code must be independently reproducible from scratch by the organizers (after the competition), if not possible: no prize
* [x] All training data is publicly available (no pre-trained private neural networks, as they are not reproducible for us)
    - All the data was downloaded from renku, except the orography which we provide in the repo.
* [x] Code is well documented, readable and reproducible.
* [x] Code to reproduce training and predictions is preferred to run within a day on the described architecture. If the training takes longer than a day, please justify why this is needed. Please do not submit training piplelines, which take weeks to train.
    - Our models comprises deep neural architecture which typically train over many days 
    depending on the GPU power used. In our case, the training of parts of the models 
    lasts about two days on a GTX 1080 ti. There training times are not out of the ordinary
    for a convolutional neural network in our experience.
    - The data preparation step lasts about an hour for us, but is is done using about 53 
    jobs in parallel on a computing cluster. That means that it could take more than 24hrs 
    if only consumer-grade equipment is used. The usage of so many data is justified by
    common principles in deep machine learning where data volumes typically improve 
    model performance.

## Model architecture

Our model is an opportunistic mixture model. It is a blend
of: 
* EMOS corrected forecasts from ECMWF, ECCC and NCEP
* CNN corrected forecast for ECMWF
* Climatology

A second CNN decides the relative weights. Then, a weighted
average of the 5 models is performed. The schematic below summarizes 
the model.

![Model Schematic](./s2s-model.png)

We call our model opportunistic because the weighting model has 
to detect where there is predictability and use the forecasts there.
It has to detect where the is no opportunity for predictability and 
use climatology in these circumstances.

The EMOS models only use the predicted parameter as a predictor. That is, is we want to correct
the mean of the ECCC `t2m` variable, we only use the mean of the ECCC `t2m` as a predictor variable. Our EMOS models are used to correct 4 parameters: `t2m` mean, `t2m` standard deviation, `tp` mean and `tp` standard deviation. To make the precipitation data
look a little more like normally distributed data, we apply a cubic root on the 
precipitation values before we correct them with EMOS.

The convolutionnal models use 18 features as predictors. Most of the predictors
are field from the ECMWF hindcast/forecast. The list of these variable is available
in the `crims2s/conf/fields/both_easy.yaml` file. Some of the predictors used by the
convolutional models are constant throughout all the training examples. These
predictors are
* latitude and longitude
* orog (downloaded [here](https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/index.html?Set-Language=fr)) and available in the root of our repo.
* day of year.

The optimization is done by minimizing the RPS for each predicted variable. We use
the Adam optimizer throughout, although the hyperparameters vary if we are 
training an EMOS or a convolutional model.

Countless other decisions were taken when designing this pipeline. Hopefully they
can be better communicated in the future. In the meantime, the source code in this
repository consists in a rough desription of these decisions.

## Reproducing

Here are the instructions to reproduce our results. The instructions
are unfortunately longer than we would have hoped. If any problem is
encountered during reproduction please feel free to reach out. We 
also have the possibility to provide intermediate datasets. For
instance if it is impossible to regenerate an ml-ready dataset using our
code, we could provide one.

### Package installation

This guide assumes that a conda installation is already available.

```bash
conda create -n s2s_repro -f environment.yml
conda activate s2s_repro
pip install -e <path_to_our_repo>
```

### Data preparation

#### Split the multi-level files into smaller chunks

The NetCDF files that contain multiple vertical levels were a little bit difficult 
to manipulate on our end. For this reason, our code assumes that there is a version
of these files that has been split into one netcdf by vertical level. We wrote
a small utility to do this. The splitting is only necessary for some fields from the
ECMWF model. The list of fields that should be split is configured in the 
`crims2s/splitter/conf/config.yaml`.

```
s2s_split_plev base_dir=<path_to_s2s_data> set=train
s2s_split_plev base_dir=<path_to_s2s_data> set=test
```

Here, `<path_to_s2s_data>` should contain the data pulled from climetlab, so
for instance `<path_to_s2s_data>/training-input` should exist and contain the 
necessary model files.

When this script is done, there should be new files named `ecmwf-hindcast-u200-20200102.nc` and so on.


#### Make an ML-Ready dataset

The next step is to rehash the data so that it can be batched easily. This is done
using the `s2s_mldataset` script. This script generates one file per forecast.

The `s2s_mldataset` program takes the `base_dir` parameter as input. In `base_dir`
it should be able to find the 
* `training-input/`
* `test-input/`
* `hindcast-like-observations_2000-2019_biweekly_tercile-edges.nc`
* `hindcast-like_observations_2000-2019_biweekly_deterministic.nc`
* `hindcast-like_observations_2000-2019_biweekly_terciled.nc`    
* `forecast-like_observations_2020_biweekly_deterministic.nc`
* `forecast-like_observations_2020_biweekly_terciled.nc`     


To generate the zeroth example of year 2000, run
```
s2s_mldataset base_dir=<base_dir> output_dir=<dataset_dir> index=0 set.year=2000
```

To generate an example for the third forecast of every year, run
```
s2s_mldataset base_dir=<base_dir> output_dir=<dataset_dir> index=2
```

To generate the whole training set, run
```
s2s_mldataset base_dir=<base_dir> output_dir=<dataset_dir> index="range(0,53)" -m
```
Keep in mind that this command was usually run in parallel on our cluster. This
might take a while.

To generate the test set, run
```
s2s_mldataset base_dir=<base_dir> output_dir=<test_dataset_dir> index="range(0,53)" set=test -m
```


### Model training

The overall procedure is to 
1. Train EMOS models for ECMWF, NCEP and ECCC
2. Train a convolutionnal post-processing model for ECMWF
3. Use checkpoints from the above models to initialize an ensemble model. Train the 
convolutional weight model that glues all the above models together.

#### Training EMOS models

Every time an EMOS model is trained, the logger prints the current working
directory. Remember what that directory is. It will be useful we we want to load
the checkpoints to initialize an ensemble model later.

To train ECMWF EMOS, run
```
s2s_train experiment=emos experiment.dataset.dataset_dir=<dataset_dir>
```

To train ECCC EMOS, run
```
s2s_train experiment=emos_eccc experiment.dataset.dataset_dir=<dataset_dir>
```

To train NCEP EMOS, run
```
s2s_train experiment=emos_ncep experiment.dataset.dataset_dir=<dataset_dir>
```


#### Training the convolutional post-processing model

Run
```
s2s_train experiment=conv_JG experiment.dataset.dataset_dir=<dataset_dir>
```


#### Training the convolutional ensemble model

Using the working directories that we noted when training the 4 models, we can
now launch the training of a convolutionnal ensemble model.

To do so, first edit the `crims2s/training/conf/experiment/models/bayes_multi_jg.yaml`
file. Replace all the `checkpoint_dir` keys with the appropriate working directory.
Note that you don't have to specify precisely where the checkpoint is. Only provide
the working directory and the checkpoint will be found automatically.

Once the checkpoints are specified, run
```
s2s_train experiment=bayes_multi_jg experiment.dataset.dataset_dir=<dataset_dir>
```



### Validation

Once a model is trained, you can run it on the 2020 data for verification purposes.
To do so, run

```
s2s_infer checkpoint_dir=<ensemble_model_checkpoint_dir> output_file=model_output.nc test_dataset_dir=<test_dataset_dir>
```
The script will let you know where your output file has been stored.