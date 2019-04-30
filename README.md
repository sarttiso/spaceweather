# Geomagnetic Storm Forecasting

Notebooks for a project with aim of predicting the response of Earth's external magnetic field to solar activity via ''deep learning''.

Note to self: first ssh @ela.cscs.ch _then_ ssh daint

## Overview

This project is structured primarily in Jupyter Notebooks that present theory alongside application with data. The notebooks and data included are listed and briefly described below.

The data for all of the notebooks is included in this repository, but to be able to run the notebooks themselves, it is most convenient to use the supplied conda environment file ([DL.yml](DL.yml)) to create the corresponding conda environment as described [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Notebooks
-   [forecasting_Dst_LSTM](forecasting_Dst_LSTM.ipynb) : This notebook features a vanilla LSTM forecasting scheme that achieves state of the art results in multiple hour ahead forecasting of Dst, which has been a primary focus of the community.
-   [forecasting_Est_LSTM](forecasting_Est_LSTM.ipynb) : This notebook presents forecasting of Est, which corresponds to the entirely external component of Dst.
-  [forecasting_Est_BNN](forecasting_Est_BNN.ipynb) : This notebook contains external coefficient forecasting using probabilistic neural networks. The techniques explored include modeling network weights probabilistically, modeling network outputs probabilistically, and mixing-and-matching these approaches with their deterministic counterparts.
- [burtons_model](burtons_model.ipynb) :
- [solar_wind](solar_wind.ipynb) :


### Python modules

The following files contain useful functions and classes used throughout the notebooks.

- [helper_functions](helper_functions.py) :
- [aux_keras](aux_keras.py) :
- [bdl_tensorflow](bdl_tensorflow.py) :


### Data

Data are organized by files containing different datasets organized roughly by source.

- [omni_hourly_1998-2018_nogaps](omni_hourly_1998-2018_nogaps.h5) :
- [omni_hourly_1998-2018](omni_hourly_1998-2018_nogaps) :
- [NS41_GPS_hourly](NS41_GPS_hourly.h5) :
- [external_coefficients](external_coefficients.ipynb) :


## Goals/To do:

-   [x] First zonal harmonic prediction with basic LSTM network,  [forecasting_Est_LSTM](forecasting_Est_LSTM.ipynb)
-   [x] Dst prediction with basic LSTM network (to compare with previous results), [forecasting_Dst_LSTM](forecasting_Dst_LSTM.ipynb)
-   [x] Implement Bayes-by-Backprop for modeling weight uncertainty in recurrent neural networks
-   [x] Train networks only on storm time series :arrow_right: not much of a sensitivity to this actually
-   [x] Consider subsets of input OMNI data to train on and compare effectiveness of corresponding networks :arrow_right: SW parameters are by far the most informative (IMF, speed, particle density, and temperature)
-   [ ] Update [forecasting_Est_LSTM](forecasting_Est_LSTM.ipynb) to reflect prediction of Est vs the external coefficient provided by Alexander (possibly just keep Alexander's data, but change notation)
-   [ ] Implement Garnelo et al.'s Bayesian neural network architecture and apply to first zonal harmonic prediction
