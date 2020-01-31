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
- [burtons_model](burtons_model.ipynb) : This notebook implements the simple empirical model of [Burton et al. 1975](https://doi.org/10.1029/JA080i031p04204), which models Dst evolution via a first order equation in time depending on various solar wind parameters.
- [solar_wind_corona_storms](solar_wind_corona_storms.ipynb) : This notebook provides visualizations of various storms, solar wind observations, and solar disk observations.
- [synthetic_geomagnetic_storms](synthetic_geomagnetic_storms.ipynb) : This notebook implements a toy model of geomagnetic storms in which solar activity is simulated as a series of impulses representing potentially geoeffective mass ejections that in turn generate geomagnetic storms. The goal of this notebook is to explore architectures capable of learning from sparse, impulse-like input time series, which is the basic task necessary for forecasting geomagnetic storms from observations of the solar disk.


### Python modules

The following files contain useful functions and classes used throughout the notebooks.

- [helper_functions](helper_functions.py) : Various functions ranging from reliability curve computation to plotting.
- [aux_keras](aux_keras.py) : Functions used by the forecasting_Dst_LSTM notebook.
- [bdl_tensorflow](bdl_tensorflow.py) : Contains implementations of variational Dense and LSTM layers, along with a class that defines a mixed Gaussian prior.


### Data

Data are organized by files containing different datasets organized roughly by source.

- [omni_hourly_alldata_smallfilled](omni_hourly_alldata_smallfilled.h5) : All measurements included in the OMNI low-res dataset without flags and uncertainties. Gaps of 72 hours or less have been filled via linear interpolation. This dataset contains all OMNI low res data with no-data values replaced by np.nans.
- [NS41_GPS_hourly](NS41_GPS_hourly.h5) : Magnetic field measurements from GPS satellite NS41, used by Gruet et al. 2018 to achieve improved forecasting.
- [GEOS_xrs_hourly_1986-2018_nogaps](GEOS_xrs_hourly_1986-2018_nogaps.h5) : Short and long channel x-ray flux measurements gathered by the [GOES mission](https://www.ngdc.noaa.gov/stp/satellite/goes/) satellites from 1986-2018. More data is available, but the specification of primary and secondary satellites since 1986 facilitated downloading this dataset.
- [external_coefficients](external_coefficients.ipynb) : 
- [cme_hourly_complete](cme_hourly_complete.h5) :


## Goals/To do:

-   [x] First zonal harmonic prediction with basic LSTM network,  [forecasting_Est_LSTM](forecasting_Est_LSTM.ipynb)
-   [x] Dst prediction with basic LSTM network (to compare with previous results), [forecasting_Dst_LSTM](forecasting_Dst_LSTM.ipynb)
-   [x] Implement Bayes-by-Backprop for modeling weight uncertainty in recurrent neural networks
-   [x] Train networks only on storm time series :arrow_right: not much of a sensitivity to this actually
-   [x] Consider subsets of input OMNI data to train on and compare effectiveness of corresponding networks :arrow_right: SW parameters are by far the most informative (IMF, speed, particle density, and temperature)
-   [ ] Update [forecasting_Est_LSTM](forecasting_Est_LSTM.ipynb) to reflect prediction of Est vs the external coefficient provided by Alexander (possibly just keep Alexander's data, but change notation)
-   [ ] Implement Garnelo et al.'s Bayesian neural network architecture and apply to first zonal harmonic prediction
