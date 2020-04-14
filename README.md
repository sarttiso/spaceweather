# Geomagnetic Storm Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3751682.svg)](https://doi.org/10.5281/zenodo.3751682)

Notebooks for a project with aim of predicting the response of Earth's external magnetic field to solar activity via ''deep learning''.

Note to self: first ssh @ela.cscs.ch _then_ ssh daint

## Overview

This project is structured primarily in Jupyter Notebooks that present theory alongside application with data. The notebooks and data included are listed and briefly described below.

The data for all of the notebooks is included in this repository, but to be able to run the notebooks themselves, it is most convenient to use the supplied conda environment file ([DL.yml](DL.yml)) to create the corresponding conda environment as described [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

This project also makes use of [GitHub's large file storage (LFS)](https://git-lfs.github.com/) versioning system for data files, which are approximately 360 MB. These data are hdf5 files in the 'data' folder, and you will need git lfs installed to clone these files.

### Notebooks
- [forecasting_Dst_LSTM](forecasting_Dst_LSTM.ipynb) : This notebook features a vanilla LSTM forecasting scheme that achieves state of the art results in multiple hour ahead forecasting of Dst, which has been a primary focus of the community.
- [forecasting_Est_LSTM](forecasting_Est_LSTM.ipynb) : This notebook presents forecasting of Est, which corresponds to the entirely external component of Dst.
- [forecasting_Est_prob_NN](forecasting_Est_prob_NN.ipynb) : This notebook contains Est forecasting using probabilistic neural networks. The techniques explored include modeling network weights probabilistically, modeling network outputs probabilistically, and mixing-and-matching these approaches with their deterministic counterparts.
- [burtons_model](burtons_model.ipynb) : This notebook implements the simple empirical model of [Burton et al. 1975](https://doi.org/10.1029/JA080i031p04204), which models Dst evolution via a first order equation in time depending on various solar wind parameters.
- [solar_wind_corona_storms](solar_wind_corona_storms.ipynb) : This notebook provides visualizations of various storms, solar wind observations, and solar disk observations.
- [synthetic_geomagnetic_storms](synthetic_geomagnetic_storms.ipynb) : This notebook implements a toy model of geomagnetic storms in which solar activity is simulated as a series of impulses representing potentially geoeffective mass ejections that in turn generate geomagnetic storms. The goal of this notebook is to explore architectures capable of learning from sparse, impulse-like input time series, which is the basic task necessary for forecasting geomagnetic storms from observations of the solar disk.
- [external_coefficient_statistics](external_coefficient_statistics.ipynb) : This notebook computes statistics for Est and assesses reasonable marginal distributions over Est.
- [manuscript_figures](manuscript_figures.ipynb) : This notebook generates figures for a manuscript submitted to Geophysical Research Letters.


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
- [external_coefficients](external_coefficients.ipynb) : Coefficients of Earth's external magnetic field on a spherical harmonic basis.
- [cme_hourly_complete](cme_hourly_complete.h5) : [SOHO LASCO CME database](https://cdaw.gsfc.nasa.gov/CMElist/) at hourly sampling. For hours with multiple CMEs, I took the data from the most energetic one.
- [GOES_flare_locations_hourly_1975-2016](GOES_flare_locations_hourly_1975-2016.h5) : GOES flare location database at hourly sampling. For hours with multiple flares, I took the the data from the most central flare on the solar disk for the reason that solar activity near the center of the solar disk is often associated with geoeffective storms.
- [dst_est_ist](dst_est_ist.h5) : Decomposition of Dst into Est and Ist as computed and provided by [NOAA](https://www.ngdc.noaa.gov/geomag/est_ist.shtml).
