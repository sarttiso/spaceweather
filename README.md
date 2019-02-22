## Geomagnetic Storm Prediction

Notebooks for a project with aim of predicting the response of Earth's external magnetic field to solar activity via ''deep learning''.

Goals/To do:

-   [x] First zonal harmonic prediction with basic LSTM network,  [forecasting_q10_LSTM](forecasting_q10_LSTM.ipynb)
-   [ ] Dst prediction with basic LSTM network (to compare with previous results), [forecasting_Dst_LSTM](forecasting_Dst_LSTM.ipynb)
-   [ ] Implement Garnelo et al.'s Bayesian neural network architecture and apply to first zonal harmonic prediction
-   [ ] Train networks only on storm time series
-   [ ] Consider subsets of input OMNI data to train on and compare effectiveness of corresponding networks
-   [ ] Interrogate neurons of trained networks to elucidate their sensitivities to particular inputs and thereby try to uncover whether or not a physical model was approximated by the network (e.g. did the network construct any terms analogous to those outlined by Burton et al. 1975?)

Note to self: first ssh @ela.cscs.ch *then* ssh daint
