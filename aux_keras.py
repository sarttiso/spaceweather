# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:29:18 2019

Library with helper functions and classes for deep learning with Keras.

@author: Adrian
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

"""
Given a batch size and training data, train network on the data, and
return the trained network

IN:
dat_in_train (np.ndarray): data inputs with shape (ndata, 1, nfeat_in)
dat_out_train (np.ndarray): data outputs with shape (ndata, nfeat_out)
batch_size (int/double):
nunits (int/double): (default 400) number of hidden units in LSTM
epochs (int): (default 100) epochs for training

OUT:
rnn (keras.Sequential): trained network
hist ()
"""
def train_network(dat_in_train, dat_out_train, batch_size, nunits=7, \
                  epochs=100, stateful=True):
    assert len(dat_in_train.shape) == 3, 'check shape of dat_in_train'
    assert len(dat_out_train.shape) == 2, 'check shape of dat_out_train'

    nfeat_in = dat_in_train.shape[2]
    nfeat_out = dat_out_train.shape[1]

    # recurrent architecture, create input and output datasets
    rnn = Sequential()

    rnn.add(LSTM(nunits,
            name='LSTM_1',
            stateful=True,
            input_shape=(1,nfeat_in),
            batch_size=batch_size,
            return_sequences=False,
            activation='relu'))
    rnn.add(Dense(nfeat_out,
                  name='Dense'))
    opt = keras.optimizers.RMSprop(lr=0.001)
    rnn.compile(loss='mse',optimizer=opt)

    # fit model
    if stateful:
        hist = np.zeros(epochs)
        for ii in range(epochs):
            rnn.reset_states()
            tmp = rnn.fit(dat_in_train,
                dat_out_train,
                epochs=1,
                batch_size=batch_size,
                shuffle=False,
                verbose=0)
            hist[ii] = tmp.history['loss'][0]
            print('Epoch %d, MSE %1.2e' % (ii, hist[ii]))
    else:
        hist = rnn.fit(dat_in_train,
            dat_out_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1)
        hist = hist.history['loss']
    return rnn, hist


"""
Given a trained network, evaluated it on testing data and plot the
scatter plots.

IN:
dat_in_test (np.ndarray): data inputs with shape (ndata, 1, nfeat_in)
dat_out_test (np.ndarray): data outputs with shape (ndata, nfeat_out)
rnn (keras.Sequential): trained network
batch_size (int/double):
feature_names (list): list of names of output features, length nfeat_out

OUT:
produces plot

"""
def test_network(dat_in_test, dat_out_test, rnn, batch_size, feature_names, \
                 scaler_output):
    assert len(dat_in_test.shape) == 3, 'check shape of dat_in_test'
    assert len(dat_out_test.shape) == 2, 'check shape of dat_out_test'
    rnn.reset_states()
    dat_pred = rnn.predict(dat_in_test,batch_size=batch_size)
    plt.figure()
    x = scaler_output.inverse_transform(dat_out_test)
    y = scaler_output.inverse_transform(dat_pred)
    plotcorr(x, y, title=feature_names)
    return dat_pred
