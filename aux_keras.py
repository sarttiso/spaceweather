# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:29:18 2019

Library with helper functions and classes for deep learning with Keras.

@author: Adrian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout


"""
From a data array with observations in rows and features in columns, split 
split into testing and training data based on which rows to use (datain_idx), 
batch_sizes, which features for prediction (incols) and which features as 
targest (outcols). Since we're concerned with time series forecasting, the 
number of steps to forecast ahead is lahead.

IN:
data_in: 
data_out: 
batch_size:
train_percent:
lahead:

OUT:
data_in_train:
data_out_train:
data_in_test:
data_out_test:

 """
def datasplit(data_in, data_out, batch_size, train_percent=0.8, lahead=1, \
              stateful=True):
    
    data_in = data_in[0:-lahead,:]
    data_out = data_out[lahead:,:]
    
    ndat = data_in.shape[0]
    
    if stateful:
        # limit for training data
        lidx = int((train_percent*ndat)-((train_percent*ndat) % batch_size))
        # limit for testing data
        ridx = int(ndat - (ndat % batch_size))
    
        trainidx = np.zeros(ndat, dtype=bool)
        trainidx[0:lidx] = True
        testidx = np.zeros(ndat, dtype=bool)
        testidx[lidx:ridx] = True
    else:
        nbatch = int(np.floor(ndat/batch_size))
    
        # now construct training and testing sets
        batchrand = np.random.permutation(int(nbatch))
        trainbatch = batchrand[0:int(train_percent*nbatch)]
        testbatch = batchrand[int(train_percent*nbatch):]
        
        trainidx = np.zeros(ndat, dtype=bool)
        for batchidx in trainbatch:
            trainidx[batchidx*batch_size:batch_size*(batchidx + 1)] = True
            
        testidx = np.zeros(ndat, dtype=bool)
        for batchidx in testbatch:
            testidx[batchidx*batch_size:batch_size*(batchidx + 1)] = True
    
    data_in_train = data_in[trainidx,:]
    data_out_train = data_out[trainidx,:]
    data_in_test = data_in[testidx,:]
    data_out_test = data_out[testidx,:]
    return data_in_train, data_out_train, data_in_test, data_out_test

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
    curtest = np.concatenate((scaler_output.inverse_transform(dat_out_test),\
                              scaler_output.inverse_transform(dat_pred)), axis=1)
    plotcorr(curtest, title=feature_names)
    return dat_pred

"""
Plot straight line in current axis.
"""
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'k--')

 
"""
Plot scatter plot with 1:1 line and point density contours.
IN:
data (nd.array): ndata x 2 array
"""
def plotcorr(data, title=''):
    plt.scatter(data[:,0], data[:,1], marker='.')
    k = kde.gaussian_kde(data.T)
    x, y = data.T
    nbins = 50
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))                                      
    plt.contour(xi,yi,zi.reshape(xi.shape), 10)
    abline(1,0)
    r2 = np.corrcoef(data[:,0], data[:,1])[0,1]**2
    plt.title('%s, $r^2$ = %1.2f' % (title, r2))
    plt.show()
