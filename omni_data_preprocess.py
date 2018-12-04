#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:33:58 2018

@author: adrianraph
"""

#%% packages

# general
import numpy as np
from sklearn import preprocessing
import os
import pdb
import datetime as dt

# deep learning
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU

# file management, io
import pandas as pd
import h5py

# plotting
import matplotlib.pyplot as plt
import seaborn as sb

#%% load saved data
# load hdf file
omni_lr = pd.read_hdf('../OMNI_data/omni_hourly_1998-2017.h5')
# let's ignore the values from 2018 (just from January 1st)
omni_lr = omni_lr[omni_lr['Year'] != 2018]

# now split up data
yr = omni_lr['Year'].values
yr_unique = np.unique(yr)
nyr = len(yr_unique)
t = omni_lr['python time'].values
# predictors
X = omni_lr[['Bx (nt, GSE, GSM)', \
             'By (nt, GSM)', \
             'Bz (nt, GSM)', \
             'SW Proton Density (N/cm^3)', \
             'SW Plasma Speed (km/s)', \
             'Alpha/Prot. Ratio', \
             'f10.7_index', \
             'R (sunspot no.)']].values
# array with gaps filled
Xgapsfilled = X.copy()
# concise feature names
feat = ['Bx', 'By', 'Bz', 'Proton Density', 'SW Speed',\
        'Alpha/Proton Ratio', 'f10.7 index', 'Sunspot No.']
nfeat = X.shape[1]
# targets (Dst for now)
Y = omni_lr['Dst-Index (nT)'].values

#%% GAPS

# list gaps by year for each variable and total at end
nanvals = np.max(X,axis=0)
# replace nanval for sunspots; no missing data there
nanvals[-1] = 999.9

gaps_year = np.zeros((nyr,nfeat))
gaps_total = np.zeros((1,nfeat))

for jj in range(nfeat):
    for ii in range(nyr):
        # gap percent
        gaps_year[ii,jj] = np.sum(X[yr==yr_unique[ii],jj] == nanvals[jj])/ \
            np.sum(yr==yr_unique[ii]) * 100
    gaps_total[0,jj] = np.sum(X[:,jj] == nanvals[jj])/X.shape[0] * 100
        
gaps = pd.DataFrame(gaps_year,index=yr_unique,columns=feat)
gaps = gaps.append(pd.DataFrame(gaps_total,columns=feat,index=['Total']))


#%% recurrent architecture for B field

# Find sequences of a given value within an input vector. 
#
# IN:
# x: vector of values in which to find sequences
# val: scalar value to find sequences of in x
# noteq: (false) whether to find sequences equal or not equal to the supplied
#    value
#
# OUT:
# idx: array that contains in rows the number of total sequences of val, with 
#   the first column containing the begin indices of each sequence, the second 
#   column containing the end indices of sequences, and the third column 
#   contains the length of the sequence.
def findseq(x,val,noteq=False):
    x = x.copy().squeeze()
    assert len(x.shape) == 1, "x must be vector"
    # indices of value in x, and
    # compute differences of x, since subsequent occurences of val in x will 
    # produce zeros after differencing. append nonzero value at end to make 
    # x and difx the same size
    if noteq:
        validx = np.argwhere(x != val).squeeze()
        x[validx] = val+1
        difx = np.append(np.diff(x),1)
    else:
        validx = np.argwhere(x == val).squeeze()
        difx = np.append(np.diff(x),1)
    nval = len(validx)
    # if val not in x, warn user
    if nval == 0:
        warnings.warn("value val not found in x")
        return 0
    
    # now, where validx is one and difx is zero, we know that we have 
    # neighboring values of val in x. Where validx is one and difx is nonzero, 
    # we have end of a sequence
    
    # now loop over all occurrences of val in x and construct idx
    c1 = 0
    idx = np.empty((1,3))
    while c1 < nval:
        curidx = np.array([[validx[c1],validx[c1],1]])
        c2 = 0
        while difx[validx[c1]+c2] == 0:
            curidx[0,1] += 1
            curidx[0,2] += 1
            c2 += 1
        idx = np.append(idx,curidx,axis=0)
        c1 = c1+c2+1
    idx = idx[1:,:].astype(int)
    return idx


#%% construct data batches and split for training, validation, testing

# depending on batch size and gaps in data, construct batches
#
# IN: 
#  data: nd-array of data with shape (nobs, nfeatures)
#  dataidx: nd-array of beginning and end indices of data sequences in data with
#   shape (nsequences,3) and columns (beg index, end index, length of sequence)
#  batch_size: integer of batch size to split data into
#
# OUT:
#  batches: nd-array with size (nbatches*batch_size, 1, nfeatures), as required
#   by keras; a more intuitive shape would be (nbatches, batch_size, nfeatures)
#
def getbatches(data,dataidx, batch_size):
    assert dataidx.shape[1] == 3, 'dataidx must be correct shape'
    assert batch_size < np.max(dataidx[:,2]), \
         'at least one data sequence must be long enough for given batch_size'
    nseq = dataidx.shape[0]
    nfeatures = data.shape[1]
    # within each interval of data, split into largest possible number of 
    # batches with length batch_size
    X = np.zeros((batch_size, nfeatures))
    for ii in range(nseq):
        # skip this sequence if it is too short
        if dataidx[ii,2] < batch_size:
            continue
        nbatch = np.floor(dataidx[ii,2]/batch_size)
        curidx = np.arange(dataidx[ii,0],dataidx[ii,0]+int(nbatch*batch_size))
        curdat = data[curidx,:].reshape((int(nbatch*batch_size), nfeatures))
        X = np.append(X, curdat, axis=0)
    # remove first batch_size zeros at beginning
    X = X[batch_size:,:]
    return X



# From a data array with observations in rows and features in columns, split 
# split into testing and training data based on which rows to use (datain_idx), 
# batch_sizes, which features for prediction (incols) and which features as 
# targest (outcols). Since we're concerned with time series forecasting, the 
# number of steps to forecast ahead is lahead.
#
# IN:
#
# OUT:
#
def datasplit(data, datain_idx, batch_size, incols, \
              outcols, train_percent=0.8, lahead=1):
    nfeatin = len(incols)
    dataout_idx = datain_idx.copy()
    dataout_idx[:,0] = dataout_idx[:,0]+lahead
    dataout_idx[:,2] = dataout_idx[:,2]-lahead
    dat_in = getbatches(data[:,incols], 
                        datain_idx, 
                        batch_size).reshape(-1,1,nfeatin)
    dat_out = getbatches(data[:,outcols], 
                         dataout_idx, batch_size)
    nbatch = dat_in.shape[0]/batch_size
    # now construct training and testing sets
    trainbatch = np.random.permutation(int(nbatch))[0:int(0.8*nbatch)]
    trainidx = np.zeros(dat_in.shape[0],dtype=bool)
    for ii in range(trainbatch.shape[0]):
        curidx = np.arange(trainbatch[ii]*batch_size, \
                           trainbatch[ii]*batch_size+batch_size)
        trainidx[curidx] = True
    testidx  = np.logical_not(trainidx)
    
    dat_in_train = dat_in[trainidx,:,:]
    dat_out_train = dat_out[trainidx,:]
    dat_in_test = dat_in[testidx,:,:]
    dat_out_test = dat_out[testidx,:]
    return dat_in_train, dat_out_train, dat_in_test, dat_out_test
    

# we want network to predict next value
lahead = 1
# let's work with batches
batch_size = 1500 # in hours
# indices of uninterrupted data sequences
datain_idx = findseq(X[:,1],nanvals[1],noteq=True)

# subsequent batches, let's use Bx, By, Bz, and sunspots as input
incols = [0,1,2,7]
nfeatin = len(incols)
outcols = [0,1,2]
nfeatout = len(outcols)
dat_in_train, dat_out_train, dat_in_test, dat_out_test = \
        datasplit(X,
              datain_idx,
              batch_size,
              incols,
              outcols,
              train_percent=0.8,
              lahead=lahead)

# scale outputs
#mu = np.mean(dat_in_train, axis=0)
#std = np.std(dat_in_train, axis=0)
#dat_in_train = (dat_in_train-mu)/std
#dat_in_test = (dat_in_test-mu)/std
#dat_out_train = (dat_out_train-mu[0,0:nfeatout])/std[0,0:nfeatout]
#dat_out_test = (dat_out_test-mu[0,0:nfeatout])/std[0,0:nfeatout]

# recurrent architecture, create input and output datasets
rnn = Sequential()
rnn.add(LSTM(200, 
        name='LSTM1',
        stateful=False,
        input_shape=(1,nfeatin),
        batch_size=batch_size,
        return_sequences=False,
        activation='relu'))
#rnn.add(GRU(100,
#            name='GRU1',
#            input_shape=(1,nfeatin),
#            batch_size=batch_size,
#            return_sequences=False,
#            activation='tanh'))
#rnn.add(LSTM(100, 
#        name='LSTM2',
#        stateful=False,
#        input_shape=(1,3),
#        batch_size=batch_size,
#        return_sequences=True,
#        activation='relu'))
rnn.add(Dense(nfeatout,
              name='Dense'))
opt = keras.optimizers.Adam(lr=0.001)
rnn.compile(loss='mse',optimizer=opt)

# fit model
for ii in range(100):
    print(ii)
    rnn.reset_states()
    rnn.fit(dat_in_train,
            dat_out_train,
            epochs=1,
            batch_size=batch_size,
            shuffle=False,
            verbose=0)

# evaluate on test data
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

rnn.reset_states()
dat_pred = rnn.predict(dat_in_test,batch_size=batch_size)
for ii in range(nfeatout):
    print(np.corrcoef(dat_out_test[:,ii],dat_pred[:,ii])[0,1]**2)
    plt.scatter(dat_out_test[:,ii],dat_pred[:,ii])
    sb.kdeplot(dat_out_test[:,ii].squeeze(),
               dat_pred[:,ii].squeeze(),
               n_levels=10)
    abline(1,0)
    plt.title(feat[outcols[ii]])

# use network to fill gaps in IMF data
gapidx = findseq(X[:,1],nanvals[1])
ngaps = gapidx.shape[0]

for ii in range(ngaps):
    curgap = gapidx[ii,2]
    for jj in range(curgap):
        cur_in = X[gapidx[ii,0]+jj-batch_size:gapidx[ii,0]+jj,incols]
        cur_in = cur_in.reshape(-1,1,nfeatin)
        rnn.reset_states()
        cur_pred = rnn.predict(cur_in,batch_size=batch_size)
        Xgapsfilled[gapidx[ii,0]+jj,outcols] = cur_pred[-1,:]

#%% having filled gaps in IMF, now train on f10.7 index

# indices of uninterrupted data sequences
datain_idx = findseq(X[:,6],nanvals[6],noteq=True)

lahead = 1
batch_size = 1000

# subsequent batches, let's use Bx, By, Bz, sunspots, and previous f10.7 as input
incols = [0,1,2,6,7]
nfeatin = len(incols)
outcols = [6]
nfeatout = len(outcols)
dat_in_train, dat_out_train, dat_in_test, dat_out_test = \
        datasplit(X,
              datain_idx,
              batch_size,
              incols,
              outcols,
              train_percent=0.8,
              lahead=lahead)
        
# RNN
rnn = Sequential()
rnn.add(LSTM(200, 
        name='LSTM1',
        stateful=False,
        input_shape=(1,nfeatin),
        batch_size=batch_size,
        return_sequences=False,
        activation='relu'))
rnn.add(Dense(nfeatout,
              name='Dense'))
opt = keras.optimizers.Adam(lr=0.001)
rnn.compile(loss='mse',optimizer=opt)

# fit
for ii in range(100):
    print(ii)
    rnn.reset_states()
    rnn.fit(dat_in_train,
            dat_out_train,
            epochs=1,
            batch_size=batch_size,
            shuffle=False,
            verbose=0)

rnn.reset_states()
dat_pred = rnn.predict(dat_in_test,batch_size=batch_size)

for ii in range(nfeatout):
    print(np.corrcoef(dat_out_test[:,ii],dat_pred[:,ii])[0,1]**2)
    plt.scatter(dat_out_test[:,ii],dat_pred[:,ii])
    sb.kdeplot(dat_out_test[:,ii].squeeze(),
               dat_pred[:,ii].squeeze(),
               n_levels=10)
    abline(1,0)

gapidx = findseq(X[:,1],nanvals[1])
ngaps = gapidx.shape[0]

for ii in range(ngaps):
    curgap = gapidx[ii,2]
    for jj in range(curgap):
        cur_in = X[gapidx[ii,0]+jj-batch_size:gapidx[ii,0]+jj,incols]
        cur_in = cur_in.reshape(-1,1,nfeatin)
        rnn.reset_states()
        cur_pred = rnn.predict(cur_in,batch_size=batch_size)
        X[gapidx[ii,0]+jj,outcols] = cur_pred[-1,:]