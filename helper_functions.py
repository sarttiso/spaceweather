# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:36:17 2019

@author: Adrian
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import kde
import tensorflow as tf
import sklearn
from sklearn import preprocessing

"""
Find sequences of a given value within an input vector.

IN:
 x: vector of values in which to find sequences
 val: scalar value to find sequences of in x
 noteq: (false) whether to find sequences equal or not equal to the supplied
    value

OUT:
 idx: array that contains in rows the number of total sequences of val, with
   the first column containing the begin indices of each sequence, the second
   column containing the end indices of sequences, and the third column
   contains the length of the sequence.
"""
def findseq(x, val, noteq=False):
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

"""
Plot scatter plot with 1:1 line and point density contours.
IN:
data (nd.array): ndata x 2 array
"""
def plotcorr(x, y, title=''):
    x = x.squeeze()
    y = y.squeeze()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    plt.figure()
    plt.scatter(x, y, marker='.')
    plotcontours(x, y)
    abline(1,0)
    r2 = np.corrcoef(x.reshape(-1), y.reshape(-1))[0,1]**2
    plt.title('%s, $r^2$ = %1.2f' % (title, r2))
    plt.show()


"""
This function plots point density contours.

IN:
    x (nd.array): xcoords of points
    y (nd.array): ycoords of points
    ncountours (int): number of contours to plot
    nbins (int): number of bins over which to estimate point densities
"""
def plotcontours(x, y, ncontours=10, nbins=50):
    k = kde.gaussian_kde(np.concatenate((x,y), axis=1).T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.contour(xi,yi,zi.reshape(xi.shape), ncontours)

"""
Plot straight line in current axis.
"""
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    line = plt.plot(x_vals, y_vals, 'k--')
    return line[0]


"""
Isolate periods of storms from external coefficient time series.

IN:
data (nd.array): time series of Dst data to find storms in
threshold (float): defines the threshold for which to find indices of intervals
less (bool): (default = true) whether to define storms as greater than or less
    than the threshold value
nhr_before (int): (default = 36) number of hours before to include in the storm
    time series
nhr_after (int): (default = 100) number of hours after peak values to include in
    the storm time series

OUT:
stormidx (nd.array bool): array of l
"""
def findstorm(data, threshold, less=True, nhr_before = 36, nhr_after = 100):

    ndat = data.shape[0]

    # first, threshold all q10 values
    if less:
        main_idx = data < threshold
    else:
        main_idx = data > threshold

    # indices of q10 > 50
    thres_idx = findseq(main_idx, 1)

    # begin with first entry of thresholded indices
    storm_idx = thres_idx[0,:].reshape(1,-1)
    # counter
    c = 0
    for ii in range(len(thres_idx)-1):
        if (thres_idx[ii+1,0] - storm_idx[c,1]) < 48:
            storm_idx[c,1] = thres_idx[ii+1,1]
            storm_idx[c,2] = storm_idx[c,1] - storm_idx[c,0] + 1
        else:
            storm_idx = np.append(storm_idx, thres_idx[ii+1,:].reshape(1,-1), \
                                  axis=0)
            c += 1

    nstorm = storm_idx.shape[0]

    for ii in range(nstorm):
        storm_idx[ii,0] = np.max((0, storm_idx[ii,0]-nhr_before))
        storm_idx[ii,1] = np.min((ndat, storm_idx[ii,0]+nhr_after+nhr_before))

    storm_idx = storm_idx[:, 0:2]

    return storm_idx


"""
This function converts indices from the format given by findseq() or findstorm()
above to logical indices.

IN:
idx (nd.array): array of shape (ninterval x 2) where values in the first column
    designate that beginning of indexed intervals and values in the second
    column designate the end of indexed intervals.
length (int): length of the output array of logical indices

OUT:
idx_logical (boolean nd.array): logical indices
"""
def convertidx(idx, length):
    idx_logical = np.zeros(length, dtype=bool)
    ninterval = idx.shape[0]

    for ii in range(ninterval):
        idx_logical[idx[ii,0]:idx[ii,1]+1] = True

    return idx_logical


"""
Compute root mean squared error

IN:
res (nd.array): vector (1D ndarray) of residuals to compute RMSE on

OUT:
rmse (float): RMSE of given residuals
"""
def RMSE(res):
    assert len(res.shape) == 1, "data should be 1D ndarray"
    rmse = np.sqrt(np.mean(res**2))
    return rmse


def datasplit(data_in, data_out, batch_size, train_percent=0.8, lahead=1, \
              sequential=True):
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
        sequential: (boolean) whether to keep batches in order

    OUT:
        data_in_train:
        data_out_train:
        data_in_test:
        data_out_test:

    """
    # make sure to cast to float32
    data_in = data_in.astype(np.float32)
    data_out = data_out.astype(np.float32)

    data_in = data_in[0:-lahead,:]
    data_out = data_out[lahead:,:]

    ndat = data_in.shape[0]

    if sequential:
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


def reliability(pred, obs, thres, bin_edges, exc='geq', first=True, bootstrap=500):
    """
    This function computes reliability curves for first exceedance of a given threshold in a time series. Given a probilistic forecast of exceedance of the threshold, this function will compute the actual observed exceedance rate of the threshold (at that probability level). Given that we  always have finite data, we first bin the data into intervals of probability.
    
    IN:
        pred: list of objects with cdf() method (e.g. scipy stats object) to evaulate distribution function
        obs: array of observations equal in length to pred
        thres: threshold for which to compute exceedance rates
        bin_edges: array-like of bin edges of probabilities to consider
        exc: 'geq' (default) or 'leq', direction in which to consider exceedance
        first (bool): (default=True) whether to only consider first exceedance in a sequence of consecutive exceedances or to consider all exceedances of threshold
        
    OUT:
        obs_exc: for bins, returns observed exceedance rates
    """
    # for each prediction, compute exceedance probability of threshold
    prob_pred = pred.cdf(thres)
    
    # take other complementary probability mass for greater than or equal
    if exc =='geq':
        prob_pred = 1-prob_pred
    
    # ignore nans
    val_idx = np.logical_not(np.isnan(prob_pred))
    prob_pred = prob_pred[val_idx]
    obs = obs[val_idx]
    
    # set any predicted probabilities of 1 to 0.9999 (pathological case)
    prob_pred[prob_pred==1] = 0.9999
    
    nobs = len(obs)
    
    # compute exceedances indices
    if exc == 'geq':
        exc_idx = obs > thres
    elif exc == 'leq':
        exc_idx = obs < thres
    
    # if we only want first exceedances, find them
    if first:
        exc_idx = findseq(exc_idx, 1)
        exc_idx = exc_idx[:, 0]
        tmp_idx = np.zeros(nobs, dtype=bool)
        tmp_idx[exc_idx] = True
        exc_idx = tmp_idx
    
    nbins = len(bin_edges) - 1
    obs_exc = np.zeros((nbins, bootstrap))
    consist = np.zeros((nbins, bootstrap))
    
    # bin data by probability bins
    bin_idx = np.digitize(prob_pred, bin_edges)-1
    
    for tt in range(bootstrap):
        # resample within each bin
        cur_exc_idx = exc_idx.copy()
        for ii in range(nbins):
            cur_bin_idx = np.where(bin_idx==ii)[0]
            n_in_bin = len(cur_bin_idx)
            sam_idx = np.random.randint(0, n_in_bin, n_in_bin)
            re_sam_idx = cur_bin_idx[sam_idx]
            cur_exc_idx[cur_bin_idx] = cur_exc_idx[re_sam_idx]

        bin_c = np.bincount(bin_idx, weights=cur_exc_idx)/np.bincount(bin_idx)
        obs_exc[0:len(bin_c), tt] = bin_c
        
    # also compute consistency computation (see Brocker and Smith, 2007)
    for tt in range(bootstrap):
        z = np.random.rand(nobs)
        y = z < prob_pred
        bin_c = np.bincount(bin_idx, weights=y)/np.bincount(bin_idx)
        consist[0:len(bin_c), tt] = bin_c
    
    obs_exc_mean = np.mean(obs_exc, axis=1)
    obs_exc_std = np.std(obs_exc, axis=1)
    obs_exc_pct = np.percentile(obs_exc, [2.5, 97.5], axis=1)
    
    # compute consistency intervals 2.5-97.5 quantiles
    consist_pct = np.percentile(consist, [2.5, 97.5], axis=1)
           
    return obs_exc_mean, obs_exc_pct, consist_pct
   
# not implemented as stand-alone function yet
# def brier():
#     brier = 0;
#     obs_exc_total = np.sum(exc_idx)/nobs
#     bin_cen = (bin_edges[1:] + bin_edges[:-1]) / 2
#     for ii in range(nbins):
#         brier += 1/nobs*(n_in_bin*(obs_exc[ii]-bin_cen[ii])**2) - \
#                  1/nobs*(n_in_bin*(bin_cen[ii]-obs_exc_total)**2)
#     brier += obs_exc_total*(1-obs_exc_total)


class Dataset():
    """
    Class for sampling from input and ouput data to create training and testing 
    splits. This class allows the user to avoid nan values in the data during 
    sampling and thereby isolate sequences of consecutive data (in the case of
    series data) of size batch_size. No data must be specified as np.nans.
    """
    def __init__(self, data_in, data_out):
        """
        input and output data must have same number of entries, with data
        dimensionality along the second and following dimensions.
        
        data_in (nd.array): dimensions (n_obs x input_dim)
        data_out (nd.array): dimensions (n_obs x output_dim)
        """
        self.n_obs = data_in.shape[0]
        assert self.n_obs == data_out.shape[0], \
            'number of obsevations must be same in data_in and data_out'
            
        # input and output dimensionalities
        self.d_in = data_in.shape[1]
        self.d_out = data_out.shape[1]
        
        # store data
        self.data_in = data_in.astype(np.float32)
        self.data_out = data_out.astype(np.float32)
        
        # compute indices of overlapping data (i.e. points without any nans)
        self.dat_idx = \
         np.logical_and(np.all(np.logical_not(np.isnan(data_in)), axis=1),
                        np.all(np.logical_not(np.isnan(data_out)), axis=1))
        
        
    def split(self, 
              batch_size, 
              train_frac=0.8, 
              test_frac=0.2, 
              val_frac=0., 
              shuffle=False,
              overlap=False,
              ext_array=None):
        """
        Split data into training, testing (and validation) sets
        
        shuffle: barring no data gaps, keep data in consecutive order. if true,
            shuffle order of batches NOT IMPLEMENTED YET
        overlap: (default False) in regions where continuous data length is not 
            a multiple of the batch_size, allow the final batch to overlap with
            the previous batch NOT IMPLEMENTED YET
        idx_array: (default None) array to be split in the same way as the 
            training, testing, and validation sets. Creates 3 additional
            outputs
        """
        assert train_frac + test_frac + val_frac == 1.0, \
            'splits must sum to one'
        
        # indices and legnths of data sequences
        dat_seq = findseq(self.dat_idx, 1)
        
        assert np.any(batch_size < dat_seq[:,2]), \
            'batch_size too large for data gaps'
            
        # make sure extra array, if requested, appropriate size
        if np.all(ext_array != None):
            assert len(ext_array) == self.n_obs, 'ext_array wrong size'
        
        # only interested in those longer than batch size
        dat_seq = dat_seq[np.argwhere(dat_seq[:,2] > batch_size).squeeze()]
        
        # make sure we have 2D array
        dat_seq = dat_seq.reshape(-1, 3)
    
        # total number of continuous data sequences longer than batch_size
        n_seq = dat_seq.shape[0]
            
        # create batch indices
        idx = np.zeros(self.n_obs, dtype=bool)
        
        # construct batches
        batches_in = []
        batches_out = []

        for ii in range(n_seq):
            n_batch_in_seq = np.floor(dat_seq[ii,2]/batch_size).astype(int)
            cur_idx = slice(dat_seq[ii, 0], 
                        dat_seq[ii, 0] + n_batch_in_seq*batch_size)
            idx[cur_idx] = True
            # include extra batch at end with overlap?
            if overlap:
                pass
                
        batches_in = self.data_in[idx]
        batches_out = self.data_out[idx]
        
        # also split ext array if requested
        if np.all(ext_array != None):
            batches_ext_array = ext_array[idx]
            
        # split batches into training, testing, and validation sets
        n_batch = int(batches_in.shape[0]/batch_size)
        
        n_train = np.floor(train_frac*n_batch).astype(int)
        n_val = np.floor(val_frac*n_batch).astype(int)
        n_test = n_batch - n_train - n_val
    
        if shuffle:
            pass
        
        # training data
        train_idx = slice(0, batch_size * n_train)
        train_in = batches_in[train_idx]
        train_out = batches_out[train_idx]
        
        # testing data
        test_idx = slice(batch_size * n_train, 
                         batch_size * (n_train + n_test))
        test_in = batches_in[test_idx]
        test_out = batches_out[test_idx]
        
        # validation data
        val_idx = slice(batch_size * (n_train + n_test),
                        batch_size * (n_train + n_test + n_val))
        val_in = batches_in[val_idx] 
        val_out = batches_out[val_idx]
        
        # split ext_array
        if np.all(ext_array != None):
            train_ext = batches_ext_array[train_idx]
            test_ext = batches_ext_array[test_idx]
            val_ext = batches_ext_array[val_idx]
        
        return train_in, train_out, test_in, test_out, val_in, val_out, \
            train_ext, test_ext, val_ext
    
    
class PeriodicHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            print('Epoch %d: Loss = %1.2f, MSE = %1.2e' % \
                  (epoch, logs['loss'], logs['mse']))