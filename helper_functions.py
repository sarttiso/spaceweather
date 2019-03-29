# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:36:17 2019

@author: Adrian
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import kde
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