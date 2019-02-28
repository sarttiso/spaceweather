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
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    plt.figure()
    plt.scatter(x, y, marker='.')
    k = kde.gaussian_kde(np.concatenate((x,y), axis=0))
    nbins = 50
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))                                      
    plt.contour(xi,yi,zi.reshape(xi.shape), 10)
    abline(1,0)
    r2 = np.corrcoef(x, y)[0,1]**2
    plt.title('%s, $r^2$ = %1.2f' % (title, r2))
    plt.show()
 
    
"""
Plot straight line in current axis.
"""
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'k--')