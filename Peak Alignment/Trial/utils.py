# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:18:39 2023

@author: alvak
"""

import numpy as np
# from scipy.linalg import lstsq
import scipy.sparse as sp
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
import time

#signalMZ = mzDataExpt[0, :].reshape((-1, 1))


## Find baseline
def baseCorrAls(y, l, p):
    m = len(y)
    D = (np.diff(eye(m).toarray(), 2)).T
    D = sp.csc_matrix(D, dtype=np.float32)
    w = np.ones(m)
    for i in range(10):
        W = diags(w, 0, shape=(m, m))
        DTD = D.T @ D
        C = splu(W + l * DTD)
        wy = w * y.ravel()
        z = C.solve(wy)
        w = p * (y.ravel() > z) + (1 - p) * (y.ravel() < z)
    return z

def calcAUC(signal, etime):
    val1 = signal[:-1]
    val2 = signal[1:]
    incTime = np.diff(etime)
    resArea = np.sum((val1 + val2) * incTime / 2)
    return resArea

def calcNoise(signal):
    neighbour1 = signal[:-2]
    neighbour2 = signal[2:]
    meanNeighbours = (neighbour1 + neighbour2) / 2
    diffWithNeighbour = signal[1:-1] - meanNeighbours
    noiseVal = np.median(np.abs(diffWithNeighbour))
    return noiseVal

import math

"""
subfunction to partition the whole signal
"""

def partitionSignal(smdInitial, thrSig, N, splitSignalSize):
    nPart = math.ceil(N / splitSignalSize)
    partStIndex = np.zeros(nPart, dtype=int)
    k = 1
    partStIndex[k-1] = k
    flgPartition = True
    
    while flgPartition:
        k += 1
        # get to the next start point
        stNext = partStIndex[k-2] + splitSignalSize
        # check if this point and its surrounding 10 points are all below
        # the threshold, if not, then go to the next point along the line
        # and search until a section is found
        surrPoints = 10
        while True:
            # secVal = smdInitial(stNext-surrPoints : stNext+surrPoints)
            if all(smdInitial[stNext-surrPoints-1 : stNext+surrPoints] < thrSig):
                break
            else:
                stNext += 1
                
            # make sure that the index stNext is not near the end of the data
            if stNext+surrPoints >= N:
                stNext = 0
                break
        partStIndex[k-1] = stNext
        
        # check if the number of points remaining is smaller than splitSignalSize
        # or not, exit while loop if true
        if N - stNext - surrPoints <= splitSignalSize or stNext == 0:
            flgPartition = False
            break
        
    partStIndex = np.array((partStIndex[partStIndex != 0]))
    partEndIndex = np.concatenate((partStIndex[1:]-1, np.array([N])))
    
    return partStIndex, partEndIndex
    
