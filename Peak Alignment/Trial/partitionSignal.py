# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:32:40 2023

@author: alvak
"""
import numpy as np
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
        
    partEndIndex = np.concatenate((partStIndex[1:]-1, np.array([N])))
    partStIndex = np.array((partStIndex[partStIndex != 0]))
    return partStIndex, partEndIndex
    
