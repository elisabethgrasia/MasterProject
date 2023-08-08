# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:06:24 2023

@author: alvak
"""

"""
subfunction to calculate the noise of a signal.

The noise is the median of all absolute differences between
the signal at i and the mean of its immediate neighbours
i-1 and i+1
"""
import numpy as np

def calcNoise(signal):
    neighbour1 = signal[:-2]
    neighbour2 = signal[2:]
    meanNeighbours = (neighbour1 + neighbour2) / 2
    diffWithNeighbour = signal[1:-1] - meanNeighbours
    noiseVal = np.median(np.abs(diffWithNeighbour))
    return noiseVal