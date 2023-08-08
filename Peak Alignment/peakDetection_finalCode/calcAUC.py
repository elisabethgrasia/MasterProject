# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:07:20 2023

@author: alvak
"""

"""
subfunction to calculate the areua under the curve of a signal.

The area is calculated using the Trapezoidal rule since some of 
the peaks are very sharp to use first principle. 
The inputs are the part of the signal with the peak, and the time or
the timesteps corresponding to the signal.
"""
import numpy as np


def calcAUC(signal, etime):
    val1 = signal[:-1]
    val2 = signal[1:]
    incTime = np.diff(etime)
    resArea = np.sum((val1 + val2) * incTime / 2)
    return resArea