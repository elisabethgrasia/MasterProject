# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:38:21 2023

@author: alvak
"""

import pandas as pd
import numpy as np

cnn_data = pd.read_csv("chrom_temp3.csv", header=None, delimiter="\t")

df = pd.read_csv("peak_alignment_csv.csv", header=None)

massZ = df.iloc[0, 1:]
dataRaw = df.iloc[1:, ]
dataRaw = df.iloc[1:, ].T

for i in range(1, len(dataRaw)):
    A = [np.array(dataRaw.iloc[0,:]), np.array(dataRaw.iloc[i, :])]


def read_data(path, **kwargs):
    from scipy.interpolate import interp1d
    #data = pd.read_csv(path, **kwargs)
    
    df = pd.read_csv("peak_alignment_csv.csv", header=None)

    massZ = df.iloc[0, 1:]
    dataRaw = df.iloc[1:, ]
    dataRaw = df.iloc[1:, ].T
    
    
    x, y = data[0], data[1]
    # Obtain interpolation function (input original x (time) and y (signal))
    f = interp1d(x, y)
    # Create new x(same x_min and x_max but different number of data)
    x_new = np.linspace(x.min(), x.max(), 8192)
    # Obtain new y (based on new x)
    y_new = f(x_new)
    # return both new x and new y
    return x_new, y_new

