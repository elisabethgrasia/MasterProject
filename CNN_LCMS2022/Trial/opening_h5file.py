# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:41:13 2023

@author: alvak
"""

import h5py

filename = "cnn_weights.h5"

h5 = h5py.File(filename,'r')

datasetNames = [n for n in h5.keys()]
for n in datasetNames:
    print(n)
    
    

#futures_data = h5['futures_data']  # VSTOXX futures data
#options_data = h5['options_data']  # VSTOXX call option data

#h5.close()