# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 17:17:25 2023

@author: alvak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import time

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import sys
sys.path.append('../src/')

from models import ConvNet
from preprocessing import LabelEncoder

def read_data(path, i, **kwargs):
    from scipy.interpolate import interp1d
    
    
    # cnn_data = pd.read_csv("chrom_temp3.csv", header=None, delimiter="\t")
    # data = pd.read_csv(path, **kwargs)
    
    df = pd.read_csv("peak_alignment_csv.csv", header=None)
    
    massZ = df.iloc[0, 1:]
    dataRaw = df.iloc[1:, ]
    dataRaw = df.iloc[1:, ].T
    
    data = [np.array(dataRaw.iloc[0,:]), np.array(dataRaw.iloc[i, :])]
    x, y = data[0], data[1]
    
    ##Normalization
    # normalized_y = (y - np.mean(y) )/ np.std(y)
    
    
    
    
    # Obtain interpolation function (input original x (time) and y (signal))
    f = interp1d(x,y)
    # Create new x(same x_min and x_max but different number of data)
    x_new = np.linspace(x.min(), x.max(), 8192)
    # Obtain new y (based on new x)
    y_new = f(x_new)
    # return both new x and new y
    return x_new, y_new

NUM_TRAIN_EXAMPLES = int(1e2)
NUM_TEST_EXAMPLES = int(1e2)

NUM_CLASSES = 3
NUM_WINDOWS = 256
INPUT_SIZE = 8192

# Define label encoder to decode predictions later on
label_encoder = LabelEncoder(NUM_WINDOWS)


# Build model (should have the same parameters as the model in cnn_train)
model = ConvNet(
    filters = [64, 128, 128, 256, 256],
    kernel_sizes = [9, 9, 9, 9, 9],
    dropout = 0.0,
    pool_type='max',
    pool_sizes=[2, 2, 2, 2, 2],
    conv_block_size=1,
    input_shape=(INPUT_SIZE, 1),
    output_shape=(NUM_WINDOWS, NUM_CLASSES),
    residual=False
)

# Load pretrained weights
model.load_weights('cnn_weights_000.h5')
time_process =[]

for i in range(1,317):
    start = time.time()

    # Read data (applies linear interpolation to get the right dimension)
    t, s = read_data('chrom_temp3.csv',i, header=None, delimiter='\t')
    
    # Normalize and add batch dimension (model expext batched data)
    s_norm = s[None, :, None] / s.max()
    
    # Pass to model and make predictions
    preds = model(s_norm)[0]
    
    # .decode will filter out predictions below threshold
    # and compute the appropriate locations of the peaks
    probs, locs, areas = label_encoder.decode(preds, threshold=0.5)
    
    if i==1:
        print("Predicted locations:\n", locs * t.max())
        print("\nPredicted areas:\n", areas)
        print("\nPredicted probabilities:\n", probs)
        
        # Visualize locations in the chromatogram
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        ax.plot(t,np.squeeze(s_norm))
        
        for (prob, loc, area) in zip(probs, locs, areas):
            # y = s.min() - s.max() * 0.05 # location of the triangles
            y =-0.01# location of the triangles
            ax.scatter(loc*t.max(), y, color='C1', marker='^', s=100, edgecolor='black')   
            ax.tick_params(axis='both', labelsize=16)
            ax.set_ylabel('Signal', fontsize=16)
            ax.set_xlabel('Time', fontsize=16)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    end = time.time()
    
    time_process.append(end-start)
    
    print("Processing time :", end-start)
    
    
import matplotlib.pyplot as plt
plt.plot(time_process)