# -*- coding: utf-8 -*-
"""
Created on Sat May  6 21:57:14 2023

@author: alvak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from GCPeakDetection import GCPeakDetection
import utils
import time

 
# other show plot flags
showPlotMZ = True   # ind ms data and baseline
showPlotMain = True 

splitSignalSize = 100
degPoly = 3
thresholdSD = [0, 10]
thresholdFD = [0, 5]
sigThreshold = 2e4
thresholdSignal = [1, sigThreshold]
splitCoelution = False
showPlot = True

      
df = pd.read_csv('peak_alignment_csv.csv', header = None)

massZ = df.iloc[0, 1:]
dataRaw = df.iloc[1:, ]
dataRaw = df.iloc[1:, ].T

dC = dataRaw.iloc[1:, 92:]

Nmz = len(massZ)

## Initialise the results
resMSPeak = []
peakData = []

## Separate the raw data into time stamps and intensity
mzDataExpt = dC.iloc[1:, ]
mzDataExpt = mzDataExpt.to_numpy()
timesteps = dC.iloc[0, :]

## If we want to normalise the data
#if normaliseData == 2:
#    data = df.loc[:, (df > 0).all()]     
#    meanBlankSample = data.mean()
#    mzDataExpt = mzDataExpt / meanBlankSample
#elif normaliseData == 3:
    
 #   mzDataExpt = mzDataExpt / meanAllSamples
    
# ==============================================================================
# ======= redefine the signal threshold using the overall noise ================
sigThresholdNormFactor = 10

neighbour1 = mzDataExpt[:, :-2]
neighbour2 = mzDataExpt[:, 2:]
meanNeighbours = (neighbour1 + neighbour2) / 2
diffWithNeighbours = mzDataExpt[:, 1:-1] - meanNeighbours
noiseValMZ = np.median(abs(diffWithNeighbours), axis = 1)
sigThresholdNorm = noiseValMZ.mean() * sigThresholdNormFactor
thresholdSignal = [1, sigThresholdNorm]
    
NPeaksOverAllMz = 0

for mzi in range(Nmz):
    start_time = time.time()
    signalMZ = mzDataExpt[mzi+59, :].reshape((-1, 1))
    if signalMZ.max() <= 10:
        lambdaALS = 2e3
    else: 
        lambdaALS = 5e3
        
    pALS = 1e-4
    
    # to save time, we save the baseline into .mat variable, thus
    # we do not have look for the baseline everytime we run the code
    filename = 'my_dCfile{}.mat'.format(massZ[mzi+59])
    
    if os.path.isfile(filename):
        import scipy.io as sio
        # Load the .mat file
        data = sio.loadmat(filename)
        # Extract the variable from the data dictionary
        baselineMZ = np.squeeze(data['my_variable'])
        
    else:
        baselineMZ = utils.baseCorrAls(signalMZ, lambdaALS, pALS)
        import scipy.io as sio
        # Save the variable in a .mat file
        sio.savemat('my_dCfile{}.mat'.format(massZ[mzi+59]), {'my_variable': baselineMZ})
    
    # ----------------------------------------------------------------
    # after we got the baseline,        
    # showing the plot
    if showPlotMZ:
        hFigMZ = plt.figure(4)
        plt.clf()
        plt.plot(timesteps, signalMZ, 'b', linewidth = 1)
        plt.plot(timesteps, baselineMZ, 'r')
        strTitle = 'M/Z = {}'.format(massZ[mzi+1])
        plt.title(strTitle, fontsize = 12, fontweight = 'bold')
        #plt.show()
        #print('press any key to continue......')
        #input()
       
    # It seems it's a good idea to always take the baseline drift off
    # the reason for checking the drift was mainly to account for one
    # channel which had a large bleed for 1 minute.     
    signalMZbc = signalMZ.ravel() - np.transpose(baselineMZ)
    
    # check to see if the signal is greater than noise or not
    # if normaliseData == 2
    if sigThresholdNormFactor:
        sigAboveNoiseLogical = (signalMZbc > sigThresholdNorm)
    else:
        sigAboveNoiseLogical = (signalMZbc > sigThreshold)
    
    totalSigAboveNoise = sum(sigAboveNoiseLogical)
    if totalSigAboveNoise > 5:
        resPeak = GCPeakDetection(signalMZbc, timesteps, splitSignalSize,
                                 degPoly, thresholdSD, thresholdFD, 
                                 thresholdSignal, splitCoelution, showPlot, max(signalMZbc))
