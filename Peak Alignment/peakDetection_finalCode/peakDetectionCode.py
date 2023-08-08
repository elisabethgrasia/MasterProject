"""
Created on Sat Mar 25 23:13:01 2023

@author: alvak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from baseCorrALS import baseCorrAls
import scipy
from scipy.linalg import lstsq
from scipy.linalg import cholesky
from scipy.sparse import spdiags, diags, eye
from itertools import chain
import sys
import time
from GCPeakDetection import GCPeakDetection


# exporting the data p.falciparum CHMI trial
## the data is divided into ambient and breath sample, 
## however we tried to explore the ambient first

df = pd.read_csv('peak_alignment_csv.csv', header = None)
df.head()
df.shape

#================================================================
#====================== Parameters setting ======================
splitSignalSize = 100
degPoly = 3
thresholdSD = [0, 10]
thresholdFD = [0, 5]
sigThreshold = 2e4
thresholdSignal = [1, sigThreshold]
splitCoelution = False
showPlot = True

# flag to save result
# -- turn this off if we want to look at data and results as we go.
flagSave = False
# other show plot flags
showPlotMZ = True   # ind ms data and baseline
showPlotMain = True # result of each peak detection

# write all command window output to a text file
if flagSave:
    f = open('...\project-elisabethgrasia\Peak Alignment' + 'allScreenOutputs.txt', 'w')
    sys.stdout = f

if showPlotMZ:
    hFigMZ = plt.figure(4)
    props = {'LineWidth': 1}
    propsFont = {'FontSize': 12, 'FontWeight': 'bold'}
    
if showPlotMain:
    hFigMain = plt.figure(5)
    props = {'LineWidth': 1}
    propsMarker = {'MarkerSize': 8}
    propsFont = {'FontSize': 12, 'FontWeight': 'bold'}
    
# results column
colPeakStart = 1   
colPeakEnd = 2
colPeakMax = 3
colPeakHeight = 4
colPeakArea = 5

if not showPlotMain and not showPlot:
    timeCurrent = time.localtime()
    print(f'Current time: {timeCurrent.tm_mon} - {timeCurrent.tm_mday}{timeCurrent.tm_hour}:{timeCurrent.tm_min}:{timeCurrent.tm_sec:.1f} \n\n')
#====================================================================
#======================= Extract the Data ===========================
massZ = df.iloc[0, 1:]
dataRaw = df.iloc[1:, ]
dataRaw = df.iloc[1:, ].T

# minimizing the dataRaw for code testing purposes,
# use all the spectra, but only use 5 masses
# dtR = dataRaw.iloc[:, 0:5]


Nmz = len(massZ)

## Initialise the results
resMSPeak = [[]] * Nmz
peakData = [[]] * Nmz

## Separate the raw data into time stamps and intensity
mzDataExpt = dataRaw.iloc[1:, ]
mzDataExpt = mzDataExpt.to_numpy()
timesteps = dataRaw.iloc[0, :]

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
    signalMZ = mzDataExpt[0, :].reshape((-1, 1))
    if signalMZ.max() <= 10:
        lambdaALS = 2e3
    else: 
        lambdaALS = 5e3
        
pALS = 1e-4

baselineMZ = baseCorrAls(signalMZ, lambdaALS, pALS)


#import pickle
#pickle.dump(baselineMZ, open("baselineMZ", "wb"))

# showing the plot
if showPlotMZ:
    plt.clf()
    plt.plot(timesteps, signalMZ, 'b', linewidth = 5)
    plt.plot(timesteps, baselineMZ, 'r')
    strTitle = 'M/Z = %d' % massZ[mzi]
    plt.title(strTitle, **{'fontsize': 12, 'fontweight': 'bold'})
    plt.show()
    print('press any key to continue......')
    input()
   
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
    print('-------------')
    print(f'Now doing peak detection for M/Z = {massZ[mzi]}')
    resMSPeak[mzi] = GCPeakDetection(signalMZbc, timesteps, splitSignalSize,
                                     degPoly, thresholdSD, thresholdFD, 
                                     thresholdSignal, splitCoelution, showPlot, max(signalMZbc))
    
    # total Peaks found
    NPeaksTotal = resMSPeak[mzi].shape[0]
    
    # if the total number of peaks isn't 0, then save this result
    if NPeaksTotal > 0:
        # need to map the detected peaks into a vector of the same size as the
        # original data, that's all zeros except at peak maxima with values of
        # the peak area
        peakData[mzi] = np.zeros(len(timesteps))
        for npeak in range(NPeaksTotal):
            # find the index in the timestep with the peak maximum time
            indPeakMax = np.where(timesteps == resMSPeak[mzi][npeak, colPeakMax])[0][0]
            peakData[mzi][indPeakMax] = resMSPeak[mzi][npeak, colPeakArea]
        
        # save the peaks for this mz into the appropriate folder
        #if flagSave:
            #fnameSaveMZ = '{}mz{}/{}'.format('...\project-elisabethgrasia\Peak Alignment', massZ[mzi], fname)
            #np.savetxt(fnameSaveMZ, peakData[mzi], delimiter=",")
            if showPlotMain:
    
    # plot all the peaks on the full signal
                plt.figure(hFigMain)
                plt.clf()
                plt.plot(timesteps, signalMZ, 'b', linewidth=1)
                plt.grid(True)
                propsMarker = {'MarkerSize': 8}
                plt.plot(resMSPeak[mzi][:, colPeakStart], np.zeros(NPeaksTotal), 'm^', **propsMarker, MarkerFaceColor='m')
                plt.plot(resMSPeak[mzi][:, colPeakEnd], np.zeros(NPeaksTotal), 'rv', **propsMarker, MarkerFaceColor='r')
                plt.plot(resMSPeak[mzi][:, colPeakMax], np.zeros(NPeaksTotal), 'go', **propsMarker, MarkerFaceColor='g')
                strTitle = 'M/Z = %d' % massZ[mzi]
                plt.title(strTitle, *propsFont)
                print('press any key to continue......')
                input()

# keep a running total of peaks found for this sample
        NPeaksOverAllMz = NPeaksOverAllMz + NPeaksTotal
    
# put some lines and space in the command window output
print('\n\n\n')
#print(f'Finished peak detection of all mass/z for {fname}')
print(f'Total number of peaks found over all mass was {NPeaksOverAllMz}')
print('\n=========================================================')
print('\n\n\n')

# if we don't want to see the peak results, then output the time of start
# and end of this script
if not showPlotMain and not showPlot:
    print('\n\nAll files processed. \n\n')
    timeCurrent = time.localtime()
    print(f'Current time: {timeCurrent.tm_mon}-{timeCurrent.tm_mday} {timeCurrent.tm_hour}:{timeCurrent.tm_min}:{timeCurrent.tm_sec:.1f} \n\n')
    