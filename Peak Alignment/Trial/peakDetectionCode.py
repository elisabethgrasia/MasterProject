"""
Created on Sat Mar 25 23:13:01 2023

@author: alvak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import utils
from scipy.linalg import lstsq
from scipy.linalg import cholesky
from scipy.sparse import spdiags, diags, eye
from itertools import chain
import sys
import time
import os
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
resMSPeak = []
peakData = []
NPeaksOverAllMz = 0

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
    

start = time.time()

for mzi in range(Nmz):
    start_time = time.time()
    signalMZ = mzDataExpt[mzi, :].reshape((-1, 1))
    if signalMZ.max() <= 10:
        lambdaALS = 2e3
    else: 
        lambdaALS = 5e3
        
    pALS = 1e-4
    
    baselineMZ = utils.baseCorrAls(signalMZ, lambdaALS, pALS)
    # to save time, we save the baseline into .mat variable, thus
    # we do not have look for the baseline everytime we run the code
    #filename = 'my_file{}.mat'.format(massZ[mzi+1])
    
    #if os.path.isfile(filename):
     #   import scipy.io as sio
        # Load the .mat file
      #  data = sio.loadmat(filename)
        # Extract the variable from the data dictionary
       # baselineMZ = np.squeeze(data['my_variable'])
        
    #else:
     #   baselineMZ = utils.baseCorrAls(signalMZ, lambdaALS, pALS)
      #  import scipy.io as sio
        # Save the variable in a .mat file
       # sio.savemat('my_file{}.mat'.format(massZ[mzi+1]), {'my_variable': baselineMZ})
    
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
        plt.show()
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
        
        print('-------------')
        print(f'Now doing peak detection for M/Z = {massZ[mzi+1]}')
        resPeak = GCPeakDetection(signalMZbc, timesteps, splitSignalSize,
                                         degPoly, thresholdSD, thresholdFD, 
                                         thresholdSignal, splitCoelution, showPlot, max(signalMZbc))
        resMSPeak.append(resPeak)
        
        # total Peaks found
        NPeaksTotal = resMSPeak[mzi].shape[0]
        
        # if the total number of peaks isn't 0, then save this result
        if NPeaksTotal > 0:
            # need to map the detected peaks into a vector of the same size as the
            # original data, that's all zeros except at peak maxima with values of
            # the peak area
            peakData.append(np.zeros(len(timesteps)))
            for npeak in range(NPeaksTotal):
                # find the index in the timestep with the peak maximum time
                indPeakMax = np.where(timesteps == (resMSPeak[mzi][npeak][colPeakMax-1]))[0][0]
                peakData[mzi][indPeakMax] = resMSPeak[mzi][npeak][colPeakArea-1]
            
            # save the peaks for this mz into the appropriate folder
            #if flagSave:
                #fnameSaveMZ = '{}mz{}/{}'.format('...\project-elisabethgrasia\Peak Alignment', massZ[mzi], fname)
                #np.savetxt(fnameSaveMZ, peakData[mzi], delimiter=",")
                           
        if showPlotMain:
           # plot all the peaks on the full signal
            hFigMain = plt.figure(5)
            plt.clf()
            plt.plot(timesteps, signalMZ, 'b', linewidth=1)
            plt.grid(True)
            plt.plot(resMSPeak[mzi][:, colPeakStart-1], np.zeros(NPeaksTotal), 'm^', markersize = 8, markerfacecolor='m')
            plt.plot(resMSPeak[mzi][:, colPeakEnd-1], np.zeros(NPeaksTotal), 'rv', markersize = 8, markerfacecolor='r')
            plt.plot(resMSPeak[mzi][:, colPeakMax-1], np.zeros(NPeaksTotal), 'go', markersize = 8, markerfacecolor='g')
            strTitle = 'M/Z = {}'.format(massZ[mzi+1])
            plt.title(strTitle, fontsize = 12, fontweight = 'bold')
            plt.show()
            #print('press any key to continue......')
            #input()
 
       
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Execution time:", execution_time)            
    NPeaksOverAllMz = NPeaksOverAllMz + NPeaksTotal
    print("Total peaks found so far:", NPeaksOverAllMz)
    # keep a running total of peaks found for this sample

end = time.time()
        
# put some lines and space in the command window output
print('\n\n\n')
#print(f'Finished peak detection of all mass/z for {fname}')
print(f'Total number of peaks found over all mass was {NPeaksOverAllMz}')
print('\n=========================================================')
print('\n\n\n')
print("Total time for this peak detection code:", end-start)
print('\n=========================================================')
print('\n\n\n')


# if we don't want to see the peak results, then output the time of start
# and end of this script
if not showPlotMain and not showPlot:
    print('\n\nAll files processed. \n\n')
    timeCurrent = time.localtime()
    print(f'Current time: {timeCurrent.tm_mon}-{timeCurrent.tm_mday} {timeCurrent.tm_hour}:{timeCurrent.tm_min}:{timeCurrent.tm_sec:.1f} \n\n')
    