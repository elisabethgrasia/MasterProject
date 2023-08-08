"""
Created on Sat Mar 25 23:13:01 2023

@author: alvak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import scipy
import GCPeakDetection
from scipy.linalg import lstsq
from scipy.linalg import cholesky
from scipy.sparse import spdiags
from scipy import sparse

# exporting the data p.falciparum CHMI trial
## the data is divided into ambient and breath sample, 
## however we tried to explore the ambient first

df = pd.read_csv('peak_alignment_csv.csv', header = None)
df.head()
df.shape

# Parameters setting
splitSignalSize = 100
degPoly = 3
thresholdSD = [0, 10]
thresholdFD = [0, 5]
sigThreshold = 2e4
thresholdSignal = [1, sigThreshold]
splitCoelution = False
showPlot = True

showPlotMZ = True
showPlotMain = True

### Results column
colPeakStart = 1
colPeakEnd = 2
colPeakMax = 3
colPeakHeight = 4
colPeakArea = 5

massZ = df.iloc[0, 1:]
dataRaw = df.iloc[1:, ]
dataRaw = dataRaw.T

# minimizing the dataRaw for code testing purposes,
# use all the spectra, but only use 5 masses
dtR = dataRaw.iloc[:, 0:5]

Nmz = len(massZ)

## Initialise the results
resMSPeak = np.array((Nmz, 1), list)
peakData = np.array((Nmz, 1), list)

## Separate the raw data into time stamps and intensity
mzDataExpt = dataRaw.iloc[1:, ]    #dtR instead of dataRaw for data increment
mzDataExpt = mzDataExpt.to_numpy()
timesteps = dataRaw.iloc[0, :]  # dtR instead of dataRaw for data increment


## If we want to normalise the data
#if normaliseData == 2:
#    data = df.loc[:, (df > 0).all()]     
#    meanBlankSample = data.mean()
#    mzDataExpt = mzDataExpt / meanBlankSample
#elif normaliseData == 3:
    
 #   mzDataExpt = mzDataExpt / meanAllSamples
    
# redefine the signal threshold using the overall noise
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
    signalMZ = mzDataExpt[0, ].reshape((-1, 1))
    if signalMZ.max() <= 10:
        lambdaALS = 2e3
    else: 
        lambdaALS = 5e3
        
pALS = 1e-4
    
## Find baseline
def baseCorrAls(y, l, p):
    m = len(y)
    D = np.transpose(np.diff(sparse.eye(m).toarray(), 2))
    w = np.ones((m, 1))
    for i in range(10):
       wt = np.transpose(w)
       W = spdiags(wt, 0, m, m).toarray()
       Dt = np.transpose(D)
       C = cholesky(W + l  * np.matmul(Dt, D))
       Ct = np.transpose(C)
       wy = np.multiply(wt, y)
       z = lstsq(C, (lstsq(Ct, wy)[0]))[0]
       w = p * (y > z) + (1 - p) * (y < z)
       w = w[:, 0]
    return z[:, 0]

baselineMZ = baseCorrAls(signalMZ, lambdaALS, pALS)

fig = plt.figure()
fig.canvas.manager.set_window_title('Figure 4')
plt.clf()
plt.plot(timesteps, signalMZ, 'b', linewidth = 5, label = 'signalMZ')
plt.plot(timesteps, baselineMZ, 'r', markersize = 8, label='Baseline')
plt.title(r'M/Z')
plt.legend(loc='lower left')
plt.show()
    
## taking baseline drift off 
# to account for one channel which had a large bleed for 1 minute

signalMZbc = signalMZ - (np.transpose(baselineMZ))

# Check to see if the signal is greater that noise or not 
# If normaliseData == 2
if np.count_nonzeros(sigThresholdNormFactor) == 0 :
    sigAboveNoiseLogical = (signalMZbc > sigThresholdNorm)
else:
    sigAboveNoise =sum(signalMZbc > sigThreshold)
    
totalSigAboveNoise = sum(sigAboveNoiseLogical)
if totalSigAboveNoise > 5:
    print('-----------------\n')
    print('Now doing peak detection for M/Z = %d\n', massZ(mzi))
    resMSPeak = GCPeakDetection(signalMZbc, timesteps,
                                     splitSignalSize, degPoly, thresholdSD,
                                     thresholdFD, thresholdSignal, 
                                     splitCoelution, showPlot, max(signalMZbc))
    
    # Total Peaks found
    NPeaksTotal = resMSPeak[mzi][0]
    
    ## If the total number of peak is not 0.
    ## then save this result
    if NPeaksTotal > 0:
        # need to map the detected peaks into a vector of the same size original data
        # that is all zeros exept at peak maxima with values of the peak area
        peakData[mzi] = np.zeros(len(timesteps), 1)
        for npeak in range(NPeaksTotal):
            # find the index in the timesteps with the peak maximum time
            indPeakMax = np.where(timesteps == resMSPeak[mzi](npeak, colPeakMax))
            peakData[mzi][indPeakMax] = resMSPeak[mzi](npeak, colPeakArea)
     
    
    # Plot resulting peaks
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Figure 5')
        plt.clf()
        plt.plot(timesteps, signalMZ, 'b', linewidth = 1)
        plt.grid()
        plt.plot(resMSPeak[mzi][:, ], np.zeros(NPeaksTotal, 1), 'm^', )
        plt.plot(resMSPeak[mzi][:, colPeakEnd], np.zeros(NPeaksTotal, 1), 'rv')
        plt.plot(resMSPeak[mzi][:, colPeakMax], np.zeros(NPeaksTotal, 1), 'go')
        plt.title()
        
        plt.show()        
        
        # save the full results
       # if flagSave == True:
        #    fnameSaveFull = [dirSave, '0fullResults/', fname(1:end-4), '.mat'];
         #   save(fnameSaveFull, 'resMSPeak', 'peakData', 'dataRaw', 'massZ');
   
   
        # put some lines and space in the command window output
        #print('\n\n\n');
        #print('Finished peak detection of all mass/z for %s\n', #name);
        #print('Total number of peaks found over all mass was %d\n', NPeaksOverAllMz)
        #print('\n=========================================================');
        #print('\n\n\n');
   

   
        
                              
    
    



