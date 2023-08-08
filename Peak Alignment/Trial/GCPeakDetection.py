# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:57:37 2023

@author: alvak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from scipy.signal import savgol_filter

# Aiming to find 1D Chromatogram peaks using Savitzky-Golay smoothing

def GCPeakDetection(gcData, timesteps, splitSignalSize,
                    degPoly,thresholdSD, thresholdFD, thresholdSignal, splitCoelution,
                    showPlot, maxYLimPlot):
    """
    Arguments:
        - gcData : a vector containing the values of GC measurements
        - timesteps : a vector of equal length to gcData, containing the elution times of the recording. 
                        If this input is not given or is empty, then the default is time index.
        - splitSignalSize : 0 if not splitting the input signal into smaller chunks for peak detection,
                            default is 100 (input is empty).
        - degPoly : the degree of polynomial for the SG smoothing, default is 3.
        - thresholdSD : threshold for the second derivative, any SD below this threshold is considered a candidate
                        for peak. The input should be in a vector two values: [v1, v2], where
                        v1 indicates whether it's a absolute value threshold (1) or a relative value to the noise (0).
                        v2 is a value for the actual threshold. Default is [0 5], ie 5 times the noise.'
        - thresholdFD : threshold for the first derivative, where we consider the start and end of any peak.
                        The input is in the same format as threshold SD. Default is [0,2]. Note [1, 0.05] seems to be
                        reasonable for normalised data.
        - splitCoelution : true if want to split the strong co-elution peaks, use with caution. Default if false.
        - showPlot : true if want to see the peak detection in action. Default is false.
        - maxYLimPlot : value for the maximum y-axis for the plot. Default is min(1000, max(gcData))
        
    Returns:
        - resPeaks : a mx5 matrix of m peaks found in the input data, each row is in the order of
                    [peak start; peak end; peak maximum; peak height; peak area], the first 3 values are either time
                    index or elution time depending on whether the second input is given or not.
    """
    
    # Initialisation, checking input data and setting default values
    if gcData.ndim > 1:
        raise ValueError('data input should be a 1D vector.')
            
    N = len(gcData)
        
    if timesteps is None:
        timesteps = list(range(1, N+1))
    if timesteps.ndim > 1 and timesteps.size != N:
        raise ValueError("time step input should be a 1D vector.")
    elif len(timesteps) != N:
        raise ValueError('Input data and time steps do not match in size.')
        
    if splitSignalSize is None or splitSignalSize == 0:
        splitSignalSize = 100
    if splitSignalSize > N:
        splitSignalSize = 0 # use the whole signal
        
    if degPoly is None:
        degPoly = 3
        
    if thresholdSD is None:
        # default is 5 * noise of SD
        thresholdSD = [0, 5]
    elif len(thresholdSD) == 2:
        # check to make sure the input threshold values are valid
        if not(thresholdSD[0] == 0 or thresholdSD[0] == 1):
            raise ValueError("Threshold for second derivative input 1 should be 0 or 1")
    else:
        raise ValueError("Threshold for second derivative not valid format.")
            
    if thresholdFD is None:
        # default is 2 * noise of FD
        thresholdFD = [0, 2]
    elif len(thresholdFD)==2:
        # check to make sure the input threshold values are valid
        if not(thresholdFD[0] != 0 or thresholdFD[0] != 1):
            raise ValueError("Threshold for first derivative first inpyt should be 0 or 1")
    else:
        raise ValueError("Threshold for first derivative not valid format.")
            
    if thresholdSignal is None:
        # default is baseline + 2 * noise
        thresholdSignal = [0, 2]
    elif len(thresholdSignal)==2:
        # check to make sure the input threshold values are valid
        if not(thresholdSignal[0] != 0 or thresholdSignal[0] != 1):
            raise ValueError("Threshold for signal input 1 should be 0 or 1")
    else:
        raise ValueError("Threshold for signal not valid format.")
            
    if splitCoelution is None:
        splitCoelution = False
        
    if showPlot is None:
        showPlot = False
        
    if showPlot is None:
        if max(gcData) > 1000:
            yULim = 1000
        else:
            yULim = max(gcData)
    elif not showPlot:
        yULim = max(gcData)
    else:
        yULim = maxYLimPlot
    

    # ++++++++++++++++ Initialise the output matrix ++++++++++++++
    resPeaks = np.zeros((0, 5))
    colPeakStart = 0
    colPeakEnd = 1
    colPeakMax = 2
    colPeakHeight = 3
    colPeakArea = 4
    
    if showPlot:
        hFig = plt.figure(num=1)
        plt.plot(timesteps, gcData, linewidth= 1)
        plt.ylim([0, yULim])
        plt.grid(True)
        plt.show()
        
    # =================================================================================
    # Step 1: divide the input signal into manageable chunks
    # =================================================================================
    ## run an initial SG smoothing to find signal's noise
    ## divide the signal into roughly size of splitSignalSize but make sure that no actual signal is divided up
        
    ### Initial smoothing (window size = 5)
    smdInitial = savgol_filter(gcData, window_length=5, polyorder=degPoly, deriv=0)
    
    ### Noise calculation
    # turns out using the noise criterion defined in the calcNoise function
    # below doesn't work well here, so just going to use the median value
    
    noiseSmdInitial = utils.calcNoise(smdInitial)
    baselineSmdInitial = np.median(smdInitial)
    if not thresholdSignal[0]:
        thrSig = baselineSmdInitial + thresholdSignal[1] * noiseSmdInitial
    else:
        thrSig = thresholdSignal[1]
        
    ### Find all indices where the smoothed signal is less than noise x 2
    ### indNoiseSmdInitial = np.where(smdInitial < thrSig)[0]
    
    # plot the threshold for noise
    if showPlot:
        plt.figure(hFig)
        plt.plot(timesteps, smdInitial, 'r:', linewidth=1)
        plt.plot([timesteps[1], timesteps[14893]], [thrSig, thrSig], 'b--', linewidth=1)
        plt.show()
        
    ### Find all the partitions, save the start of partition indices
    if splitSignalSize == 0:
        # TODO: make sure this bit makes sense.
        partStIndex = 0
        partEndIndex = N - 1
    else:
        partStIndex, partEndIndex = utils.partitionSignal(smdInitial, thrSig, N, splitSignalSize)
        
        
    ### Total number of partitions
    NPartition = len(partStIndex)

    ### Plot these partitions out
    if showPlot:
        plt.figure(hFig)
        for i in range(len(partStIndex)):
            plt.plot([timesteps[partStIndex[i]], timesteps[partStIndex[i]]], [0, yULim], 'k--')
        plt.show()
    
    # if the total number of partition found is too small, i.e. the signal
    # threshold was set too low, then we need to reset that. 
    #  nb. this isn't in the referenced paper either. 
    if splitSignalSize and NPartition <= 3:
        # if the threshold was set to a certain value, then use noise as the
        # threshold, if we used noise as a threshold, then double the threshold value
        if thresholdSignal[0] == 1:
            thresholdSignal = [0, 2]
        elif thresholdSignal[0] == 0:
            thresholdSignal[1] = np.matmul(thresholdSignal[1], 2)
        else:
            raise ValueError("Threshold for signal input 1 should be 0 or 1")
        
        thrSig = baselineSmdInitial + np.matmul(thresholdSignal[1], noiseSmdInitial) 
                                                 
        # recalculate the partitions
        partStIndex, partEndIndex= utils.partitionSignal(smdInitial, thrSig, N, splitSignalSize)
        NPartition = len(partStIndex)
        
        # replot if the flag was on
        if showPlot:
            plt.figure(hFig)
            for i in range(len(partStIndex)):
                plt.plot([timesteps[partStIndex[i]], timesteps[partStIndex[i]]], [0, yULim], 'k--', linewidth=1)
            plt.show()
            
   # if showPlot:
       # print('Chromatogram split into smaller sections.')
       # input('Press any key to continue...\n')
        
    # ===================================================================================    
    ### Find peaks for each partition of the data
    NPeaksTotal = 0
    for partNumb in range(1, NPartition+1):
        # get the data associated with this section
        dataPartOrig = gcData[(partStIndex[partNumb-1]-1) : partEndIndex[partNumb-1]]
        smdInitPart = smdInitial[(partStIndex[partNumb-1]-1) : partEndIndex[partNumb-1]]
        NDataPart = len(dataPartOrig)
        timePart = timesteps[(partStIndex[partNumb-1]-1) : partEndIndex[partNumb-1]]
        
        # If any signal in this partition is larger than the threshold for
        # signal we defined earlier, then we'll check for peaks
        if any(smdInitPart - thrSig > 0):
        
            # ============================================================================
            # === Step 2: Find the best window size for the smoothing ====================
            ## -- search through all odd size between 5 and 41
            ## --  use the Durbin-Watson (DW) Test
            ## -- best window size is one which gives DW closest to 2
        
            maxWinSize = 41
            if partNumb == NPartition and len(dataPartOrig) < maxWinSize+4:
                maxWinSize = len(dataPartOrig) - 5
        
            windowSize = list(range(5, maxWinSize+1, 2))
            valDW = np.zeros((len(windowSize)))
            for wsi in range(len(windowSize)):
                dataPartSmd = savgol_filter(dataPartOrig, window_length=windowSize[wsi], polyorder=degPoly, deriv=0)
                diffSmdOrig = dataPartOrig - dataPartSmd
                valDW[wsi] = np.sum((diffSmdOrig[1:] - diffSmdOrig[:-1]) ** 2) / np.sum(np.array([(x ** 2) for x in diffSmdOrig])) * (NDataPart / (NDataPart - 1))
        
            # Find the difference between the DW values and 2
            diffDWn2 = np.abs(valDW - 2)
            # Find the actual window size
            temp, indWS = np.min(diffDWn2), np.argmin(diffDWn2)
            WS = windowSize[indWS]
                # print(f'window size: {WS}')
                
            # The smoothed data using the discovered window size
            # ZD - zero-th derivative
            smdZD = savgol_filter(dataPartOrig, window_length=WS, polyorder=degPoly, deriv=0)
            
            # plotting the smoothed WS
            if showPlot:
                hFigPart = plt.figure(2)
                hFigPart.clf()
                plt.title(f"Partition Number {partNumb+1}", fontsize=14)
                plt.plot(timePart, dataPartOrig, 'b-', linewidth=2)
                plt.plot(timePart, smdZD, 'r:', linewidth=2)
                if max(dataPartOrig) > yULim:
                    plt.ylim([0, yULim])
                plt.grid(True)
                plt.show()
        
            # =========================================================================
            # ============= Step 3: Peak Detection ====================================
            ## We will only do peak detection of the signal in 
            ## the partition is more than the threshold set
        
        # ---- comment out the previous "if any( > 0)" line and uncomment
        # the following if you want to see what the signal in the partition
        # looks like
        # if any(smdZD - thrSig > 0)
        
        
            ## ===============================================================
            ## ====== Step 3a. Calculate derivatives =========================
        
            ### First derivative
            smdFD = savgol_filter(dataPartOrig, window_length=WS, polyorder=degPoly, deriv=1)
        
            ### Second derivative
            smdSD = savgol_filter(dataPartOrig, window_length=WS, polyorder=degPoly, deriv=2)
        
            ### Third derivative -- only for moderate co-elutions
            smdTD = savgol_filter(dataPartOrig, window_length=WS, polyorder=degPoly, deriv=3)
        
            ### Plot the first and second derivatives
            if showPlot:
                plt.figure(hFigPart)
                plt.plot(timePart, smdFD, 'g--', linewidth=2)
                plt.plot(timePart, smdSD, 'm--', linewidth=2) 
                plt.legend(['original data', 'SG smoothed', 'SG 1st derivative', 'SG 2nd derivative'])
                plt.show()  
        
            ## =======================================================================================
            ## ====== Step 3b. Use noise of the SD to threshold the zones of -ve SD ==================
        
            ### calculate the noise
            noiseSD = utils.calcNoise(smdSD)
    
            ### threshold is either some multiple of the noise or a set value
            if not thresholdSD[0]:
                thrSD = thresholdSD[1] * noiseSD
            else:
                thrSD = thresholdSD[1]
        
            timePart = timePart.values
            ### plot the threshold for SD and the signal
            if showPlot:
                plt.figure(hFigPart)
                plt.plot([timePart[0], timePart[-1]], [-thrSD, -thrSD], 'b--', linewidth=1)
                plt.plot([timePart[0], timePart[-1]], [thrSig, thrSig], '--', label='data2', linewidth=1, color=[0.33,0, 0])
                plt.legend(['original data', 'SG smoothed', 'SG 1st derivative', 'SG 2nd derivative', 'data1', 'data2'], fontsize='x-small')
                plt.show()
    
            ## =======================================================================================
            ## ====== Step 3c. Find the negative zones of second derivative ==========================
        
            ### Find SD less than -thrSD
            indSD = np.where(smdSD < -thrSD)[0]
        
            ### Check if there are any negative zones
            if indSD.any():
                # find, within these indices, the zones of -ve SD, i.e. those with
                # continues indices
                indIndSD_NegZoneEnd = np.concatenate((np.where(np.diff(indSD) > 1)[0], np.array([len(indSD)-1])))
                indIndSD_NegZoneSta = np.concatenate([[0], indIndSD_NegZoneEnd[:-1] + 1])
    
                # get the actual indices of these zones
                indSD_NegZoneEnd = indSD[indIndSD_NegZoneEnd]
                indSD_NegZoneSta = indSD[indIndSD_NegZoneSta]
    
                # number of negative zones found
                nSDNegZones = indSD_NegZoneEnd.size
            else:
                nSDNegZones = 0
        
        
            ## =======================================================================================
            ## ===== Step 3d. Use the negative zones of SD to find the peaks =========================
        
        # don't think this is necessary. 
        ## recalculate the signal threshold for this partition
        ## noiseSmd = calcNoise(smdZD);
        ## baselineSmd = median(smdZD);
        ## if ~thresholdSignal(1)
        ##      thrSig2 = baselineSmd + thresholdSignal(2) * noiseSmd;
        ## else
        ##      thrSig2 = thresholdSignal(2);
    
        # need a threshold for the first derivative for start and end of peak
            noiseFD = utils.calcNoise(smdFD)
            if not thresholdFD[0]:
                thrFD = thresholdFD[1] * noiseFD
            else:
                thrFD = thresholdFD[1]    
    
            # check to make sure this threshold isn't too big 
            # don't think this is wise -- only applicable to cases where the
            # data is normalised. 
            # if thrFD > 0.05:
                #   thrFD = 0.05
                ### plot the threshold for FD
            if showPlot:
                plt.figure(hFigPart)
                plt.plot([timePart[0], timePart[-1]], [thrFD, thrFD], '--', label='data3', linewidth=1, color=[0, 0.33, 0])
                plt.plot([timePart[0], timePart[-1]], [-thrFD, -thrFD], '--', label='data4', linewidth=1, color=[0, 0.33, 0])
                plt.legend(['original data', 'SG smoothed', 'SG 1st derivative', 'SG 2nd derivative', 'data1', 'data2', 'data3', 'data4'], bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize='x-small')
                plt.show()
    
            # need change of sign in FD, for checking end of the peak
            # this isn't in the ref'd paper.
            changeInSignFD = abs(np.diff(smdFD > 0))
    
            # initialise peak results
            numbPeaksPart = 0           # number of peaks found in this partition
            resPeaksPart = np.zeros((150, 5))
        
            for nZone in range(nSDNegZones):
                # first check that the ZD value with the minimum SD is actually
                # above the threshold set for a valid signal 
                # [temp, indMinSD] = min(smdSD(indSD_NegZoneSta(nZone):indSD_NegZoneEnd(nZone)) )
                # indPeakMax = indSD_NegZoneSta(nZone)+indMinSD-1; 
    
                # get the ZD values within the negative zone
                smdZDinNegZone = smdZD[indSD_NegZoneSta[nZone]:indSD_NegZoneEnd[nZone]+1]
            
                # the peak is the maximum ZD value
                temp, indMaxZD = np.max(smdZDinNegZone), np.argmax(smdZDinNegZone)
                indPeakMax = indSD_NegZoneSta[nZone] + indMaxZD
                
                # also check the first derivative in this zone is above the threshold FDd
                smdFDinNegZone = smdFD[indSD_NegZoneSta[nZone]:indSD_NegZoneEnd[nZone]+1]
                
                # check to make sure that the previous peak doesn't end after
                # the current peak maximum. if so, then we need to do something
                # about it in the peak detection loop. 
                # -- this isn't in the ref'd paper. 
                prevPeakEndB4CurrPeak = False
                if numbPeaksPart == 0:
                    prevPeakEndB4CurrPeak = True
                elif timePart[indPeakMax] > resPeaksPart[numbPeaksPart-1, colPeakEnd]:
                    prevPeakEndB4CurrPeak = True
                        
                
                # conditions for considering this zone: 
                # -- the smoothed signal in this zone above thrZD, and 
                # -- any abolute value of FD in this zone is above thrFD.
                # else go to the next zone
                # -- if the length of the zone is greater than one (not in ref'd paper)
                if (smdZDinNegZone > thrSig).any() and (abs(smdFDinNegZone) > thrFD).any() and (len(smdZDinNegZone) > 1):
                    # increment the count of peak numbers in this partition
                    numbPeaksPart += 1
                    
                    # we can write down the time index for the peak maximum and its value
                    resPeaksPart[numbPeaksPart-1, colPeakMax] = timePart[indPeakMax]
                    resPeaksPart[numbPeaksPart-1, colPeakHeight] = smdZD[indPeakMax]
                    
                    # if previous peak's end is after the current peak maximum,
                    # then we need to change the peak ending time for the
                    # previous peak
                    if not prevPeakEndB4CurrPeak:
                        # get the index for the end of the previous negative SD zone. 
                        indEndPrevSD = indSD_NegZoneEnd[nZone]
                        
                        # starting from this index, we search in the third
                        # derivative for the change in sign between two time indices. 
                        indTemp = indEndPrevSD
                        while True:
                            if indTemp > indSD_NegZoneSta[nZone]:
                                indEndPrevSD = indTemp - 1
                                break
                            elif np.diff((smdTD[indTemp:indTemp+2] > 0)) == 0:
                                indTemp += 1
                            else:
                                indEndPrevSD = indTemp+1
                                break
                            
                        resPeaksPart[numbPeaksPart-2, colPeakEnd] = timePart[indEndPrevSD]
            
                    
                    # search in the first derivative for start of peak
                    indFD_startOfPeak = indSD_NegZoneSta[nZone]
                    while True:
                        # if the current value is at the very first index of
                        # the partition, then we've found peak
                        if indFD_startOfPeak == 0:
                            break
                        
                        # if the current value and the previous value have
                        # different signs, then we've found start of peak 
                        if changeInSignFD[indFD_startOfPeak-1]:
                            break
                        
                        # use absolute value of the FD, for those cases where
                        # the FD at the beginning of the zone isn't positive
                        if abs(smdFD[indFD_startOfPeak]) > thrFD:
                            # go to the previous FD value
                            indFD_startOfPeak -= 1
                        else:
                            # found the start of the peak
                            break
                        
                        # also check that the previous value is: 
                        # -- the very first value of the partition, or 
                        # -- the same as the last peak's end.
                        # if so, start found.
                        if indFD_startOfPeak == 0:
                            break
                        elif numbPeaksPart>1 and timePart[indFD_startOfPeak] == resPeaksPart[numbPeaksPart-1, colPeakEnd]:
                            break
                    
                
                    #  write down the start of the peak 
                    resPeaksPart[numbPeaksPart-1, colPeakStart] = timePart[indFD_startOfPeak]
                    
                    # search in the first derivative for the end of the peak
                    indFD_endOfPeak = indSD_NegZoneEnd[nZone]
                    while True:
                        # also check that the current value is not the very 
                        # last value of the partition, if so, end found.
                        if indFD_endOfPeak == NDataPart-1:
                            break
                        
                        # if the current value and the next value have
                        # different signs, then we've found peak end
                        if changeInSignFD[indFD_endOfPeak]: 
                            break
                            break
                        # absolute value of the FD above the threshold, for
                        # those cases that the FD at the end of the zone isn't
                        # negative.
                        if abs(smdFD[indFD_endOfPeak]) > thrFD:
                            # go to the next FD value
                            indFD_endOfPeak += 1
                        else:
                            # found the end of the peak
                            break;
                        
                    # write down the end of the peak 
                    resPeaksPart[numbPeaksPart-1, colPeakEnd] = timePart[indFD_endOfPeak]
                    
                    ## =======================================================================================
                    ## ===== Step 3e. 
                    ## Check if the SD in the negative zones has more than 1 dip,
                    ## if so then we have a case of strong co-elusion, and 
                    ## require a split of this peak into the separate peaks
        
                    ### First calculate the number of change of sign in the third derivative
                    ### i. Get the third derivative values in the negative zone
                    smdTDinNegZone = smdTD[indSD_NegZoneSta[nZone]:indSD_NegZoneEnd[nZone]+1]
        
                    ### ii. Get the logical index of TD values greater than 0
                    ###     (+ve and -ve values of TD), any difference between consequent
                    ###     indices that is not 0 shows there is a change in sign
                    indChangeInSignTD = np.where(np.diff(smdTDinNegZone > 0))[0]
        
                    ### iii. Count the number of change is sign
                    nChangeInSignTD = len(indChangeInSignTD)
        
                    ### iv. Calculate the number of minimum in the SD
                    if np.mod(nChangeInSignTD, 2):
                        # if n is odd
                        nDipsInSD = (nChangeInSignTD + 1) / 2
                    else:
                        # if n is even
                        # nb. don't know how this could happen but I'll leave
                        # this in here
                        nDipsInSD = nChangeInSignTD / 2
        
                    
                    # If there are more than one dips in the SD -ve zone, then
                    # we'll need to split the current peak
                    if nDipsInSD > 1 and splitCoelution:
                        # set aside another matrix for these results
                        multiPeakZoneRes = np.zeros((nDipsInSD, 5))
            
                        # the start and end of the peak region now goes into
                        # the start of peak 1 and end of last peak
                        multiPeakZoneRes[0, colPeakStart] =  resPeaksPart[numbPeaksPart, colPeakStart]
                        multiPeakZoneRes[nDipsInSD, colPeakEnd] = resPeaksPart[numbPeaksPart, colPeakEnd]
            
                        # keep track the indices for the start and end
                        mpi1 = np.zeros((nDipsInSD, 1))
                        mpi2 = np.zeros((nDipsInSD, 1))
                        mpi1[0] = indFD_startOfPeak
                        mpi2[-1] = indFD_endOfPeak
        
                        ### v. Find all the minimum SD in the -ve zone, these will be the peak
                        ###     maxima, this when the TD change from -ve to +ve
                        indMinsSD = np.where(np.diff(smdTDinNegZone > 0) == 1)[0]
                        indPeakMaxInMultiZone = indSD_NegZoneSta[nZone] + indMinsSD-1
                        multiPeakZoneRes[:, colPeakMax] = np.transpose(timePart[indPeakMaxInMultiZone])
                        multiPeakZoneRes[:, colPeakHeight] = np.transpose(smdZD[indPeakMaxInMultiZone])
                            
                        ### vi. When the TD change from +ve to -ve that is when the split should be
                        indMaxsSD = np.where(np.diff(smdTDinNegZone > 0) == -1)[0]
                        indPeakSplit = indSD_NegZoneSta[nZone] + indMaxsSD - 1
                        if not np.mod(nChangeInSignTD, 2):
                                # if n is even, then we need to make sure the split happens after the peak
                                indPeakSplit_indActual = np.where(indPeakSplit > indPeakMaxInMultiZone[0])[0]
                                indPeakSplit = indPeakSplit[indPeakSplit_indActual]
                            
                        multiPeakZoneRes[0:nDipsInSD-1, colPeakEnd] = np.transpose(timePart[indPeakSplit])
                        multiPeakZoneRes[1:nDipsInSD, colPeakStart] = np.transpose(timePart[indPeakSplit])
                        mpi1[1:nDipsInSD] = indPeakSplit
                        mpi2[0:nDipsInSD-1] = indPeakSplit
                            
            
                        ## =======================================================================================
                        ## ===== Step 3f. a. Calculate the area under the curve
                        for nDip in range(nDipsInSD):
                            multiPeakZoneRes[nDip, colPeakArea] = utils.calcAUC(smdZD[mpi1[nDip]:mpi2[nDip]],timePart[mpi1[nDip]:mpi2[nDip]])
                            
                            #write the whole temporary matrix into the matrix for
                            #this partition's result
                            resPeaksPart[numbPeaksPart:numbPeaksPart+nDipsInSD-1, :] = multiPeakZoneRes
                            
                            #update the number of peaks found in this partition
                            numbPeaksPart = numbPeaksPart + nDipsInSD - 1
                    else:        
                            # if there's only one dip
                            
                    ## =======================================================================================
                    ## ===== STEP 3f. b. calculate the area under the curve
                       # first check that the peak is not just one point, if
                       # it is, then delete the current row, and decrement the count
                       if len(smdZD[indFD_startOfPeak:indFD_endOfPeak+1]) > 1:
                           areaPeak = utils.calcAUC(smdZD[indFD_startOfPeak:indFD_endOfPeak+1], timePart[indFD_startOfPeak:indFD_endOfPeak+1])
                           # write down the area under the peak
                           resPeaksPart[numbPeaksPart-1, colPeakArea]= areaPeak
                       else:
                           resPeaksPart[numbPeaksPart, :] = []
                           numbPeaksPart = numbPeaksPart - 1
                     
    
                # end if : the -ve zone is valid for peak
                
                # end for : all the -ve zones in this partition
            
            # plot all the peaks in this partition on the figure
            if showPlot:
                indRow = np.where(np.any(resPeaksPart > 0, axis = 1))[0]
                if numbPeaksPart > 0:
                    plt.figure(hFigPart)
                    plt.plot(resPeaksPart[indRow, colPeakStart], np.zeros(numbPeaksPart),'m^', markersize=12, markerfacecolor= 'm')
                    plt.plot(resPeaksPart[indRow, colPeakEnd], np.zeros(numbPeaksPart), 'rv', markersize=12, markerfacecolor= 'r')
                    plt.plot(resPeaksPart[indRow, colPeakMax], np.zeros(numbPeaksPart), 'go', markersize=12, markerfacecolor= 'g')
                    plt.show()
                else:
                    plt.figure(hFigPart)
                    plt.show()
            
            # Output number of peaks found in the partition
            print(f'Partition number {partNumb}, number of peaks found: {numbPeaksPart}.   ', end='')
            #if showPlot:
            #    input('Press any key to continue. \n')
            #else:
            #    print()
            
            # update peaks for the full chromatogram
            NPeaksTotal = NPeaksTotal + numbPeaksPart
            if resPeaksPart[0, 0] != 0:
                indRow = np.where(np.any(resPeaksPart > 0, axis = 1))[0]
                resPeaks = np.vstack((resPeaks, resPeaksPart[indRow]))
            else:
                pass
            
            # end of peak detection for each partition
        # analysed all data presented. 
    
    if showPlot:
        # plot all the peaks on the full signal
        plt.figure(hFig)
        plt.plot(timesteps, gcData, linewidth=1)
        plt.ylim([0, yULim])
        plt.grid(True)
        plt.plot(resPeaks[:, colPeakStart], np.zeros(NPeaksTotal), 'm^', markersize=8, markerfacecolor='m')
        plt.plot(resPeaks[:, colPeakEnd], np.zeros(NPeaksTotal), 'rv', markersize=8, markerfacecolor='r')
        plt.plot(resPeaks[:, colPeakMax], np.zeros(NPeaksTotal), 'go', markersize=8, markerfacecolor='g')
        plt.show()
    
    print('------------------')
    print('Total number of peaks found:', NPeaksTotal)
    
    return resPeaks

    
        
        
        