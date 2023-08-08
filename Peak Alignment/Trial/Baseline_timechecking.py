# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:08:40 2023

@author: alvak
"""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, cholesky, lstsq, lu
from scipy.sparse import spdiags, diags, eye, csc_matrix
import time
from scipy.sparse.linalg import spsolve, splu
import scipy.sparse as sp


# ------------------ dataset ------------------------------------------------
df = pd.read_csv('peak_alignment_csv.csv', header = None)

massZ = df.iloc[0, 1:]
dataRaw = df.iloc[1:, ]
dataRaw = df.iloc[1:, ].T


mzDataExpt = dataRaw.iloc[1:, ]
mzDataExpt = mzDataExpt.to_numpy()
timesteps = dataRaw.iloc[0, :]

Nmz = len(massZ)


# ----- parameter settings ---------------------------------------------------
sigThresholdNormFactor = 10

neighbour1 = mzDataExpt[:, :-2]
neighbour2 = mzDataExpt[:, 2:]
meanNeighbours = (neighbour1 + neighbour2) / 2
diffWithNeighbours = mzDataExpt[:, 1:-1] - meanNeighbours
noiseValMZ = np.median(abs(diffWithNeighbours), axis = 1)
sigThresholdNorm = noiseValMZ.mean() * sigThresholdNormFactor
thresholdSignal = [1, sigThresholdNorm]

# -------------------- baseline analysis ------------------------------------
 
def baseCorrAls(y, l, p):
    start_time = time.time()
    m = len(y)
    D = (np.diff(eye(m).toarray(), 2)).T
    # creating sparse matrix
    # D = sp.csr_matrix(D,  dtype=np.float32)
    D = sp.csc_matrix(D,  dtype=np.float32)
    #end_time0 = time.time()
    #execution_time = end_time0 - start_time
    #print("Execution time diags eye:", execution_time)
    
    # D = spdiags([-1, 2, 1], [0, 1, 2,], [m-2, m]).toarray()
    w = np.ones(m)
    for i in range(10):
        #start_time1 = time.time()
        # W = spdiags(w, 0, m, m)  ## does not work
        
        
        
        # --------------------------- Original Code -----------------
        # W = diags(w, 0, shape=(m, m))
        # DTD = D.T @ D
        # Cholesky factor is divided into factor and solve
        # C, lower = cho_factor(W + l * DTD)
        # end_time1 = time.time()
        # execution_time = end_time1 - start_time
        # print("Execution time cholesky factor:", execution_time)
        
        # wy = w * y.ravel()
        # z = cho_solve((C, lower), wy)
        # end_time2 = time.time()
        # execution_time = end_time2 - start_time
        # print("Execution time cholesky solve:", execution_time)
        
        # ---------------------------- 1st Trial --------------------
        # Cholesky factor in one line
        # W = np.diag(w) ## not compatible with numpy
        # DTD = D.T @ D
        # C = np.linalg.cholesky(W + l * DTD)
        # z = spsolve(C, spsolve(C.T, w * y))
        # end_time1 = time.time()
        # execution_time = end_time1 - start_time
        # print("Execution time cholesky factor:", execution_time)
        
        # ----------------------------- 2nd Trial -------------------
        #W = diags(w, 0, shape=(m, m))
        #DTD = D.T @ D
        #C = cholesky(W + l * DTD)
        #z = lstsq(C, lstsq(C.T, (np.matmul(w, y))))
        #end_time1 = time.time()
        #execution_time = end_time1 - start_time
        #print("Execution time cholesky factor:", execution_time)
      
        # ----------------------------- 3rd Trial -------------------
        #W = diags(w, 0, shape=(m, m))
        #DTD = D.T @ D
        #C, lower = cho_factor(W + l * DTD.toarray())
        #C = sp.csr_matrix(C,  dtype=np.float32)
        #wy = w * y.ravel()
        #z = cho_solve((C.toarray(), lower), wy)
        #end_time2 = time.time()
        #execution_time = end_time2 - start_time1
        #print(f'Execution time cholesky in loop {i}:', execution_time)
        
        # ----------------------------- 4th Trial -------------------
        #W = diags(w, 0, shape=(m, m))
        #DTD = D.T @ D
        #C, lower = cho_factor(W + l * DTD.toarray())
        #wy = w * y.ravel()
        #z = cho_solve((C, lower), wy)
        #end_time2 = time.time()
        #execution_time = end_time2 - start_time1
        #print(f'Execution time cholesky in loop {i}:', execution_time)
      
        # ----------------------------- 5th Trial -------------------
        W = diags(w, 0, shape=(m, m))
        DTD = D.T @ D
        C = splu(W + l * DTD)
        wy = w * y.ravel()
        z = C.solve(wy)
        #end_time2 = time.time()
        #execution_time = end_time2 - start_time1
        #print(f'Execution time cholesky in loop {i}:', execution_time)
      
        w = p * (y.ravel() > z) + (1 - p) * (y.ravel() < z)
    
    end_time3 = time.time()
    execution_time = end_time3 - start_time
    print(f'Execution time baseline function for M/Z {mzi+35}:', execution_time)
    
    return z
    
for mzi in range(Nmz):
    start_time = time.time()
    signalMZ = mzDataExpt[mzi, :].reshape((-1, 1))
    if signalMZ.max() <= 10:
        lambdaALS = 2e3
    else: 
        lambdaALS = 5e3
        
    pALS = 1e-4

    baselineMZ = baseCorrAls(signalMZ, lambdaALS, pALS)    
    