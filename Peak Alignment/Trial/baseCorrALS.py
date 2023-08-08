# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:45:20 2023

@author: alvak
"""

import numpy as np
import scipy
# from scipy.linalg import lstsq
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import time

#signalMZ = mzDataExpt[0, :].reshape((-1, 1))


## Find baseline
def baseCorrAls(y, l, p):
    start_time = time.time()
    m = len(y)
    D = np.transpose(np.diff(eye(m).toarray(), 2))
    # D = spdiags([-1, 2, 1], [0, 1, 2,], [m-2, m]).toarray()
    w = np.ones(m)
    for i in range(10):
        W = diags(w, 0, shape=(m, m))
        DTD = D.T @ D
        C, lower = cho_factor(W + l * DTD)
        wy = w * y.ravel()
        z = cho_solve((C, lower), wy)
        w = p * (y.ravel() > z) + (1 - p) * (y.ravel() < z)
    return z
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Execution time:", execution_time)
    


