#cython:language_level=3
import ctypes
cimport numpy as np
import numpy as np
from numpy import log2

cdef extern from "FGA.hpp":
  int FGA(int row, int col, int nlambda, double* lambdas, double* spectra, float* data, int iter, int binskip);

np.import_array()

def runFGA(dataArray, lambdas, spectra, iter, binskip):
    return FGA(dataArray.shape[0], dataArray.shape[1], lambdas.shape[0], <double*>np.PyArray_BYTES(lambdas), <double*>np.PyArray_BYTES(spectra), <float*>np.PyArray_BYTES(dataArray.astype(np.float32)), iter, binskip);
