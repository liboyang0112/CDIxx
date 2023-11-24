import ctypes
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from numpy import log2

cdef extern from "imageFile.h":
    struct imageFile:
        int rows;
        int cols;
cdef extern from "imgio.h":
    void writeComplexImage(const char* name, void* data, int row, int column);
    void writeFloatImage(const char* name, void* data, int row, int col);
    float *readImage_c(const char* name, imageFile* f, void* funcptr);
    void plotPng(const char* label, float* data, char* cache, int rows, int cols, char iscolor);
    void cvtLog(float* data, int nele);

np.import_array()

def writeFloat(path, array):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    print("write to", array.dtype, "image", fname);
    if array.dtype == np.single or array.dtype == np.double:
        writeFloatImage(fname, np.PyArray_BYTES(array), array.shape[0], array.shape[1]);
    elif array.dtype == np.csingle or array.dtype == np.cdouble:
        print("write to complex image", fname);
        writeComplexImage(fname, np.PyArray_BYTES(array), array.shape[0],array.shape[1]);
    else:
        print("data type not known");
        exit();

def writePng(path, array, cache, iscolor, islog = 1):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    if islog:
        cvtLog(<float*>np.PyArray_BYTES(array), array.size);
    if array.dtype == np.single or array.dtype == np.double:
        plotPng(fname, <float*>np.PyArray_BYTES(array), np.PyArray_BYTES(cache), array.shape[0], array.shape[1], iscolor);

def readImage(path):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    cdef imageFile f;
    data = readImage_c(fname, &f, <void*>0);
    cdef np.npy_intp len[2];
    len[0] = f.rows;
    len[1] = f.cols;
    img = np.PyArray_SimpleNewFromData(2, len, np.NPY_FLOAT, data);
    return img;
