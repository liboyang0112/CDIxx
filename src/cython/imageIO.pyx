#cython:language_level=3
import ctypes
cimport numpy as np
import numpy as np
from numpy import log2

cdef extern from "imageFile.hpp":
    struct imageFile:
        char type;
        int rows;
        int cols;
        int nchann;
        int typesize;
cdef extern from "imgio.hpp":
    void writeComplexImage(const char* name, void* data, int row, int column);
    void writeFloatImage(const char* name, void* data, int row, int col);
    float *readImage_c(const char* name, imageFile* f, void* funcptr);
    void plotPng(const char* label, float* data, char* cache, int rows, int cols, char iscolor);
    void cvtLog(float* data, int nele);
    int writePng(const char* png_file_name, void* data , int height, int width, int bit_depth, char colored);

np.import_array()

def writeFloat(path, array):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    print("write to", array.dtype, "image", fname);
    if array.dtype == np.dtype('float32'):
        writeFloatImage(fname, np.PyArray_BYTES(array), array.shape[0], array.shape[1]);
    elif array.dtype == np.dtype('float64'):
        writeFloatImage(fname, np.PyArray_BYTES(array.astype(np.dtype("float32"))), array.shape[0], array.shape[1]);
    elif array.dtype == np.csingle or array.dtype == np.cdouble:
        print("write to complex image", fname);
        writeComplexImage(fname, np.PyArray_BYTES(array), array.shape[0],array.shape[1]);
    else:
        print("data type not known:", array.dtype);

def writePNG(path, array, cache = None, iscolor = 0, islog = 0):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    if islog:
        cvtLog(<float*>np.PyArray_BYTES(array), array.size);
    if array.dtype == np.single or array.dtype == np.double:
        if cache is None:
            print("save type:", array.dtype)
            raise ValueError('Please feed a cache for type conversion!')
        plotPng(fname, <float*>np.PyArray_BYTES(array), np.PyArray_BYTES(cache), array.shape[0], array.shape[1], iscolor);
    if array.dtype == np.uint16:
        writePng(fname, <void*>np.PyArray_BYTES(array), array.shape[0], array.shape[1], 16, 0);
    if array.dtype == np.uint8:
        bytes = <unsigned char*>np.PyArray_BYTES(array)
        writePng(fname, bytes, array.shape[0], array.shape[1], 8, 1);

def readImage(path):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    cdef imageFile f;
    data = readImage_c(fname, &f, <void*>0);
    cdef np.npy_intp len[2];
    len[0] = f.rows;
    len[1] = f.cols;
    if f.type == 4:
        img = np.PyArray_SimpleNewFromData(2, len, np.NPY_CFLOAT, data);
    else:
        img = np.PyArray_SimpleNewFromData(2, len, np.NPY_FLOAT, data);
    return img;
