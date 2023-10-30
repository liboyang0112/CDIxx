import ctypes
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "imgio.h":
    void writeComplexImage(const char* name, void* data, int row, int column);
    void writeFloatImage(const char* name, void* data, int row, int col);
    float *readImage_c(const char* name, int *row, int *col, void* funcptr);

np.import_array()

def writeFloat(path, array):
    fn = path.encode("utf8");
    cdef np.npy_intp* ind = <np.npy_intp*> malloc(2*sizeof(int));
    cdef np.npy_intp* ind0 = <np.npy_intp*> malloc(2*sizeof(int));
    ind[0] = array.shape[0];
    ind[1] = array.shape[1];
    ind0[0] = 0;
    ind0[1] = 0;
    cdef char* fname = fn;
    print("write to", array.dtype, "image", fname);
    if array.dtype == np.single or array.dtype == np.double:
        writeFloatImage(fname, np.PyArray_GetPtr(array, ind0), ind[0], ind[1]);
    elif array.dtype == np.csingle or array.dtype == np.cdouble:
        print("write to complex image", fname);
        writeComplexImage(fname, np.PyArray_GetPtr(array, ind0), ind[0],ind[1]);
    else:
        print("data type not known");
        exit();
    free(ind);

def readImage(path):
    fn = path.encode("utf8");
    cdef char* fname = fn;
    cdef int row, col;
    data = readImage_c(fname, &row, &col, <void*>0);
    cdef np.npy_intp len[2];
    len[0] = row;
    len[1] = col;
    img = np.PyArray_SimpleNewFromData(2, len, np.NPY_FLOAT, data);
    return row, col, img;
