cimport numpy as np
import os

cdef extern from "cdilmdb.hpp":
    int initLMDB(int* handle, const char*)
    void readLMDB(int handle, int *ndata, void*** data, size_t** data_size, int *keyval);

np.import_array()
class cythonLoader:
    def __init__(self, db_path, chan, row, col, chanl, rowl ,coll):
        self.row = row;
        self.col = col;
        self.rowl = rowl;
        self.coll = coll;
        self.chan = chan;
        self.chanl = chanl;
        self.len = self.row*self.col*self.chan
        self.lenl = self.rowl*self.coll*self.chanl
        self.pystr = db_path.encode("utf8")
        cdef char* path = self.pystr
        cdef int handle;
        self.length = initLMDB(&handle, path)
        self.handle = handle;
        print("Imported dataset:", db_path, ", containing ", self.length, " samples")
    def __getitem__(self, index):
        cdef np.npy_intp len = self.len
        cdef np.npy_intp lenl = self.lenl
        cdef size_t datasz = len*sizeof(float)
        cdef size_t labelsz = lenl*sizeof(float)
        cdef handle = self.handle;
        cdef int key = index
        cdef int ndata;
        cdef size_t init_size = self.chan*self.row*self.col*sizeof(float);
        cdef size_t *data_size = &init_size;
        cdef void **data;
        readLMDB(handle, &ndata, &data, &data_size, &key);
        imgnp = np.PyArray_SimpleNewFromData(1, &len, np.NPY_FLOAT, data[0]);
        labnp = np.PyArray_SimpleNewFromData(1, &lenl, np.NPY_FLOAT, data[1]);
        return imgnp, labnp
    def __len__(self):
        return self.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.pystr.decode("utf8") + ')'
