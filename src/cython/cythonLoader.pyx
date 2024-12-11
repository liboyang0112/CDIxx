#cython:language_level=3
#We don't want to import pytorch here for compilation performance.
import os
cimport numpy as np
np.import_array()

cdef extern from "cdilmdb.hpp":
    int initLMDB(int* handle, const char*)
    void readLMDB(int handle, int *ndata, void*** data, size_t** data_size, int *keyval);
    int fillLMDB(int handle, int *keyval, int ndata, void** data, size_t* data_size);
    void saveLMDB(int handle);

class cythonLoader:
    def __init__(self, db_path):
        pystr = db_path.encode("utf8")
        cdef char* path = pystr
        cdef int handle = 0;
        self.length = initLMDB(&handle, path)
        self.handle = handle;
        print("Imported dataset @", os.getpid(), " :", db_path, ", containing ", self.length, " samples")
    def read(self, index):
        cdef handle = self.handle;
        cdef int key = index;
        cdef int ndata = 0;
        cdef size_t *data_size[2];
        cdef void **data = NULL;
        readLMDB(handle, &ndata, &data, data_size, &key);
        cdef np.npy_intp len = int(data_size[0][0] / sizeof(float));
        cdef np.npy_intp lenl = int(data_size[1][0] / sizeof(float));
        imgnp = np.PyArray_SimpleNewFromData(1, &len, np.NPY_FLOAT, data[0]);
        labnp = np.PyArray_SimpleNewFromData(1, &lenl, np.NPY_FLOAT, data[1]);
        return imgnp, labnp
    def fill(self, index, datas):
        cdef handle = self.handle;
        cdef int key = index
        cdef size_t data_size[2];
        cdef void *data[2];
        img = datas[0].astype(np.dtype("float32"));
        lab = datas[1].astype(np.dtype("float32"));
        data[0] = np.PyArray_BYTES(img);
        data[1] = np.PyArray_BYTES(lab);
        data_size[0] = img.nbytes
        data_size[1] = lab.nbytes
        fillLMDB(handle, &key, 2, data, data_size);
    def commit(self):
        cdef handle = self.handle;
        saveLMDB(handle)


