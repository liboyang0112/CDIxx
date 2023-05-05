cimport numpy as np
from torch import utils, tensor
#import torch.utils.data as data
#from torchvision.transforms import transforms
#from torchvision.datasets import ImageFolder
#from torchvision import transforms, datasets
import os

cdef extern from "cdilmdb.h":
    int initLMDB(const char*)
    void readLMDB(void**, size_t*, void**, size_t*, int*)

np.import_array()
class unetDataLoader(utils.data.Dataset):
    def __init__(self, db_path, chan, row, col, chanl, rowl, coll, device, transform=None, target_transform=None):
        self.row = row;
        self.col = col;
        self.rowl = rowl;
        self.coll = coll;
        self.chan = chan;
        self.chanl = chanl;
        self.len = row*col*chan
        self.lenl = rowl*coll*chanl
        self.pystr = db_path.encode("utf8")
        cdef char* path = self.pystr
        self.length = initLMDB(path)
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        cdef np.npy_intp len = self.len
        cdef np.npy_intp lenl = self.lenl
        cdef size_t datasz = len*sizeof(float)
        cdef size_t labelsz = lenl*sizeof(float)
        cdef int key = index
        cdef int *data;
        cdef int *label;
        readLMDB(<void**>&data, &datasz, <void**>&label, &labelsz, &key);
        imgnp = np.PyArray_SimpleNewFromData(1, &len, np.NPY_FLOAT, <void*>data);
        labnp = np.PyArray_SimpleNewFromData(1, &lenl, np.NPY_FLOAT, <void*>label);
        img = tensor(imgnp).reshape([self.chan, self.row, self.col]).to(self.device);
        lab = tensor(labnp).reshape([self.chanl, self.rowl, self.coll]).to(self.device);
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        return img, lab
    def __len__(self):
        return self.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.pystr.decode("utf8") + ')'
