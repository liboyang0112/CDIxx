#!/usr/bin/env python
from torch import utils, tensor
from cythonLoader import cythonLoader

class unetDataLoader(utils.data.Dataset):
    def __init__(self, db_path, chan, row, col, chanl, rowl, coll, device, transform=None, target_transform=None):
        self.loader = cythonLoader(db_path)
        self.path = db_path
        self.chan = chan
        self.row = row
        self.col = col
        self.chanl = chanl
        self.rowl = row
        self.coll = coll
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        imgnp, labnp = self.loader.read(index)
        img = tensor(imgnp).to(self.device).reshape([self.chan, self.row, self.col])
        lab = tensor(labnp).to(self.device).reshape([self.chanl, self.rowl, self.coll])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        return img,lab
    def __len__(self):
        return self.loader.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'
